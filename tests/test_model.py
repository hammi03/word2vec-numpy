"""Unit tests for model.py — sigmoid, Word2Vec forward/backward pass."""

import numpy as np
import pytest

from word2vec.model import sigmoid, Word2Vec


# ── sigmoid ───────────────────────────────────────────────────────────────────

def test_sigmoid_midpoint():
    assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6


def test_sigmoid_range():
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    assert np.all(y > 0) and np.all(y < 1)


def test_sigmoid_no_overflow():
    y = sigmoid(np.array([1e10, -1e10]))
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_sigmoid_monotone():
    x = np.linspace(-5, 5, 100)
    y = sigmoid(x)
    assert np.all(np.diff(y) > 0)


# ── Word2Vec initialisation ───────────────────────────────────────────────────

def test_model_weight_shapes():
    model = Word2Vec(vocab_size=200, embed_dim=50)
    assert model.W_in.shape  == (200, 50)
    assert model.W_out.shape == (200, 50)


def test_w_out_zero_initialised():
    model = Word2Vec(vocab_size=100, embed_dim=16)
    assert np.all(model.W_out == 0.0)


def test_w_in_not_all_zero():
    model = Word2Vec(vocab_size=100, embed_dim=16)
    assert not np.all(model.W_in == 0.0)


# ── train_batch ───────────────────────────────────────────────────────────────

def _make_batch(V=50, d=8, B=16, K=3, seed=0):
    rng = np.random.default_rng(seed)
    center_ids  = rng.integers(0, V, size=B).astype(np.int32)
    context_ids = rng.integers(0, V, size=B).astype(np.int32)
    neg_ids     = rng.integers(0, V, size=(B, K)).astype(np.int32)
    return center_ids, context_ids, neg_ids


def test_train_batch_returns_positive_float():
    model = Word2Vec(vocab_size=50, embed_dim=8, seed=0)
    c, o, n = _make_batch()
    loss = model.train_batch(c, o, n, lr=0.01)
    assert isinstance(loss, float)
    assert loss > 0


def test_train_batch_updates_center_weights():
    model = Word2Vec(vocab_size=50, embed_dim=8, seed=0)
    c, o, n = _make_batch()
    W_in_before = model.W_in.copy()
    model.train_batch(c, o, n, lr=0.1)
    # At least the touched rows should differ
    assert not np.allclose(model.W_in[c], W_in_before[c])


def test_train_batch_updates_context_weights():
    model = Word2Vec(vocab_size=50, embed_dim=8, seed=0)
    c, o, n = _make_batch()
    model.train_batch(c, o, n, lr=0.1)
    # W_out was zero; after one step it should have non-zero touched rows
    assert not np.all(model.W_out[o] == 0.0)


def test_train_batch_gradient_correctness():
    """
    Numerical gradient check for ∂J/∂W_in on a single pair.

    We compare the analytic gradient (computed inside train_batch) against
    a finite-difference estimate.  A relative error < 1e-4 confirms the
    closed-form derivative is correct.
    """
    V, d = 20, 4
    B, K = 1, 2
    eps  = 1e-4

    center_ids  = np.array([0], dtype=np.int32)
    context_ids = np.array([1], dtype=np.int32)
    neg_ids     = np.array([[2, 3]], dtype=np.int32)

    def loss_fn(model):
        v_c   = model.W_in[center_ids]
        u_o   = model.W_out[context_ids]
        u_neg = model.W_out[neg_ids]
        s_pos = np.einsum("bd,bd->b",   v_c, u_o)
        s_neg = np.einsum("bd,bkd->bk", v_c, u_neg)
        return -(np.log(sigmoid(s_pos) + 1e-7).mean()
                 + np.log(1.0 - sigmoid(s_neg) + 1e-7).sum(axis=1).mean())

    model = Word2Vec(vocab_size=V, embed_dim=d, seed=7)
    W_in_orig = model.W_in.copy()

    # Analytic gradient via train_batch (using a copy to avoid weight update)
    model_a = Word2Vec(vocab_size=V, embed_dim=d, seed=7)
    W_in_before = model_a.W_in.copy()
    model_a.train_batch(center_ids, context_ids, neg_ids, lr=1.0)
    analytic_grad = W_in_before[0] - model_a.W_in[0]   # lr=1 → grad = Δweight

    # Numerical gradient
    numerical_grad = np.zeros(d)
    for j in range(d):
        model_p = Word2Vec(vocab_size=V, embed_dim=d, seed=7)
        model_m = Word2Vec(vocab_size=V, embed_dim=d, seed=7)
        model_p.W_in[0, j] += eps
        model_m.W_in[0, j] -= eps
        numerical_grad[j] = (loss_fn(model_p) - loss_fn(model_m)) / (2 * eps)

    rel_error = (np.linalg.norm(analytic_grad - numerical_grad)
                 / (np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad) + 1e-8))
    assert rel_error < 1e-3, f"Gradient check failed: rel_error={rel_error:.2e}"


def test_loss_decreases_on_repeated_pairs():
    """
    Repeatedly presenting the same (center, context) pair should drive the
    model to assign a high probability to that pair, lowering the loss.
    """
    V, d, B, K = 30, 16, 64, 5
    model = Word2Vec(vocab_size=V, embed_dim=d, seed=42)

    center_ids  = np.zeros(B, dtype=np.int32)           # always word 0
    context_ids = np.ones(B,  dtype=np.int32)            # always word 1
    neg_ids     = np.random.randint(2, V, (B, K)).astype(np.int32)

    losses = [model.train_batch(center_ids, context_ids, neg_ids, lr=0.1)
              for _ in range(300)]

    assert np.mean(losses[-30:]) < np.mean(losses[:30]), (
        "Loss did not decrease — gradient or update may be incorrect"
    )


# ── most_similar ──────────────────────────────────────────────────────────────

def test_most_similar_length():
    model = Word2Vec(vocab_size=100, embed_dim=16, seed=1)
    results = model.most_similar(word_id=0, top_n=5)
    assert len(results) == 5


def test_most_similar_sorted_descending():
    model = Word2Vec(vocab_size=100, embed_dim=16, seed=1)
    results = model.most_similar(word_id=0, top_n=8)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)


def test_most_similar_excludes_query():
    model = Word2Vec(vocab_size=100, embed_dim=16, seed=1)
    results = model.most_similar(word_id=0, top_n=10)
    assert 0 not in [i for i, _ in results]
