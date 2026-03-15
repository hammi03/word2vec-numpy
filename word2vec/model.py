"""
model.py — Skip-gram Word2Vec with Negative Sampling in pure NumPy.

Mathematical background
-----------------------
The model maintains two embedding matrices:
  W_in  : (V, d)  — centre-word  (input)  embeddings
  W_out : (V, d)  — context-word (output) embeddings

For a training triple (centre word c, context word o, K noise words n_1..n_K):

  Loss
  ----
  J = -log σ(u_o · v_c)  −  Σ_{k=1}^{K} log σ(−u_{n_k} · v_c)

  where  v_c = W_in[c],  u_o = W_out[o],  u_{n_k} = W_out[n_k],
         σ(x) = 1 / (1 + e^{-x}).

  The first term maximises the dot product between centre and true context.
  The second term minimises the dot products with K randomly sampled noise words.

  Gradient derivation
  -------------------
  Let s_pos = u_o · v_c  and  s_k = u_{n_k} · v_c.

  Using  d/dx log σ(x) = 1 − σ(x)  and  d/dx log σ(−x) = −σ(x):

    ∂J/∂v_c     = (σ(s_pos) − 1) · u_o  +  Σ_k σ(s_k) · u_{n_k}
    ∂J/∂u_o     = (σ(s_pos) − 1) · v_c
    ∂J/∂u_{n_k} =  σ(s_k)        · v_c

  Intuitively:
    - If the model already predicts the context well (σ(s_pos) ≈ 1), the
      gradient ∂J/∂v_c from the positive pair is near zero — no update needed.
    - If a noise word has a high dot product with the centre (σ(s_k) ≈ 1),
      the gradient pushes v_c and u_{n_k} apart.
"""

import numpy as np


# ── Numerics ──────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; clips to [-500, 500] to avoid overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# ── Model ─────────────────────────────────────────────────────────────────────

class Word2Vec:
    """
    Skip-gram word2vec with negative sampling.

    Parameters
    ----------
    vocab_size : int   — number of words in the vocabulary
    embed_dim  : int   — dimensionality of the word vectors
    seed       : int   — RNG seed for reproducibility
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Initialise W_in uniformly in (−0.5/d, 0.5/d), matching the
        # original C implementation of word2vec.
        scale       = 0.5 / embed_dim
        self.W_in   = rng.uniform(-scale, scale,
                                  (vocab_size, embed_dim)).astype(np.float32)

        # W_out is zero-initialised; it learns purely from gradient updates.
        self.W_out  = np.zeros((vocab_size, embed_dim), dtype=np.float32)

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    # ── Core training step ────────────────────────────────────────────────────

    def train_batch(
        self,
        center_ids:  np.ndarray,   # int32 (B,)
        context_ids: np.ndarray,   # int32 (B,)
        neg_ids:     np.ndarray,   # int32 (B, K)
        lr:          float,
    ) -> float:
        """
        Forward pass → loss → gradients → SGD update for one mini-batch.

        All operations are vectorised over the batch dimension B.

        Parameters
        ----------
        center_ids  : indices of centre words
        context_ids : indices of true context words
        neg_ids     : indices of K noise words per training pair
        lr          : current learning rate

        Returns
        -------
        mean loss over the batch (float, for logging)
        """
        B, K = neg_ids.shape

        # ── 1. Look up embeddings ─────────────────────────────────────────────
        v_c   = self.W_in[center_ids]            # (B, d)  centre vectors
        u_o   = self.W_out[context_ids]          # (B, d)  context vectors
        u_neg = self.W_out[neg_ids]              # (B, K, d) noise vectors

        # ── 2. Dot products ───────────────────────────────────────────────────
        # s_pos[b]    = u_o[b] · v_c[b]
        # s_neg[b, k] = u_neg[b, k] · v_c[b]
        s_pos = np.einsum("bd,bd->b",   v_c, u_o)     # (B,)
        s_neg = np.einsum("bd,bkd->bk", v_c, u_neg)   # (B, K)

        # ── 3. Sigmoid activations ────────────────────────────────────────────
        sig_pos = sigmoid(s_pos)   # (B,)    probability assigned to true context
        sig_neg = sigmoid(s_neg)   # (B, K)  probability assigned to noise words

        # ── 4. Loss ───────────────────────────────────────────────────────────
        # J = −log σ(s_pos) − Σ_k log(1 − σ(s_neg_k))
        eps  = 1e-7   # numerical safety inside log
        loss = -(
            np.log(sig_pos + eps).mean()
            + np.log(1.0 - sig_neg + eps).sum(axis=1).mean()
        )

        # ── 5. Gradients ──────────────────────────────────────────────────────
        # δ_pos[b]    = σ(s_pos[b]) − 1          (error on positive pair)
        # δ_neg[b, k] = σ(s_neg[b, k])           (error on each noise pair)
        delta_pos = (sig_pos - 1.0)[:, np.newaxis]    # (B, 1)
        delta_neg = sig_neg                            # (B, K)

        # ∂J/∂v_c  = δ_pos * u_o  +  Σ_k δ_neg_k * u_{neg_k}
        grad_v_c = (
            delta_pos * u_o                                   # (B, d)
            + np.einsum("bk,bkd->bd", delta_neg, u_neg)      # (B, d)
        )

        # ∂J/∂u_o  = δ_pos * v_c
        grad_u_o = delta_pos * v_c                            # (B, d)

        # ∂J/∂u_{neg_k}  = δ_neg_k * v_c
        grad_u_neg = (
            delta_neg[:, :, np.newaxis]                       # (B, K, 1)
            * v_c[:, np.newaxis, :]                           # (B, 1, d)
        )                                                     # (B, K, d)

        # ── 6. SGD update ─────────────────────────────────────────────────────
        # np.add.at is unbuffered: if the same index appears multiple times
        # in a batch (e.g. a frequent word), each gradient is applied
        # independently — unlike W_in[ids] -= g which only keeps the last write.
        np.add.at(self.W_in,  center_ids,  -lr * grad_v_c)
        np.add.at(self.W_out, context_ids, -lr * grad_u_o)
        np.add.at(self.W_out, neg_ids,     -lr * grad_u_neg)

        return float(loss)

    # ── Inference helpers ─────────────────────────────────────────────────────

    def get_vector(self, word_id: int) -> np.ndarray:
        """Return the L2-normalised centre-word vector for word_id."""
        v = self.W_in[word_id].astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-8)

    def most_similar(self, word_id: int, top_n: int = 10) -> list:
        """
        Return the top_n most similar words by cosine similarity
        in the centre-word embedding space.
        """
        query  = self.get_vector(word_id)
        norms  = np.linalg.norm(self.W_in, axis=1).astype(np.float64) + 1e-8
        sims   = self.W_in.astype(np.float64) @ query / norms
        sims[word_id] = -2.0   # exclude the query itself
        top    = np.argpartition(sims, -top_n)[-top_n:]
        return sorted([(int(i), float(sims[i])) for i in top], key=lambda x: -x[1])
