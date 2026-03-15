"""
evaluate.py — post-training evaluation: word similarity and analogy tests.

Usage
-----
    python evaluate.py [--out outputs]
"""

import argparse
import pickle

import numpy as np


def load(out_dir: str = "outputs"):
    W_in = np.load(f"{out_dir}/W_in.npy").astype(np.float64)
    with open(f"{out_dir}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    # L2-normalise all rows once for fast cosine similarity
    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-8
    W_norm = W_in / norms
    return W_norm, vocab


def cosine_sim(W: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity between word i and word j (W must be L2-normalised)."""
    return float(W[i] @ W[j])


def analogy(W: np.ndarray, vocab, a: str, b: str, c: str, top_n: int = 5) -> list:
    """
    Solve the analogy  a : b  ::  c : ?  via vector arithmetic:

        d = normalise( v_b − v_a + v_c )

    The query vector d points in the direction of 'b relative to a',
    shifted to the neighbourhood of c.  We then find the nearest
    neighbours of d, excluding a, b, c themselves.

    This method is sometimes called '3CosAdd'.
    """
    w2i = vocab.word2idx
    if not all(w in w2i for w in [a, b, c]):
        missing = [w for w in [a, b, c] if w not in w2i]
        return [f"(words not in vocab: {missing})"]

    v = W[w2i[b]] - W[w2i[a]] + W[w2i[c]]
    v = v / (np.linalg.norm(v) + 1e-8)

    sims = W @ v
    for excl in [a, b, c]:
        sims[w2i[excl]] = -2.0

    top = np.argpartition(sims, -top_n)[-top_n:]
    return [(vocab.idx2word[i], float(sims[i]))
            for i in sorted(top, key=lambda x: -sims[x])]


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    W, vocab = load(args.out)
    w2i      = vocab.word2idx

    # ── Word similarity ───────────────────────────────────────────────────────
    section("Word Similarity  (cosine similarity)")
    pairs = [
        ("man",    "woman"),
        ("king",   "queen"),
        ("paris",  "france"),
        ("london", "england"),
        ("dog",    "cat"),
        ("good",   "bad"),
        ("good",   "great"),
        ("fast",   "slow"),
    ]
    for a, b in pairs:
        if a in w2i and b in w2i:
            sim = cosine_sim(W, w2i[a], w2i[b])
            print(f"  sim({a:>8}, {b:<8}) = {sim:+.4f}")

    # ── Nearest neighbours ────────────────────────────────────────────────────
    section("Nearest Neighbours")
    probes = ["king", "france", "linux", "dog", "terrible", "zero"]
    for word in probes:
        if word not in w2i:
            continue
        sims  = W @ W[w2i[word]]
        sims[w2i[word]] = -2.0
        top5  = np.argpartition(sims, -5)[-5:]
        words = [vocab.idx2word[i] for i in sorted(top5, key=lambda x: -sims[x])]
        print(f"  {word:>10}  →  {words}")

    # ── Word analogies ────────────────────────────────────────────────────────
    section("Word Analogies  ( a : b  ::  c : ? )")
    analogies = [
        ("man",    "king",    "woman"),     # → queen
        ("paris",  "france",  "berlin"),    # → germany
        ("good",   "better",  "bad"),       # → worse
        ("slow",   "slower",  "fast"),      # → faster
        ("france", "french",  "germany"),   # → german
    ]
    for a, b, c in analogies:
        results = analogy(W, vocab, a, b, c, top_n=3)
        print(f"  {a} : {b}  ::  {c} : {[w for w, _ in results if isinstance(w, str)]}")


if __name__ == "__main__":
    main()
