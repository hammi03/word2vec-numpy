"""
reproduce.py — deterministic training run that generates the example results
shown in the README.

Fixed configuration
-------------------
  tokens    : 1 000 000  (first 1M tokens of text8)
  dim       : 100
  epochs    : 5
  window    : 5
  neg       : 5
  batch     : 512
  lr_start  : 0.025
  seed      : 42
  min_count : 5

Running this script on any machine should produce nearest-neighbour and
analogy results that are close (though not bit-for-bit identical, since
NumPy's random state depends on the platform) to the examples in the README.

Usage
-----
    python reproduce.py
"""

import os
import pickle

import numpy as np

from word2vec.corpus import download_text8, Vocabulary
from word2vec.model  import Word2Vec
from word2vec.train  import train

# -- Fixed hyperparameters ------------------------------------------------------
TOKENS     = 1_000_000
DIM        = 100
EPOCHS     = 5
WINDOW     = 5
NEG        = 5
BATCH      = 512
LR         = 0.025
MIN_COUNT  = 5
SEED       = 42
OUT_DIR    = "outputs_reproduce"

np.random.seed(SEED)


def main():
    print("=" * 60)
    print("Reproducible word2vec training run")
    print(f"  tokens={TOKENS:,}  dim={DIM}  epochs={EPOCHS}  seed={SEED}")
    print("=" * 60)

    # -- Data ------------------------------------------------------------------
    tokens    = download_text8()[:TOKENS]
    vocab     = Vocabulary(min_count=MIN_COUNT)
    vocab.build(tokens)
    token_ids = vocab.encode(tokens)
    print(f"Vocabulary: {vocab.size:,} words  |  encoded tokens: {len(token_ids):,}")

    # -- Train -----------------------------------------------------------------
    model = Word2Vec(vocab_size=vocab.size, embed_dim=DIM, seed=SEED)
    train(model, vocab, token_ids,
          epochs=EPOCHS, window=WINDOW, neg_samples=NEG,
          batch_size=BATCH, lr_start=LR)

    # -- Save ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/W_in.npy",  model.W_in)
    np.save(f"{OUT_DIR}/W_out.npy", model.W_out)
    with open(f"{OUT_DIR}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # -- Nearest neighbours ----------------------------------------------------
    print("\nNearest neighbours:")
    W     = model.W_in.astype(np.float64)
    norms = np.linalg.norm(W, axis=1) + 1e-8
    W_n   = W / norms[:, np.newaxis]

    for word in ["king", "france", "computer", "dog", "good"]:
        if word not in vocab.word2idx:
            print(f"  {word:>10}  (not in vocabulary)")
            continue
        wid  = vocab.word2idx[word]
        sims = W_n @ W_n[wid]
        sims[wid] = -2.0
        top5 = np.argpartition(sims, -5)[-5:]
        top5 = sorted(top5, key=lambda x: -sims[x])
        print(f"  {word:>10}  ->  {[vocab.idx2word[i] for i in top5]}")

    # -- Analogies -------------------------------------------------------------
    print("\nAnalogies  (a : b :: c : ?):")
    for a, b, c in [("man", "king", "woman"), ("paris", "france", "berlin")]:
        w2i = vocab.word2idx
        if not all(w in w2i for w in [a, b, c]):
            continue
        v    = W_n[w2i[b]] - W_n[w2i[a]] + W_n[w2i[c]]
        v   /= np.linalg.norm(v) + 1e-8
        sims = W_n @ v
        for w in [a, b, c]:
            sims[w2i[w]] = -2.0
        top3 = np.argpartition(sims, -3)[-3:]
        top3 = sorted(top3, key=lambda x: -sims[x])
        print(f"  {a} : {b}  ::  {c} : {[vocab.idx2word[i] for i in top3]}")


if __name__ == "__main__":
    main()
