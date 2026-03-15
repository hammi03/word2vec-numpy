"""
train.py — entry point for training a Skip-gram Word2Vec model on text8.

Usage
-----
    python train.py [options]

    python train.py --tokens 5000000 --dim 100 --epochs 5
    python train.py --tokens 1000000 --dim 50  --epochs 3   # quick smoke test

Options
-------
  --tokens    INT    Number of tokens to use from the corpus  [default: 5_000_000]
  --dim       INT    Embedding dimensionality                 [default: 100]
  --epochs    INT    Training epochs                          [default: 5]
  --window    INT    Max context-window radius                [default: 5]
  --neg       INT    Negative samples per positive pair       [default: 5]
  --batch     INT    Mini-batch size                          [default: 512]
  --lr        FLOAT  Initial learning rate                    [default: 0.025]
  --min-count INT    Min word frequency to include in vocab   [default: 5]
  --out       STR    Output directory for saved embeddings    [default: outputs]
"""

import argparse
import os
import pickle

import numpy as np

from word2vec.corpus import download_text8, Vocabulary
from word2vec.model  import Word2Vec
from word2vec.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train Skip-gram Word2Vec on text8.")
    p.add_argument("--tokens",    type=int,   default=5_000_000)
    p.add_argument("--dim",       type=int,   default=100)
    p.add_argument("--epochs",    type=int,   default=5)
    p.add_argument("--window",    type=int,   default=5)
    p.add_argument("--neg",       type=int,   default=5)
    p.add_argument("--batch",     type=int,   default=512)
    p.add_argument("--lr",        type=float, default=0.025)
    p.add_argument("--min-count", type=int,   default=5)
    p.add_argument("--out",       type=str,   default="outputs")
    return p.parse_args()


def quick_eval(model: Word2Vec, vocab: Vocabulary) -> None:
    """Print nearest neighbours for a handful of probe words."""
    probes = ["king", "france", "computer", "dog", "good", "one"]
    print("\nNearest neighbours (cosine similarity in W_in space):")
    for word in probes:
        if word not in vocab.word2idx:
            continue
        wid     = vocab.word2idx[word]
        neighbours = model.most_similar(wid, top_n=6)
        words   = [vocab.idx2word[i] for i, _ in neighbours]
        print(f"  {word:>10}  ->  {words}")


def main():
    args = parse_args()

    # -- 1. Load corpus --------------------------------------------------------
    print("-" * 60)
    tokens = download_text8()
    tokens = tokens[: args.tokens]
    print(f"Corpus      : {len(tokens):,} tokens")

    # -- 2. Build vocabulary ---------------------------------------------------
    vocab = Vocabulary(min_count=args.min_count)
    vocab.build(tokens)
    print(f"Vocabulary  : {vocab.size:,} unique words")

    token_ids = vocab.encode(tokens)
    print(f"Encoded     : {len(token_ids):,} token indices")

    # -- 3. Initialise model ---------------------------------------------------
    model = Word2Vec(vocab_size=vocab.size, embed_dim=args.dim)
    n_params = vocab.size * args.dim * 2
    print(f"Model       : V={vocab.size}, d={args.dim}, params={n_params:,}")
    print("-" * 60)

    # -- 4. Train --------------------------------------------------------------
    train(
        model, vocab, token_ids,
        epochs      = args.epochs,
        window      = args.window,
        neg_samples = args.neg,
        batch_size  = args.batch,
        lr_start    = args.lr,
    )

    # -- 5. Quick sanity check -------------------------------------------------
    quick_eval(model, vocab)

    # -- 6. Save ---------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "W_in.npy"),  model.W_in)
    np.save(os.path.join(args.out, "W_out.npy"), model.W_out)
    with open(os.path.join(args.out, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    print(f"\nEmbeddings saved to {args.out}/")


if __name__ == "__main__":
    main()
