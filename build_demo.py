"""
build_demo.py — export pre-computed t-SNE coordinates and nearest neighbours
for the interactive web demo (docs/index.html).

Usage
-----
    python build_demo.py [--out outputs] [--n-words 500]

Writes docs/data.js, which is loaded by docs/index.html.
"""

import argparse
import json
import os
import pickle

import numpy as np

STOPWORDS = {
    "the", "of", "and", "in", "a", "to", "was", "is", "for", "on",
    "are", "as", "with", "by", "at", "an", "it", "its", "from", "or",
    "be", "that", "this", "which", "he", "his", "had", "but", "not",
    "have", "been", "were", "they", "their", "also", "one", "two",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}

WORD_GROUPS = {
    "Royalty":   ["king", "queen", "prince", "princess", "emperor",
                  "throne", "crown", "dynasty", "royal", "monarch"],
    "Countries": ["france", "germany", "england", "italy", "spain",
                  "russia", "china", "japan", "india", "brazil",
                  "austria", "sweden", "greece", "egypt", "poland"],
    "Cities":    ["paris", "berlin", "london", "rome", "madrid",
                  "moscow", "tokyo", "vienna", "athens", "cairo",
                  "amsterdam", "prague", "budapest"],
    "Animals":   ["dog", "cat", "horse", "bird", "fish", "wolf",
                  "lion", "tiger", "rabbit", "elephant", "bear",
                  "snake", "eagle", "shark", "whale"],
    "Numbers":   ["zero", "one", "two", "three", "four", "five",
                  "six", "seven", "eight", "nine", "ten",
                  "hundred", "thousand", "million"],
    "Science":   ["physics", "chemistry", "biology", "mathematics",
                  "astronomy", "geology", "genetics", "quantum",
                  "nuclear", "theory", "molecule", "particle"],
    "Sports":    ["football", "tennis", "basketball", "cricket",
                  "baseball", "swimming", "athletics", "cycling",
                  "rugby", "hockey", "volleyball", "boxing"],
}

PALETTE = {
    "Royalty":   "#FFD700",
    "Countries": "#00E5FF",
    "Cities":    "#69B4FF",
    "Animals":   "#FF8C42",
    "Numbers":   "#A8FF3E",
    "Science":   "#E040FB",
    "Sports":    "#FF5370",
    "Other":     "#888888",
}


def select_words(vocab, n_total: int) -> list:
    """Return up to n_total word indices: all WORD_GROUP members first,
    then top-frequency words filtered for length and non-stopword."""
    w2i = vocab.word2idx
    selected = {}

    # Priority 1: semantic groups
    for group, members in WORD_GROUPS.items():
        for w in members:
            if w in w2i and w not in selected:
                selected[w] = group

    # Priority 2: frequent words (sorted by index = frequency rank)
    remaining = n_total - len(selected)
    for w, idx in sorted(w2i.items(), key=lambda x: x[1]):
        if remaining <= 0:
            break
        if w in selected:
            continue
        if len(w) < 4 or len(w) > 14:
            continue
        if w in STOPWORDS or not w.isalpha():
            continue
        selected[w] = "Other"
        remaining -= 1

    return selected   # {word: category}


def compute_tsne(W: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(W) - 1),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=2000,
    )
    return tsne.fit_transform(W)


def nearest_neighbours(W_norm: np.ndarray, idx: int, candidates: list,
                       top_n: int = 10) -> list:
    """Top-n nearest neighbours of word `idx` within the candidate index list."""
    q    = W_norm[idx]
    sims = W_norm[candidates] @ q          # (n_candidates,)
    order = np.argsort(sims)[::-1]
    result = []
    for o in order:
        if candidates[o] == idx:
            continue
        result.append({"word": None, "sim": float(sims[o])})
        if len(result) == top_n:
            break
    return result, order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",     default="outputs")
    p.add_argument("--n-words", type=int, default=500)
    args = p.parse_args()

    print("Loading embeddings ...")
    W_in = np.load(os.path.join(args.out, "W_in.npy")).astype(np.float64)
    with open(os.path.join(args.out, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-8
    W_norm = W_in / norms

    print("Selecting words ...")
    word_cat = select_words(vocab, args.n_words)
    words    = list(word_cat.keys())
    indices  = [vocab.word2idx[w] for w in words]
    print(f"  {len(words)} words selected")

    print("Running t-SNE ...")
    vecs   = W_norm[indices]
    coords = compute_tsne(vecs)

    # Normalise to [-1, 1]
    for dim in range(2):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        coords[:, dim] = 2 * (coords[:, dim] - lo) / (hi - lo + 1e-8) - 1

    print("Computing nearest neighbours ...")
    records = []
    for i, (w, idx) in enumerate(zip(words, indices)):
        sims    = W_norm[indices] @ W_norm[idx]
        order   = np.argsort(sims)[::-1]
        neighbors = []
        for o in order:
            if indices[o] == idx:
                continue
            neighbors.append({"word": words[o], "sim": round(float(sims[o]), 4)})
            if len(neighbors) == 10:
                break

        records.append({
            "word":      w,
            "x":         round(float(coords[i, 0]), 4),
            "y":         round(float(coords[i, 1]), 4),
            "category":  word_cat[w],
            "neighbors": neighbors,
        })

    # Write docs/data.js
    os.makedirs("docs", exist_ok=True)
    out_path = os.path.join("docs", "data.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("const EMBED_DATA = ")
        json.dump({
            "words":   records,
            "palette": PALETTE,
            "groups":  list(WORD_GROUPS.keys()) + ["Other"],
        }, f, ensure_ascii=False)
        f.write(";\n")

    print(f"Wrote {out_path}  ({os.path.getsize(out_path)//1024} KB)")


if __name__ == "__main__":
    main()
