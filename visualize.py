"""
visualize.py — t-SNE visualisation of learned word embeddings.

Projects the high-dimensional word vectors down to 2-D with t-SNE and
renders a colour-coded scatter plot grouped by semantic category.

Usage
-----
    # train first (or use reproduce.py for a deterministic run)
    python train.py --tokens 1000000 --dim 100 --epochs 5

    # then visualise
    python visualize.py [--out outputs] [--save embeddings.png]

Dependencies (visualisation only — not required for training)
-------------------------------------------------------------
    pip install matplotlib scikit-learn
"""

import argparse
import os
import pickle
import sys

import numpy as np

# ── Word groups to highlight ──────────────────────────────────────────────────

WORD_GROUPS = {
    "Royalty":   ["king", "queen", "prince", "princess", "emperor",
                  "throne", "crown", "dynasty", "royal", "monarch"],
    "Countries": ["france", "germany", "england", "italy", "spain",
                  "russia", "china", "japan", "india", "brazil",
                  "austria", "sweden", "greece", "egypt"],
    "Cities":    ["paris", "berlin", "london", "rome", "madrid",
                  "moscow", "tokyo", "vienna", "athens", "cairo"],
    "Animals":   ["dog", "cat", "horse", "bird", "fish", "wolf",
                  "lion", "tiger", "rabbit", "elephant", "bear", "snake"],
    "Numbers":   ["one", "two", "three", "four", "five",
                  "six", "seven", "eight", "nine", "ten"],
    "Science":   ["physics", "chemistry", "biology", "mathematics",
                  "astronomy", "geology", "genetics", "quantum"],
    "Sports":    ["football", "tennis", "basketball", "cricket",
                  "baseball", "swimming", "athletics", "cycling"],
}

# Vibrant palette suited to a dark background
PALETTE = {
    "Royalty":   "#FFD700",   # gold
    "Countries": "#00E5FF",   # cyan
    "Cities":    "#69B4FF",   # sky blue
    "Animals":   "#FF8C42",   # orange
    "Numbers":   "#A8FF3E",   # lime
    "Science":   "#E040FB",   # purple
    "Sports":    "#FF5370",   # coral
}


def load_embeddings(out_dir: str):
    W = np.load(os.path.join(out_dir, "W_in.npy")).astype(np.float64)
    with open(os.path.join(out_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    return W / norms, vocab


def collect_words(vocab, groups):
    """Return (words, vectors, colors, group_labels) for all words in vocab."""
    words, indices, colors, group_labels = [], [], [], []
    seen = set()
    for group, members in groups.items():
        for w in members:
            if w in vocab.word2idx and w not in seen:
                words.append(w)
                indices.append(vocab.word2idx[w])
                colors.append(PALETTE[group])
                group_labels.append(group)
                seen.add(w)
    return words, indices, colors, group_labels


def run_tsne(vectors: np.ndarray, seed: int = 42):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn not found.  Install it with:  pip install scikit-learn")
        sys.exit(1)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(vectors) - 1),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=2000,
    )
    return tsne.fit_transform(vectors)


def plot(coords, words, colors, group_labels, save_path: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        print("matplotlib not found.  Install it with:  pip install matplotlib")
        sys.exit(1)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # ── Scatter ───────────────────────────────────────────────────────────────
    for group in WORD_GROUPS:
        mask  = [i for i, g in enumerate(group_labels) if g == group]
        if not mask:
            continue
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys,
                   c=PALETTE[group], s=90, alpha=0.92,
                   edgecolors="white", linewidths=0.3,
                   label=group, zorder=3)

    # ── Word labels ───────────────────────────────────────────────────────────
    # Slight random jitter so labels don't stack directly on the dot
    rng = np.random.default_rng(0)
    for i, (word, color) in enumerate(zip(words, colors)):
        dx = rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
        dy = rng.uniform(0.4, 1.2) * rng.choice([-1, 1])
        ax.annotate(
            word,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(coords[i, 0] + dx, coords[i, 1] + dy),
            fontsize=8.5,
            color=color,
            fontweight="bold",
            alpha=0.95,
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.25, lw=0.6),
            bbox=dict(boxstyle="round,pad=0.15", fc="#0D1117", alpha=0.0),
            zorder=4,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.25,
        facecolor="#1C2128",
        edgecolor="#444",
        markerscale=1.4,
        title="Category",
        title_fontsize=10,
    )
    legend.get_title().set_color("white")

    # ── Labels and title ──────────────────────────────────────────────────────
    ax.set_title(
        "Word Embeddings — Skip-gram Word2Vec  ·  t-SNE projection",
        fontsize=15, fontweight="bold", color="white", pad=18,
    )
    ax.set_xlabel("t-SNE dimension 1", fontsize=10, color="#888", labelpad=8)
    ax.set_ylabel("t-SNE dimension 2", fontsize=10, color="#888", labelpad=8)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Subtle grid
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.2, color="#666")

    # Watermark
    fig.text(0.99, 0.01, "github.com/hammi03/word2vec-numpy",
             ha="right", va="bottom", fontsize=8, color="#444", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {save_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",  default="outputs",
                   help="Directory containing W_in.npy and vocab.pkl")
    p.add_argument("--save", default="embeddings.png",
                   help="Output image path")
    args = p.parse_args()

    print("Loading embeddings ...")
    W, vocab = load_embeddings(args.out)

    print("Collecting word groups ...")
    words, indices, colors, group_labels = collect_words(vocab, WORD_GROUPS)
    print(f"  {len(words)} words found across {len(WORD_GROUPS)} categories")

    vectors = W[indices]

    print("Running t-SNE ...")
    coords = run_tsne(vectors)

    print("Rendering plot ...")
    plot(coords, words, colors, group_labels, save_path=args.save)


if __name__ == "__main__":
    main()
