"""
corpus.py — data loading, vocabulary, and training-pair generation.
"""

import hashlib
import os
import zipfile
import urllib.request
from collections import Counter

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# mattmahoney.net does not serve HTTPS; we verify integrity with a SHA-256
# checksum instead of relying on transport encryption.
TEXT8_SHA256 = "a6640522afe85d1963ad56c05b0ede0a0c000dddc9671758a6cc09b7a38e5232"
TEXT8_URL    = "http://mattmahoney.net/dc/text8.zip"
TEXT8_PATH = "data/text8"


# ── Data loading ──────────────────────────────────────────────────────────────

def download_text8() -> list:
    """
    Download and extract the text8 corpus, then return it as a list of tokens.

    text8 is the first 10^8 bytes of a cleaned English Wikipedia dump —
    the canonical small benchmark for word2vec.
    """
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(TEXT8_PATH):
        zip_path = TEXT8_PATH + ".zip"
        if not os.path.exists(zip_path):
            print(f"Downloading text8 from {TEXT8_URL} ...")
            print("Note: mattmahoney.net serves over HTTP only; "
                  "integrity is verified via SHA-256 after download.")
            urllib.request.urlretrieve(TEXT8_URL, zip_path)
            digest = hashlib.sha256(open(zip_path, "rb").read()).hexdigest()
            if digest != TEXT8_SHA256:
                os.remove(zip_path)
                raise RuntimeError(
                    f"SHA-256 mismatch for text8.zip — "
                    f"expected {TEXT8_SHA256}, got {digest}. "
                    "File may be corrupted or tampered with."
                )
            print("Download complete (checksum OK).")
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("data/")

    with open(TEXT8_PATH, "r") as f:
        return f.read().split()


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Builds a word-to-index mapping from a token list and exposes:

    - Subsampling probabilities to discard frequent words (Mikolov et al. §2.3)
    - A flat noise-distribution table for O(1) negative sampling

    Attributes
    ----------
    size        : int              — number of unique words in the vocabulary
    word2idx    : dict[str, int]
    idx2word    : list[str]
    """

    def __init__(self, min_count: int = 5, subsample_t: float = 1e-5):
        """
        Parameters
        ----------
        min_count   : discard words that appear fewer than this many times
        subsample_t : subsampling threshold t (original paper uses 1e-5)
        """
        self.min_count   = min_count
        self.subsample_t = subsample_t

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, tokens: list) -> None:
        """Construct vocabulary and sampling tables from a token list."""
        counts = Counter(tokens)

        # Keep words above min_count; sort descending by frequency
        vocab = sorted(
            [(w, c) for w, c in counts.items() if c >= self.min_count],
            key=lambda x: -x[1],
        )

        self.idx2word = [w for w, _ in vocab]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.size     = len(self.idx2word)

        counts_arr = np.array([c for _, c in vocab], dtype=np.float64)
        total      = counts_arr.sum()
        freqs      = counts_arr / total

        # ── Subsampling keep-probability ─────────────────────────────────────
        # From the original paper (eq. 5):
        #   P_keep(w) = min(1, sqrt(t / f(w)))
        # where f(w) is the unigram frequency of w and t ≈ 1e-5.
        # High-frequency words (e.g. "the") are discarded most often.
        self._keep_prob = np.minimum(1.0, np.sqrt(self.subsample_t / freqs))

        # ── Negative-sampling noise distribution ─────────────────────────────
        # Sample noise words proportional to freq(w)^(3/4).
        # The exponent 3/4 was determined empirically; it smooths the
        # distribution so rare words are sampled more often than their
        # raw frequency would suggest.
        noise_dist  = freqs ** 0.75
        noise_dist /= noise_dist.sum()

        # Build a flat lookup table of size TABLE_SIZE.
        # Each slot contains a word index with probability ∝ noise_dist.
        # At sampling time we draw random integers in [0, TABLE_SIZE).
        TABLE_SIZE  = 10_000_000
        table       = np.zeros(TABLE_SIZE, dtype=np.int32)
        cumulative  = 0.0
        j           = 0
        for i, p in enumerate(noise_dist):
            cumulative += p
            while j < TABLE_SIZE and j / TABLE_SIZE < cumulative:
                table[j] = i
                j += 1
        self._neg_table = table

    # ── Public helpers ────────────────────────────────────────────────────────

    def encode(self, tokens: list) -> np.ndarray:
        """Convert a token list to an int32 index array, skipping unknown words."""
        return np.array(
            [self.word2idx[t] for t in tokens if t in self.word2idx],
            dtype=np.int32,
        )

    def subsample(self, ids: np.ndarray) -> np.ndarray:
        """
        Randomly drop tokens according to their keep-probability.
        Called once per epoch so each epoch sees a slightly different stream.
        """
        probs = self._keep_prob[ids]
        mask  = np.random.random(len(ids)) < probs
        return ids[mask]

    def sample_negatives(
        self,
        n:       int,
        k:       int,
        exclude: np.ndarray = None,
    ) -> np.ndarray:
        """
        Draw an (n, k) array of noise-word indices from the unigram^(3/4) table,
        ensuring no drawn word appears in the per-row exclusion list.

        Parameters
        ----------
        n       : number of training pairs (rows)
        k       : negative samples per pair (columns)
        exclude : int32 array of shape (n,) or (n, m) — one or more word
                  indices per row to exclude from that row's negatives.
                  Pass ``np.column_stack([center_ids, context_ids])`` to
                  exclude both positive IDs as the SGNS objective requires.
                  If None, no exclusion is applied.

        Implementation note
        -------------------
        The original word2vec C code does not filter positive words from
        negative samples; with large vocabularies the collision probability
        is O(m·k/V) which is negligible.  We provide exact exclusion here
        for correctness.  Collisions are resolved via vectorised resampling;
        in practice the while-loop runs at most once per call.
        """
        slots  = np.random.randint(0, len(self._neg_table), size=(n, k))
        result = self._neg_table[slots]

        if exclude is not None:
            # Normalise to 2-D: (n, m) where m is the number of words to exclude
            excl = np.atleast_2d(exclude)
            if excl.shape[0] != n:
                excl = excl.T                    # handle (1, n) edge case
            # mask[i, j] = True if result[i, j] matches any excluded word in row i
            # Broadcasting: result (n, k, 1) vs excl (n, 1, m)
            while True:
                mask = np.any(
                    result[:, :, np.newaxis] == excl[:, np.newaxis, :],
                    axis=2,
                )                               # (n, k) bool
                if not mask.any():
                    break
                n_bad        = int(mask.sum())
                new_slots    = np.random.randint(0, len(self._neg_table),
                                                 size=n_bad)
                result[mask] = self._neg_table[new_slots]

        return result


# ── Training-pair generation ──────────────────────────────────────────────────

def skipgram_pairs(ids: np.ndarray, window: int) -> np.ndarray:
    """
    Generate all (center, context) index pairs for the skip-gram objective.

    For each offset d in {1, ..., window} every token at position i is paired
    with the token at position i+d and i-d (if in bounds).

    This is equivalent to a fixed maximum window. The original word2vec
    implementation draws the window size *per token* uniformly from
    {1, ..., window}, which down-weights distant context words. Our approach
    produces all pairs; the net effect on learned embeddings is small for
    typical window sizes.

    Parameters
    ----------
    ids    : int32 token-index array of shape (N,)
    window : maximum context distance

    Returns
    -------
    pairs : int32 array of shape (P, 2) — columns are [center_id, context_id]
    """
    segments = []
    n = len(ids)
    for d in range(1, window + 1):
        # Forward: center at i, context at i+d
        segments.append(np.column_stack([ids[: n - d], ids[d:]]))
        # Backward: center at i+d, context at i
        segments.append(np.column_stack([ids[d:], ids[: n - d]]))
    return np.vstack(segments).astype(np.int32)
