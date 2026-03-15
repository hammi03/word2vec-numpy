"""Unit tests for corpus.py — Vocabulary and skipgram_pairs."""

import numpy as np
import pytest

from word2vec.corpus import Vocabulary, skipgram_pairs


# ── Vocabulary.build ──────────────────────────────────────────────────────────

def test_vocabulary_builds_word2idx():
    tokens = "the cat sat on the mat the cat".split()
    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)
    assert "the" in vocab.word2idx
    assert "cat" in vocab.word2idx
    assert vocab.size == len(set(tokens))


def test_vocabulary_min_count_filters_rare_words():
    # "the"=3, "cat"=2, "sat"=1, "on"=1, "mat"=1
    tokens = "the cat sat on the mat the cat".split()
    vocab = Vocabulary(min_count=2)
    vocab.build(tokens)
    assert "the" in vocab.word2idx
    assert "cat" in vocab.word2idx
    assert "sat" not in vocab.word2idx
    assert "on"  not in vocab.word2idx


def test_vocabulary_sorted_by_frequency():
    tokens = "a a a b b c".split()
    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)
    # Most frequent word should be at index 0
    assert vocab.idx2word[0] == "a"
    assert vocab.idx2word[1] == "b"
    assert vocab.idx2word[2] == "c"


def test_vocabulary_encode_drops_unknown():
    tokens = "the cat sat".split()
    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)
    ids = vocab.encode(["the", "UNKNOWN", "cat"])
    assert len(ids) == 2
    assert ids[0] == vocab.word2idx["the"]
    assert ids[1] == vocab.word2idx["cat"]


def test_sample_negatives_shape():
    tokens = " ".join([f"word{i}" for i in range(100)] * 3).split()
    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)
    neg = vocab.sample_negatives(n=32, k=5)
    assert neg.shape == (32, 5)
    assert neg.dtype == np.int32


def test_sample_negatives_excludes_context():
    tokens = " ".join([f"word{i}" for i in range(200)] * 3).split()
    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)

    exclude = np.zeros(64, dtype=np.int32)   # always exclude word 0
    neg = vocab.sample_negatives(n=64, k=10, exclude=exclude)

    # No row should contain the excluded word index
    assert not np.any(neg == exclude[:, np.newaxis])


# ── skipgram_pairs ────────────────────────────────────────────────────────────

def test_skipgram_pairs_window1_count():
    # For n tokens with window=1, pairs at offset ±1
    # forward: (n-1) pairs, backward: (n-1) pairs → total 2*(n-1)
    ids   = np.arange(5, dtype=np.int32)
    pairs = skipgram_pairs(ids, window=1)
    assert pairs.shape == (2 * (5 - 1), 2)


def test_skipgram_pairs_window2_count():
    # window=2: offsets 1 and 2
    # offset 1: 2*(n-1) pairs, offset 2: 2*(n-2) pairs
    n     = 6
    ids   = np.arange(n, dtype=np.int32)
    pairs = skipgram_pairs(ids, window=2)
    expected = 2 * (n - 1) + 2 * (n - 2)
    assert pairs.shape == (expected, 2)


def test_skipgram_pairs_contains_expected():
    ids   = np.array([10, 20, 30], dtype=np.int32)
    pairs = skipgram_pairs(ids, window=1)
    pair_set = {tuple(p) for p in pairs.tolist()}
    assert (10, 20) in pair_set
    assert (20, 10) in pair_set
    assert (20, 30) in pair_set
    assert (30, 20) in pair_set
    assert (10, 30) not in pair_set   # distance 2, window=1


def test_skipgram_pairs_dtype():
    ids   = np.arange(10, dtype=np.int32)
    pairs = skipgram_pairs(ids, window=3)
    assert pairs.dtype == np.int32
