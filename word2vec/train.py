"""
train.py — training loop with linear learning-rate decay and progress logging.
"""

import time

import numpy as np

from .corpus import Vocabulary, skipgram_pairs
from .model  import Word2Vec


def train(
    model:       Word2Vec,
    vocab:       Vocabulary,
    token_ids:   np.ndarray,
    *,
    epochs:      int   = 5,
    window:      int   = 5,
    neg_samples: int   = 5,
    batch_size:  int   = 512,
    lr_start:    float = 0.025,
    lr_min:      float = 0.0001,
    log_every:   int   = 5_000,
) -> list:
    """
    Train the Word2Vec model for the given number of epochs.

    Learning rate follows a linear decay from lr_start to lr_min over the
    total number of batches (all epochs combined), matching the schedule used
    in the original word2vec C code.

    Parameters
    ----------
    model       : Word2Vec instance to train in place
    vocab       : Vocabulary used for subsampling and negative sampling
    token_ids   : full encoded corpus as int32 array
    epochs      : number of full passes over the corpus
    window      : maximum context window radius
    neg_samples : negative samples per positive training pair (K)
    batch_size  : number of (center, context) pairs per gradient update
    lr_start    : initial learning rate
    lr_min      : minimum learning rate (floor for decay)
    log_every   : print a status line every this many batches

    Returns
    -------
    loss_log : list of (epoch, batch, avg_loss) tuples
    """
    loss_log        = []
    global_step     = 0
    total_steps_est = None   # estimated after first epoch pair-count is known

    for epoch in range(1, epochs + 1):

        # Re-subsample each epoch: frequent words are dropped with different
        # random draws, giving each epoch a slightly varied training stream.
        ids   = vocab.subsample(token_ids)
        pairs = skipgram_pairs(ids, window)
        np.random.shuffle(pairs)

        n_pairs = len(pairs)

        # Estimate total steps on first epoch (pair count varies due to subsampling)
        if total_steps_est is None:
            total_steps_est = (n_pairs // batch_size) * epochs

        epoch_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        for start in range(0, n_pairs - batch_size + 1, batch_size):
            batch = pairs[start : start + batch_size]
            B     = len(batch)

            center_ids  = batch[:, 0]
            context_ids = batch[:, 1]
            neg_ids     = vocab.sample_negatives(B, neg_samples)

            # Linear LR decay: lr decreases uniformly from lr_start to lr_min
            progress = global_step / max(total_steps_est, 1)
            lr       = max(lr_min, lr_start * (1.0 - progress))

            loss        = model.train_batch(center_ids, context_ids, neg_ids, lr)
            epoch_loss += loss
            n_batches  += 1
            global_step += 1

            if n_batches % log_every == 0:
                avg     = epoch_loss / n_batches
                elapsed = time.time() - t0
                loss_log.append((epoch, n_batches, avg))
                print(
                    f"  epoch {epoch}/{epochs}"
                    f"  step {n_batches:>7,}"
                    f"  loss {avg:.4f}"
                    f"  lr {lr:.6f}"
                    f"  {elapsed:>6.0f}s"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} complete — "
            f"avg_loss={avg_loss:.4f}  pairs={n_pairs:,}  time={elapsed:.1f}s"
        )

    return loss_log
