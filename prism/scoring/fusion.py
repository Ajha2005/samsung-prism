"""Fuse dense and sparse rankings.

Two standard, well-behaved strategies:

  * Reciprocal Rank Fusion (RRF): combines *rank positions*, so it needs no
    score calibration between the two systems and is robust to their different
    score scales. This is the default.
  * Linear: min-max normalize each score vector to [0, 1], then take a weighted
    sum. Simple and interpretable when the scales are comparable.
"""

from __future__ import annotations

import numpy as np


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks (rank 1 = highest score). Ties broken by index."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def reciprocal_rank_fusion(
    dense: np.ndarray, sparse: np.ndarray, k: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0
) -> np.ndarray:
    """Weighted RRF over two score vectors. Higher output = more relevant."""
    dense = np.asarray(dense, dtype=np.float64)
    sparse = np.asarray(sparse, dtype=np.float64)
    if dense.shape != sparse.shape:
        raise ValueError("dense and sparse score vectors must have the same shape")
    if dense.size == 0:
        return dense.astype(np.float32)
    d_ranks = _ranks_from_scores(dense)
    s_ranks = _ranks_from_scores(sparse)
    fused = dense_weight / (k + d_ranks) + sparse_weight / (k + s_ranks)
    return fused.astype(np.float32)


def _min_max(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.astype(np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def linear_fuse(
    dense: np.ndarray, sparse: np.ndarray, dense_weight: float = 1.0, sparse_weight: float = 0.5
) -> np.ndarray:
    """Min-max normalize each signal, then weighted-sum. Higher = more relevant."""
    dense = np.asarray(dense)
    sparse = np.asarray(sparse)
    if dense.shape != sparse.shape:
        raise ValueError("dense and sparse score vectors must have the same shape")
    if dense.size == 0:
        return dense.astype(np.float32)
    return (dense_weight * _min_max(dense) + sparse_weight * _min_max(sparse)).astype(np.float32)
