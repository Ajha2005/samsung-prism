"""BM25 sparse signal and dense+sparse fusion."""

import numpy as np

from prism.scoring.bm25 import BM25Index
from prism.scoring.fusion import linear_fuse, reciprocal_rank_fusion


def test_bm25_ranks_exact_symbol_match_first():
    corpus = [
        "def reverse(items): return items[::-1]",
        "def sort(a): return sorted(a)",
        "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
    ]
    idx = BM25Index(corpus)
    scores = idx.scores("gcd greatest common divisor")
    assert int(np.argmax(scores)) == 2


def test_bm25_empty_query_is_zero():
    idx = BM25Index(["a b c", "d e f"])
    assert np.allclose(idx.scores(""), 0.0)


def test_bm25_top_k():
    idx = BM25Index(["alpha beta", "beta gamma", "gamma delta"])
    top = idx.top_k("gamma", k=2)
    assert len(top) == 2
    assert set(top) <= {0, 1, 2}


def test_rrf_rewards_agreement():
    dense = np.array([0.9, 0.1, 0.5])
    sparse = np.array([0.2, 0.1, 0.8])
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused.shape == dense.shape
    # doc 2 is 3rd/1st -> should beat doc 1 which is last in both.
    assert fused[2] > fused[1]


def test_linear_fuse_normalizes_scales():
    dense = np.array([100.0, 0.0])  # large scale
    sparse = np.array([0.0, 1.0])   # small scale
    fused = linear_fuse(dense, sparse, dense_weight=1.0, sparse_weight=1.0)
    # After min-max both contribute equally: doc0 gets dense=1, doc1 gets sparse=1.
    assert np.isclose(fused[0], fused[1])


def test_fusion_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(np.zeros(3), np.zeros(2))
