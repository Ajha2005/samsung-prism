"""End-to-end MTEB integration on a local synthetic task (no network).

Skipped automatically where ``mteb`` is not installed (e.g. the minimal offline
CI). When present, this exercises the exact code path the leaderboard run uses:
``mteb.evaluate`` drives our encoder over a retrieval task and scores it with the
official trec_eval metrics.
"""

import math

import pytest

mteb = pytest.importorskip("mteb")


def _data():
    corpus = {
        "d1": "def reverse_list(items):\n    return items[::-1]",
        "d2": "def bubble_sort(a):\n    for i in range(len(a)):\n        for j in range(len(a)-1):\n            if a[j]>a[j+1]: a[j],a[j+1]=a[j+1],a[j]\n    return a",
        "d3": "def is_palindrome(s):\n    return s == s[::-1]",
        "d4": "def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a",
    }
    queries = {
        "q1": "reverse a list",
        "q2": "bubble sort an array",
        "q3": "check if a string is a palindrome",
        "q4": "nth fibonacci number",
    }
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}, "q4": {"d4": 1}}
    return corpus, queries, qrels


def test_encoder_satisfies_mteb_protocol():
    from prism.config import BackendConfig, PipelineConfig
    from prism.encoder import as_mteb_encoder

    enc = as_mteb_encoder(PipelineConfig(name="t", backend=BackendConfig(kind="hashing")))
    assert isinstance(enc, mteb.EncoderProtocol)
    assert enc.mteb_model_meta is not None


def test_full_mteb_evaluate_offline():
    from prism.config import BackendConfig, PipelineConfig
    from prism.encoder import as_mteb_encoder
    from prism.eval.mteb_runner import run_task
    from prism.eval.synthetic_task import build_synthetic_task

    corpus, queries, qrels = _data()
    enc = as_mteb_encoder(PipelineConfig(name="t", backend=BackendConfig(kind="hashing", hashing_dim=4096)))
    task = build_synthetic_task(corpus, queries, qrels)
    out = run_task(enc, task, split="test")

    assert "ndcg_at_10" in out["scores"]
    assert 0.0 <= out["scores"]["ndcg_at_10"] <= 1.0


def test_mteb_scores_match_our_metrics():
    """MTEB's trec_eval and our metrics agree on the same rankings."""
    from prism.config import BackendConfig, PipelineConfig
    from prism.encoder import as_mteb_encoder
    from prism.eval.mteb_runner import run_task
    from prism.eval.synthetic_task import build_synthetic_task
    from prism.pipeline import CodeRetriever
    from prism.telemetry.metrics import evaluate_retrieval

    corpus, queries, qrels = _data()
    cfg = PipelineConfig(name="t", backend=BackendConfig(kind="hashing", hashing_dim=4096))

    mteb_out = run_task(as_mteb_encoder(cfg), build_synthetic_task(corpus, queries, qrels), split="test")
    ours = evaluate_retrieval(
        qrels,
        CodeRetriever(cfg).index(corpus).batch_search(queries, top_k=len(corpus)),
        k_values=(10,),
    )
    assert math.isclose(mteb_out["scores"]["ndcg_at_10"], ours["ndcg_at_10"], rel_tol=1e-4, abs_tol=1e-4)
