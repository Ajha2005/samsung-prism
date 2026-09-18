"""Metrics match trec_eval semantics on hand-computable cases."""

import math

from prism.telemetry.metrics import (
    average_precision_at_k,
    evaluate_retrieval,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_perfect_ranking():
    ranked = ["gold", "x", "y"]
    rel = {"gold": 1}
    assert ndcg_at_k(ranked, rel, 10) == 1.0
    assert mrr_at_k(ranked, rel, 10) == 1.0
    assert precision_at_k(ranked, rel, 1) == 1.0
    assert recall_at_k(ranked, rel, 10) == 1.0


def test_gold_at_rank_two():
    ranked = ["x", "gold", "y"]
    rel = {"gold": 1}
    assert math.isclose(ndcg_at_k(ranked, rel, 10), 1 / math.log2(3), rel_tol=1e-9)
    assert mrr_at_k(ranked, rel, 10) == 0.5
    assert precision_at_k(ranked, rel, 2) == 0.5


def test_no_relevant_in_topk():
    ranked = ["x", "y", "gold"]
    rel = {"gold": 1}
    assert mrr_at_k(ranked, rel, 2) == 0.0
    assert precision_at_k(ranked, rel, 2) == 0.0
    assert recall_at_k(ranked, rel, 2) == 0.0
    # But recall@3 finds it.
    assert recall_at_k(ranked, rel, 3) == 1.0


def test_map_multiple_relevant():
    ranked = ["a", "x", "b"]  # relevant a@1, b@3
    rel = {"a": 1, "b": 1}
    # AP = (1/1 + 2/3) / 2
    assert math.isclose(average_precision_at_k(ranked, rel, 10), (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_evaluate_retrieval_shapes():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d2": 0.2, "d1": 0.8}}
    m = evaluate_retrieval(qrels, results, k_values=(1, 10))
    assert set(m) >= {"ndcg_at_1", "ndcg_at_10", "mrr_at_10", "recall_at_10", "precision_at_1", "map_at_10"}
    # q1 perfect, q2 gold at rank 2 -> mrr@10 = (1 + 0.5)/2
    assert math.isclose(m["mrr_at_10"], 0.75, rel_tol=1e-9)


def test_ties_broken_deterministically():
    # Equal scores -> ranked by doc_id ascending, so metrics are stable.
    results = {"q": {"b": 0.5, "a": 0.5}}
    qrels = {"q": {"a": 1}}
    m = evaluate_retrieval(qrels, results, k_values=(1,))
    # 'a' sorts before 'b' on tie, so gold is rank 1.
    assert m["precision_at_1"] == 1.0
