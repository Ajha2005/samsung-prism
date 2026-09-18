"""Encoder shapes/normalization and end-to-end retrieval on the synthetic set."""

import numpy as np
import pytest

from prism.config import (
    BackendConfig,
    PipelineConfig,
    QueryConfig,
    ScoringConfig,
    SnippetConfig,
)
from prism.data import load_synthetic_corpus
from prism.encoder import PrePostPipelineEncoder
from prism.pipeline import CodeRetriever
from prism.telemetry.metrics import evaluate_retrieval


def _cfg(**kw):
    kw.setdefault("backend", BackendConfig(kind="hashing", hashing_dim=8192))
    return PipelineConfig(name="t", **kw)


def test_encode_shapes_and_norm():
    enc = PrePostPipelineEncoder(_cfg())
    docs = enc.encode_texts(["def a(): return 1", "def b(): return 2"], is_query=False)
    qs = enc.encode_texts(["make a function"], is_query=True)
    assert docs.shape == (2, enc.dim)
    assert qs.shape == (1, enc.dim)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)


def test_encode_empty():
    enc = PrePostPipelineEncoder(_cfg())
    assert enc.encode_texts([], is_query=True).shape == (0, enc.dim)


def test_multiview_changes_representation():
    base = PrePostPipelineEncoder(_cfg(snippet=SnippetConfig(multiview_enabled=False)))
    mv = PrePostPipelineEncoder(_cfg(snippet=SnippetConfig(multiview_enabled=True)))
    code = ["def get_user_name(user):\n    return user.name"]
    assert not np.allclose(base.encode_texts(code), mv.encode_texts(code))


def test_hyde_changes_query_representation():
    plain = PrePostPipelineEncoder(_cfg(query=QueryConfig(hyde_enabled=False)))
    hyde = PrePostPipelineEncoder(_cfg(query=QueryConfig(hyde_enabled=True, hyde_weight=0.7)))
    q = ["compute the greatest common divisor"]
    assert not np.allclose(plain.encode_texts(q, is_query=True), hyde.encode_texts(q, is_query=True))


def test_end_to_end_retrieval_beats_random():
    corpus, queries, qrels = load_synthetic_corpus()
    retriever = CodeRetriever(_cfg()).index(corpus)
    results = retriever.batch_search(queries, top_k=10)
    m = evaluate_retrieval(qrels, results, k_values=(1, 10))
    # Deterministic lexical backend should comfortably beat chance on this set.
    assert m["ndcg_at_10"] > 0.5
    assert m["recall_at_10"] > 0.7


def test_hybrid_runs_and_scores():
    corpus, queries, qrels = load_synthetic_corpus()
    cfg = _cfg(scoring=ScoringConfig(hybrid_enabled=True, fusion="rrf"))
    retriever = CodeRetriever(cfg).index(corpus)
    results = retriever.batch_search(queries, top_k=10)
    m = evaluate_retrieval(qrels, results, k_values=(10,))
    assert 0.0 <= m["ndcg_at_10"] <= 1.0


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        CodeRetriever(_cfg()).search("q")


def test_telemetry_populated():
    corpus, queries, _ = load_synthetic_corpus()
    r = CodeRetriever(_cfg()).index(corpus)
    r.batch_search(queries, top_k=5)
    t = r.telemetry.to_dict()
    assert t["num_documents"] == len(corpus)
    assert t["num_queries"] == len(queries)
    assert t["mean_query_latency_ms"] >= 0.0
