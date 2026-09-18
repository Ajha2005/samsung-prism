"""Information-retrieval metrics and run telemetry.

These implementations follow trec_eval / pytrec_eval semantics (the same engine
MTEB uses), so numbers computed here on our own synthetic tasks line up with the
official leaderboard metrics. We reimplement them (rather than only parsing
MTEB's output) so the whole pipeline is testable offline, and so the demo can
report precision@k, recall, latency, and index cost — the operational metrics
the hands-on round asks for.

Conventions:
  * ``qrels``   : {query_id: {doc_id: relevance}} with relevance > 0 == relevant.
  * ``results`` : {query_id: {doc_id: score}} (higher score == more relevant).
  * NDCG uses linear gain and a log2(rank+1) discount (1-based rank), matching
    trec_eval's ``ndcg_cut``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, float]]
Results = Mapping[str, Mapping[str, float]]


def _ranked_doc_ids(doc_scores: Mapping[str, float]) -> List[str]:
    """Documents sorted by score desc; ties broken by doc_id for determinism."""
    return [d for d, _ in sorted(doc_scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain with 1-based ranks and log2(rank+1) discount."""
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    gains = [float(relevant.get(doc, 0.0)) for doc in ranked[:k]]
    actual = dcg(gains)
    ideal_rels = sorted((float(v) for v in relevant.values()), reverse=True)[:k]
    ideal = dcg(ideal_rels)
    if ideal <= 0:
        return 0.0
    return actual / ideal


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if relevant.get(doc, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if relevant.get(doc, 0.0) > 0)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    total = sum(1 for v in relevant.values() if v > 0)
    if total == 0:
        return 0.0
    hits = sum(1 for doc in ranked[:k] if relevant.get(doc, 0.0) > 0)
    return hits / total


def average_precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    total_relevant = sum(1 for v in relevant.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc in enumerate(ranked[:k], start=1):
        if relevant.get(doc, 0.0) > 0:
            hits += 1
            score += hits / rank
    return score / min(total_relevant, k)


def evaluate_retrieval(qrels: Qrels, results: Results, k_values: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    """Average NDCG/MRR/precision/recall/MAP at each k over all judged queries.

    Metric keys mirror MTEB (e.g. ``ndcg_at_10``, ``mrr_at_10``) so results are
    directly comparable to the leaderboard.
    """
    metrics: Dict[str, float] = {}
    query_ids = [q for q in qrels if any(v > 0 for v in qrels[q].values())]
    if not query_ids:
        return {f"{m}_at_{k}": 0.0 for m in ("ndcg", "mrr", "precision", "recall", "map") for k in k_values}

    for k in k_values:
        ndcg_sum = mrr_sum = prec_sum = rec_sum = map_sum = 0.0
        for qid in query_ids:
            relevant = qrels[qid]
            ranked = _ranked_doc_ids(results.get(qid, {}))
            ndcg_sum += ndcg_at_k(ranked, relevant, k)
            mrr_sum += mrr_at_k(ranked, relevant, k)
            prec_sum += precision_at_k(ranked, relevant, k)
            rec_sum += recall_at_k(ranked, relevant, k)
            map_sum += average_precision_at_k(ranked, relevant, k)
        n = len(query_ids)
        metrics[f"ndcg_at_{k}"] = ndcg_sum / n
        metrics[f"mrr_at_{k}"] = mrr_sum / n
        metrics[f"precision_at_{k}"] = prec_sum / n
        metrics[f"recall_at_{k}"] = rec_sum / n
        metrics[f"map_at_{k}"] = map_sum / n
    return metrics


@dataclass
class RunTelemetry:
    """Operational metrics for one retrieval run (the hands-on round cares)."""

    num_queries: int = 0
    num_documents: int = 0
    index_build_seconds: float = 0.0
    index_rebuild_seconds: float = 0.0
    encode_seconds: float = 0.0
    query_seconds: float = 0.0
    quality: Dict[str, float] = field(default_factory=dict)

    @property
    def queries_per_second(self) -> float:
        return self.num_queries / self.query_seconds if self.query_seconds > 0 else 0.0

    @property
    def mean_query_latency_ms(self) -> float:
        return 1000.0 * self.query_seconds / self.num_queries if self.num_queries else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_queries": self.num_queries,
            "num_documents": self.num_documents,
            "index_build_seconds": round(self.index_build_seconds, 4),
            "index_rebuild_seconds": round(self.index_rebuild_seconds, 4),
            "encode_seconds": round(self.encode_seconds, 4),
            "query_seconds": round(self.query_seconds, 4),
            "mean_query_latency_ms": round(self.mean_query_latency_ms, 3),
            "queries_per_second": round(self.queries_per_second, 2),
            "quality": {k: round(v, 5) for k, v in self.quality.items()},
        }


class Timer:
    """Context-manager stopwatch: ``with Timer() as t: ...; t.seconds``."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start
