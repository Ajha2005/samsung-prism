"""Telemetry: IR metrics, timing, and the ablation log."""

from prism.telemetry.metrics import (
    RunTelemetry,
    evaluate_retrieval,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "RunTelemetry",
    "evaluate_retrieval",
    "ndcg_at_k",
    "mrr_at_k",
    "precision_at_k",
    "recall_at_k",
]
