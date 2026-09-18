"""Ablation harness: config -> NDCG@10 -> MRR -> kept/dropped.

The discipline that wins a leaderboard format: change one thing, measure it
against a live baseline, keep or revert. This module makes that mechanical. Feed
it named configs and an evaluator; it produces a table (Markdown + JSON) that is
simultaneously the experiment log, the regression guard, and half the slide
deck.

Two evaluators are provided:
  * :func:`offline_evaluator` — scores a config on a local corpus with the
    deterministic backend (no network); used in CI and for fast iteration.
  * :func:`apps_evaluator` — scores a config on the real CoIR AppsRetrieval split
    (needs the eval extras + network); used for the leaderboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from prism.config import PipelineConfig

# An evaluator maps a config -> {"scores": {...}, "telemetry": {...}}.
Evaluator = Callable[[PipelineConfig], Dict[str, object]]


@dataclass
class AblationRow:
    name: str
    signature: str
    ndcg_at_10: float
    mrr_at_10: float
    delta_ndcg: float
    kept: bool
    telemetry: Dict[str, object] = field(default_factory=dict)


@dataclass
class AblationLog:
    baseline_ndcg: float = 0.0
    rows: List[AblationRow] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "| Config | Signature | NDCG@10 | MRR@10 | ΔNDCG vs base | Latency (ms) | Verdict |",
            "|---|---|---:|---:|---:|---:|:--:|",
        ]
        for r in self.rows:
            lat = r.telemetry.get("mean_query_latency_ms", "")
            verdict = "✅ keep" if r.kept else "↩︎ drop"
            delta = f"{r.delta_ndcg:+.4f}" if r.name != "baseline" else "—"
            lines.append(
                f"| {r.name} | `{r.signature}` | {r.ndcg_at_10:.4f} | {r.mrr_at_10:.4f} "
                f"| {delta} | {lat} | {verdict} |"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "baseline_ndcg_at_10": self.baseline_ndcg,
                "rows": [r.__dict__ for r in self.rows],
            },
            indent=2,
        )

    def save(self, md_path: str, json_path: Optional[str] = None) -> None:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(self.to_markdown() + "\n", encoding="utf-8")
        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(json_path).write_text(self.to_json() + "\n", encoding="utf-8")


def run_ablation(
    configs: Sequence[Tuple[str, PipelineConfig]],
    evaluator: Evaluator,
    *,
    baseline_name: str = "baseline",
    keep_margin: float = 0.0,
) -> AblationLog:
    """Evaluate each config; mark it kept iff it beats baseline by ``keep_margin``.

    The first config named ``baseline_name`` (or the first config overall) sets
    the reference NDCG@10 that every other row is measured against.
    """
    log = AblationLog()
    baseline_ndcg: Optional[float] = None

    for name, config in configs:
        outcome = evaluator(config)
        scores = outcome.get("scores", {})  # type: ignore[assignment]
        telemetry = outcome.get("telemetry", {})  # type: ignore[assignment]
        ndcg = float(scores.get("ndcg_at_10", 0.0))
        mrr = float(scores.get("mrr_at_10", scores.get("mrr_at_100", 0.0)))

        if baseline_ndcg is None and (name == baseline_name or len(log.rows) == 0):
            baseline_ndcg = ndcg
            log.baseline_ndcg = ndcg

        delta = ndcg - (baseline_ndcg or 0.0)
        kept = name == baseline_name or delta > keep_margin
        log.rows.append(
            AblationRow(
                name=name,
                signature=config.describe(),
                ndcg_at_10=ndcg,
                mrr_at_10=mrr,
                delta_ndcg=delta,
                kept=kept,
                telemetry=telemetry if isinstance(telemetry, dict) else {},
            )
        )
    return log


def offline_evaluator(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Evaluator:
    """An evaluator that scores configs on a local corpus, no network needed."""
    from prism.pipeline import CodeRetriever
    from prism.telemetry.metrics import evaluate_retrieval

    def _eval(config: PipelineConfig) -> Dict[str, object]:
        retriever = CodeRetriever(config).index(corpus)
        results = retriever.batch_search(queries, top_k=min(100, len(corpus)))
        scores = evaluate_retrieval(qrels, results, k_values=(1, 3, 5, 10))
        return {"scores": scores, "telemetry": retriever.telemetry.to_dict()}

    return _eval


def apps_evaluator(*, output_dir: Optional[str] = None) -> Evaluator:
    """An evaluator that scores configs on the real CoIR AppsRetrieval split."""
    from prism.eval.mteb_runner import run_apps_retrieval

    def _eval(config: PipelineConfig) -> Dict[str, object]:
        out_path = (
            f"{output_dir}/{config.name}_appsretrieval_results.json"
            if output_dir
            else "results/appsretrieval_results.json"
        )
        payload = run_apps_retrieval(config, output_path=out_path)
        return {"scores": payload.get("scores", {}), "telemetry": {}}

    return _eval
