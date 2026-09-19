"""Run MTEB evaluations and emit the submission JSON.

The leaderboard consumes an MTEB results JSON on the CoIR ``AppsRetrieval`` test
split. :func:`run_apps_retrieval` produces exactly that (`appsretrieval_results.json`)
from a frozen config, and :func:`run_task` runs any MTEB task (used by the
offline synthetic verification too).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from prism.config import PipelineConfig, load_config


def extract_scores(task_result: Any, split: str = "test") -> Dict[str, float]:
    """Pull the flat metric dict (ndcg_at_10, mrr_at_10, ...) from a TaskResult."""
    # MTEB TaskResult exposes get_score / scores; be tolerant of shape drift.
    scores: Dict[str, float] = {}
    try:
        raw = task_result.scores  # {split: [ {subset scores...}, ... ]}
    except AttributeError:
        raw = task_result
    split_scores = None
    if isinstance(raw, dict):
        split_scores = raw.get(split) or next(iter(raw.values()), None)
    if isinstance(split_scores, list) and split_scores:
        entry = split_scores[0]
        if isinstance(entry, dict):
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    scores[k] = float(v)
    return scores


def _task_result_to_dict(task_result: Any) -> Dict[str, Any]:
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(task_result, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    # Last resort: json round-trip via the object's __dict__.
    return json.loads(json.dumps(getattr(task_result, "__dict__", {}), default=str))


def run_task(
    encoder: Any,
    task: Any,
    *,
    output_dir: Optional[str] = None,
    encode_kwargs: Optional[dict] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Evaluate ``encoder`` on ``task`` and return {scores, result_dict}."""
    import mteb

    model_result = mteb.evaluate(
        encoder,
        [task],
        encode_kwargs=encode_kwargs or {"batch_size": 32},
        overwrite_strategy="always",
        show_progress_bar=False,
        raise_error=True,
    )
    # ModelResult -> list of TaskResult.
    task_results: List[Any] = list(getattr(model_result, "task_results", model_result))
    task_result = task_results[0]
    scores = extract_scores(task_result, split=split)
    result_dict = _task_result_to_dict(task_result)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "task_result.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    return {"scores": scores, "result": result_dict}


def write_result_files(
    outcome: Dict[str, Any],
    config: PipelineConfig,
    output_path: str,
    *,
    write_debug_sidecar: bool = True,
) -> Dict[str, Any]:
    """Write the submission artifact (+ optional debug sidecar) from an outcome.

    ``output_path`` gets EXACTLY ``outcome["result"]`` (== ``task_result.to_dict()``,
    the same object the problem statement's own reference snippet writes out via
    ``json.dump(task_result.to_dict(), f, indent=2)``) — no wrapper, no added
    keys, so it matches whatever the screening pipeline parses byte-for-byte.

    A separate ``<output_path minus .json>.debug.json`` sidecar (on by default)
    carries our own config + flattened scores, for local comparison/ablation
    convenience only — never upload the sidecar as the submission artifact.

    Split out from :func:`run_apps_retrieval` so this file-format contract is
    unit-testable without a live MTEB/HuggingFace run.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outcome["result"], indent=2, default=str), encoding="utf-8")

    payload = {
        "task": "AppsRetrieval",
        "split": "test",
        "config": config.to_dict(),
        "config_signature": config.describe(),
        "main_score_ndcg_at_10": outcome["scores"].get("ndcg_at_10"),
        "mrr_at_10": outcome["scores"].get("mrr_at_10"),
        "scores": outcome["scores"],
        "submission_file": str(out),
    }
    if write_debug_sidecar:
        debug_path = out.parent / f"{out.stem}.debug.json"
        debug_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        payload["debug_file"] = str(debug_path)
    return payload


def run_apps_retrieval(
    config: Optional[PipelineConfig] = None,
    *,
    output_path: str = "results/appsretrieval_results.json",
    encode_kwargs: Optional[dict] = None,
    write_debug_sidecar: bool = True,
) -> Dict[str, Any]:
    """Run the real CoIR AppsRetrieval eval and write the submission JSON.

    Requires network access to download the dataset + model from HuggingFace
    (the judges' environment). See :func:`write_result_files` for the exact
    output file contract.
    """
    import mteb

    from prism.encoder import as_mteb_encoder

    config = load_config(config)
    encoder = as_mteb_encoder(config)
    task = mteb.get_task("AppsRetrieval")

    outcome = run_task(encoder, task, encode_kwargs=encode_kwargs, split="test")
    return write_result_files(outcome, config, output_path, write_debug_sidecar=write_debug_sidecar)
