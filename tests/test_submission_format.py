"""The submission JSON's file format is a hard external contract.

The problem statement's own reference snippet writes the leaderboard artifact
as ``json.dump(task_result.to_dict(), f, indent=2)`` — the RAW MTEB TaskResult
dict, no wrapper. Whatever the screening pipeline parses almost certainly
expects exactly that shape (top-level ``scores``, ``task_name``, ...). These
tests need no network/mteb — they exercise ``write_result_files`` directly
with a fake ``outcome``, so the contract is locked in without a live run.
"""

from __future__ import annotations

import json

from prism.config import BackendConfig, PipelineConfig
from prism.eval.mteb_runner import write_result_files

# Shaped like a real task_result.to_dict() (see mteb.TaskResult), trimmed.
_FAKE_RAW_RESULT = {
    "dataset_revision": "abc123",
    "task_name": "AppsRetrieval",
    "mteb_version": "2.21.0",
    "scores": {
        "test": [
            {
                "ndcg_at_10": 0.0662,
                "mrr_at_10": 0.0561,
                "main_score": 0.0662,
                "hf_subset": "default",
                "languages": ["eng-Latn", "python-Code"],
            }
        ]
    },
    "evaluation_time": 123.4,
    "kg_co2_emissions": None,
    "date": "2026-09-19",
    "evaluation_phases": {},
}


def _fake_outcome():
    return {
        "scores": {"ndcg_at_10": 0.0662, "mrr_at_10": 0.0561},
        "result": _FAKE_RAW_RESULT,
    }


def test_submission_file_is_the_raw_task_result_verbatim(tmp_path):
    out_path = tmp_path / "appsretrieval_results.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    write_result_files(_fake_outcome(), cfg, str(out_path))

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == _FAKE_RAW_RESULT, "submission file must equal task_result.to_dict() exactly, no wrapper"


def test_submission_file_has_no_added_top_level_keys(tmp_path):
    out_path = tmp_path / "appsretrieval_results.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    write_result_files(_fake_outcome(), cfg, str(out_path))

    written = json.loads(out_path.read_text(encoding="utf-8"))
    # None of our own bookkeeping keys leak into the submission artifact.
    for forbidden in ("config", "config_signature", "mteb_result", "main_score_ndcg_at_10", "submission_file"):
        assert forbidden not in written


def test_submission_file_scores_are_reachable_the_documented_way(tmp_path):
    """The problem statement's own path to the score: scores.test[0].ndcg_at_10."""
    out_path = tmp_path / "appsretrieval_results.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    write_result_files(_fake_outcome(), cfg, str(out_path))

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["scores"]["test"][0]["ndcg_at_10"] == 0.0662
    assert written["task_name"] == "AppsRetrieval"


def test_debug_sidecar_carries_our_bookkeeping_separately(tmp_path):
    out_path = tmp_path / "appsretrieval_results.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    payload = write_result_files(_fake_outcome(), cfg, str(out_path), write_debug_sidecar=True)

    debug_path = tmp_path / "appsretrieval_results.debug.json"
    assert debug_path.exists()
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug["main_score_ndcg_at_10"] == 0.0662
    assert debug["config_signature"] == cfg.describe()
    assert payload["debug_file"] == str(debug_path)


def test_debug_sidecar_can_be_disabled(tmp_path):
    out_path = tmp_path / "appsretrieval_results.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    payload = write_result_files(_fake_outcome(), cfg, str(out_path), write_debug_sidecar=False)

    assert "debug_file" not in payload
    assert not (tmp_path / "appsretrieval_results.debug.json").exists()


def test_output_path_with_multiple_dots_in_stem_still_produces_sane_sidecar_name(tmp_path):
    """A name like 'foo.v2.json' must keep its full stem in the sidecar name."""
    out_path = tmp_path / "foo.v2.json"
    cfg = PipelineConfig(name="baseline", backend=BackendConfig(kind="hashing"))

    payload = write_result_files(_fake_outcome(), cfg, str(out_path))

    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == _FAKE_RAW_RESULT
    assert payload["debug_file"].endswith("foo.v2.debug.json")
