"""Config loading and validation."""

import json

import pytest

from prism.config import PipelineConfig, QueryConfig, ScoringConfig, load_config


def test_defaults_valid():
    cfg = PipelineConfig()
    assert cfg.name == "baseline"
    assert cfg.backend.kind == "sentence_transformer"


def test_from_dict_nested():
    cfg = load_config(
        {
            "name": "exp1",
            "backend": {"kind": "hashing", "hashing_dim": 512},
            "query": {"hyde_enabled": True, "hyde_weight": 0.3},
        }
    )
    assert cfg.backend.kind == "hashing"
    assert cfg.backend.hashing_dim == 512
    assert cfg.query.hyde_enabled and cfg.query.hyde_weight == 0.3


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        load_config({"nope": 1})


def test_invalid_hyde_weight_rejected():
    with pytest.raises(ValueError):
        QueryConfig(hyde_weight=2.0).validate()


def test_invalid_fusion_rejected():
    with pytest.raises(ValueError):
        ScoringConfig(fusion="magic").validate()


def test_describe_signature():
    cfg = load_config(
        {
            "name": "all",
            "backend": {"kind": "hashing"},
            "query": {"hyde_enabled": True},
            "snippet": {"multiview_enabled": True},
            "scoring": {"hybrid_enabled": True},
            "versioning": {"enabled": True},
        }
    )
    sig = cfg.describe()
    assert "hyde" in sig and "multiview" in sig and "hybrid" in sig and "versioned" in sig


def test_load_from_json_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"name": "f", "backend": {"kind": "hashing"}}), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg.name == "f" and cfg.backend.kind == "hashing"
