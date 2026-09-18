"""build_backend selection, and the strict vs. lenient fallback contract.

A leaderboard eval (``strict=True``, used by ``as_mteb_encoder``) must never
silently substitute the hashing backend for a model that failed to load — it
should raise instead. The demo/ablation/CodeRetriever path (``strict=False``,
the default) should keep degrading gracefully with a loud warning, so offline
usage never hard-fails on an environment issue.

These tests monkeypatch ``SentenceTransformerBackend`` itself (not anything
inside it), so they never require torch/sentence-transformers to be installed.
"""

from __future__ import annotations

import pytest

from prism.backends import build_backend
from prism.backends.hashing import HashingBackend
from prism.config import BackendConfig


class _BoomBackend:
    """Stands in for a model load that fails for any reason."""

    def __init__(self, **kwargs):
        raise RuntimeError("simulated model load failure")


def _patch_boom(monkeypatch):
    import prism.backends.sentence_transformer as st_mod

    monkeypatch.setattr(st_mod, "SentenceTransformerBackend", _BoomBackend)


def test_hashing_backend_selected_directly():
    backend = build_backend(BackendConfig(kind="hashing", hashing_dim=64))
    assert isinstance(backend, HashingBackend)
    assert backend.dim == 64


def test_lenient_falls_back_to_hashing_with_warning(monkeypatch):
    _patch_boom(monkeypatch)
    cfg = BackendConfig(kind="sentence_transformer", model_name="some/model", hashing_dim=32)

    with pytest.warns(RuntimeWarning, match="Falling back to the deterministic hashing backend"):
        backend = build_backend(cfg, strict=False)

    assert isinstance(backend, HashingBackend)


def test_lenient_is_the_default(monkeypatch):
    _patch_boom(monkeypatch)
    cfg = BackendConfig(kind="sentence_transformer", model_name="some/model")

    with pytest.warns(RuntimeWarning):
        backend = build_backend(cfg)  # strict not passed -> defaults to False

    assert isinstance(backend, HashingBackend)


def test_strict_raises_instead_of_falling_back(monkeypatch):
    _patch_boom(monkeypatch)
    cfg = BackendConfig(kind="sentence_transformer", model_name="some/model")

    with pytest.raises(RuntimeError, match="Refusing to silently substitute"):
        build_backend(cfg, strict=True)


def test_strict_error_names_the_requested_model(monkeypatch):
    _patch_boom(monkeypatch)
    cfg = BackendConfig(kind="sentence_transformer", model_name="jinaai/jina-embeddings-v2-base-code")

    with pytest.raises(RuntimeError, match="jinaai/jina-embeddings-v2-base-code"):
        build_backend(cfg, strict=True)


def test_strict_preserves_original_exception_as_cause(monkeypatch):
    _patch_boom(monkeypatch)
    cfg = BackendConfig(kind="sentence_transformer", model_name="some/model")

    with pytest.raises(RuntimeError) as excinfo:
        build_backend(cfg, strict=True)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "simulated model load failure" in str(excinfo.value.__cause__)


def test_encoder_strict_flag_propagates_to_build_backend(monkeypatch):
    """PrePostPipelineEncoder(strict=True) must not silently degrade either."""
    _patch_boom(monkeypatch)
    from prism.config import PipelineConfig
    from prism.encoder import PrePostPipelineEncoder

    cfg = PipelineConfig(name="t", backend=BackendConfig(kind="sentence_transformer", model_name="some/model"))

    with pytest.raises(RuntimeError, match="Refusing to silently substitute"):
        PrePostPipelineEncoder(cfg, strict=True)

    # Default (lenient) still degrades gracefully.
    with pytest.warns(RuntimeWarning):
        enc = PrePostPipelineEncoder(cfg)
    assert isinstance(enc.backend, HashingBackend)
