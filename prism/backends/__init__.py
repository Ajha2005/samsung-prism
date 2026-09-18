"""Embedding backends: text -> vectors.

Backends are pluggable so the pipeline logic is independent of the model. The
competition uses :class:`~prism.backends.sentence_transformer.SentenceTransformerBackend`;
offline tests and CI use :class:`~prism.backends.hashing.HashingBackend`, which
needs no network or model download.
"""

from prism.backends.base import EmbeddingBackend
from prism.backends.hashing import HashingBackend

__all__ = ["EmbeddingBackend", "HashingBackend", "build_backend"]


def build_backend(config, *, strict: bool = False) -> EmbeddingBackend:
    """Instantiate the backend named by ``config.backend.kind``.

    When ``strict`` is False (the default — used by the demo, ablation, and the
    standalone :class:`~prism.pipeline.CodeRetriever`), a failure to load the
    sentence-transformer stack or model falls back to the deterministic hashing
    backend with a loud warning, so offline/dev usage never hard-fails on an
    environment issue.

    When ``strict`` is True (used by :func:`prism.encoder.as_mteb_encoder`, i.e.
    the actual leaderboard eval path), that fallback is disabled and the
    original exception is re-raised instead. A leaderboard run must never
    silently substitute a different backend than the one requested and report a
    misleading score under the requested model's name — better to fail loudly
    than to hand back a number nobody asked for.
    """
    from prism.config import PipelineConfig

    bc = config.backend if isinstance(config, PipelineConfig) else config

    if bc.kind == "hashing":
        return HashingBackend(dim=bc.hashing_dim, normalize=bc.normalize)

    if bc.kind == "sentence_transformer":
        try:
            from prism.backends.sentence_transformer import SentenceTransformerBackend

            return SentenceTransformerBackend(
                model_name=bc.model_name,
                normalize=bc.normalize,
                batch_size=bc.batch_size,
                device=bc.device,
                max_seq_length=bc.max_seq_length,
                query_prompt=bc.query_prompt,
                passage_prompt=bc.passage_prompt,
                trust_remote_code=bc.trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            if strict:
                raise RuntimeError(
                    f"Could not load sentence-transformers model {bc.model_name!r} "
                    f"({type(exc).__name__}: {exc}). Refusing to silently substitute "
                    f"a different backend for a leaderboard eval — fix the model load "
                    f"(missing dependency, trust_remote_code, network) and retry."
                ) from exc

            import warnings

            warnings.warn(
                f"Could not load sentence-transformers backend "
                f"({type(exc).__name__}: {exc}). Falling back to the deterministic "
                f"hashing backend. Results will NOT match the leaderboard model.",
                RuntimeWarning,
                stacklevel=2,
            )
            return HashingBackend(dim=bc.hashing_dim, normalize=bc.normalize)

    raise ValueError(f"Unknown backend kind: {bc.kind!r}")
