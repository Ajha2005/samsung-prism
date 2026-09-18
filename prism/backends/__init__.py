"""Embedding backends: text -> vectors.

Backends are pluggable so the pipeline logic is independent of the model. The
competition uses :class:`~prism.backends.sentence_transformer.SentenceTransformerBackend`;
offline tests and CI use :class:`~prism.backends.hashing.HashingBackend`, which
needs no network or model download.
"""

from prism.backends.base import EmbeddingBackend
from prism.backends.hashing import HashingBackend

__all__ = ["EmbeddingBackend", "HashingBackend", "build_backend"]


def build_backend(config) -> EmbeddingBackend:
    """Instantiate the backend named by ``config.backend.kind``.

    Falls back to the hashing backend (with a warning) if the sentence-transformer
    stack or the model is unavailable, so a run never hard-fails on environment
    issues — it degrades to a reproducible offline embedder instead.
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
