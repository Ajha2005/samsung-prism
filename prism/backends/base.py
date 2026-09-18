"""The embedding backend protocol."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Turns a list of strings into an ``(n, dim)`` float32 matrix.

    Implementations must be deterministic for a fixed input (so eval is
    reproducible) and CPU-friendly. ``embed`` is the only required method; the
    pipeline handles all pre/post-processing around it.
    """

    dim: int

    def embed(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        """Embed ``texts``.

        ``is_query`` lets a backend apply an asymmetric prompt/prefix (some
        models expect a different instruction for queries vs. passages). Most of
        the pipeline's asymmetry lives above the backend, so backends may ignore
        it.
        """
        ...


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization; safe on zero rows."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (matrix / norms).astype(np.float32)
