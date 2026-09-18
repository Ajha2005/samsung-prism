"""A BM25 sparse-lexical signal over the code corpus.

Dense embeddings capture meaning but can miss exact-symbol matches ("find where
`os.urandom` is called"); BM25 nails those. Fusing the two (see
:mod:`prism.scoring.fusion`) is a cheap, reliable win before anything exotic.

Tokenization is identifier-aware and shared with the hashing backend so the two
signals see code the same way (``camelCase``/``snake_case`` are split into
sub-words).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from prism.backends.hashing import tokenize


class BM25Index:
    """Thin wrapper over ``rank_bm25.BM25Okapi`` with code-aware tokenization."""

    def __init__(self, corpus: Sequence[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rank-bm25 is required for the sparse signal") from exc

        self._corpus_size = len(corpus)
        self._tokenized = [tokenize(doc or "") for doc in corpus]
        # BM25Okapi cannot handle an entirely empty corpus; guard it.
        if not any(self._tokenized):
            self._tokenized = [tok or ["\0"] for tok in self._tokenized] or [["\0"]]
        self._bm25 = BM25Okapi(self._tokenized)

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score per corpus document for ``query``."""
        if self._corpus_size == 0:
            return np.zeros(0, dtype=np.float32)
        tokens = tokenize(query or "")
        if not tokens:
            return np.zeros(self._corpus_size, dtype=np.float32)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)

    def top_k(self, query: str, k: int = 10) -> List[int]:
        scores = self.scores(query)
        if scores.size == 0:
            return []
        k = min(k, scores.size)
        # argpartition for the top-k, then sort just those.
        idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx])].tolist()
