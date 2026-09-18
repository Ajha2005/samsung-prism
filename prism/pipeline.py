"""End-to-end code retriever.

MTEB scores the pure encoder, but the hands-on round runs the *system* on real
queries and asks how fast it is. :class:`CodeRetriever` is that system: it
indexes a corpus, answers queries with dense (and optionally hybrid dense+BM25)
scoring, and reports operational telemetry (latency, index cost). It reuses the
exact same :class:`~prism.encoder.PrePostPipelineEncoder`, so what the demo shows
is what the leaderboard scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from prism.config import PipelineConfig, load_config
from prism.encoder import PrePostPipelineEncoder
from prism.scoring.bm25 import BM25Index
from prism.scoring.fusion import linear_fuse, reciprocal_rank_fusion
from prism.telemetry.metrics import RunTelemetry, Timer

Corpus = Union[Mapping[str, str], Sequence[str]]


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str


class CodeRetriever:
    def __init__(self, config: Optional[PipelineConfig] = None, encoder: Optional[PrePostPipelineEncoder] = None):
        self.config = load_config(config)
        self.encoder = encoder or PrePostPipelineEncoder(self.config)
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Index] = None
        self.telemetry = RunTelemetry()

    def index(self, corpus: Corpus) -> "CodeRetriever":
        """Embed and index a corpus. Accepts {id: text} or a list of texts."""
        if isinstance(corpus, Mapping):
            self.doc_ids = list(corpus.keys())
            self.doc_texts = [corpus[i] for i in self.doc_ids]
        else:
            self.doc_texts = list(corpus)
            self.doc_ids = [str(i) for i in range(len(self.doc_texts))]

        with Timer() as t:
            self.doc_embeddings = self.encoder.encode_texts(self.doc_texts, is_query=False)
        self.telemetry.encode_seconds += t.seconds
        self.telemetry.index_build_seconds = t.seconds
        self.telemetry.num_documents = len(self.doc_ids)

        if self.config.scoring.hybrid_enabled:
            self._bm25 = BM25Index(self.doc_texts)
        return self

    def _dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        return self.doc_embeddings @ query_vec

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        if self.doc_embeddings is None:
            raise RuntimeError("index() must be called before search()")
        query_vec = self.encoder.encode_texts([query], is_query=True)[0]
        dense = self._dense_scores(query_vec)

        if self.config.scoring.hybrid_enabled and self._bm25 is not None:
            sparse = self._bm25.scores(query)
            scores = self._fuse(dense, sparse)
        else:
            scores = dense

        k = min(top_k, len(self.doc_ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchHit(doc_id=self.doc_ids[i], score=float(scores[i]), text=self.doc_texts[i])
            for i in top_idx
        ]

    def _fuse(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        sc = self.config.scoring
        if sc.fusion == "rrf":
            return reciprocal_rank_fusion(dense, sparse, k=sc.rrf_k, dense_weight=sc.dense_weight, sparse_weight=sc.sparse_weight)
        return linear_fuse(dense, sparse, dense_weight=sc.dense_weight, sparse_weight=sc.sparse_weight)

    def batch_search(self, queries: Mapping[str, str], top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """Run many queries; return {query_id: {doc_id: score}} for evaluation."""
        results: Dict[str, Dict[str, float]] = {}
        qids = list(queries.keys())
        query_texts = [queries[q] for q in qids]

        with Timer() as t:
            query_vecs = self.encoder.encode_texts(query_texts, is_query=True)
        self.telemetry.encode_seconds += t.seconds

        with Timer() as t:
            for qid, qtext, qvec in zip(qids, query_texts, query_vecs):
                dense = self._dense_scores(qvec)
                if self.config.scoring.hybrid_enabled and self._bm25 is not None:
                    scores = self._fuse(dense, self._bm25.scores(qtext))
                else:
                    scores = dense
                k = min(top_k, len(self.doc_ids))
                top_idx = np.argpartition(-scores, k - 1)[:k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                results[qid] = {self.doc_ids[i]: float(scores[i]) for i in top_idx}
        self.telemetry.query_seconds += t.seconds
        self.telemetry.num_queries += len(qids)
        return results
