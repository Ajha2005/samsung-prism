"""The MTEB-scored encoder: PrePostPipelineEncoder.

MTEB judges the whole system as an *encoder* — an object that turns queries and
code snippets into comparable vectors. All of our cleverness lives in the pre-
and post-processing wrapped around a small embedding model:

    query  --> analyze --> (HyDE sketch) --> embed --> blend  --> query vector
    snippet --> clean  --> (multi-view)   --> embed --> fuse   --> doc vector

The class implements MTEB 2.x's encoder interface (``AbsEncoder.encode`` taking a
``DataLoader`` of batched inputs, keyed by ``prompt_type``), and also exposes a
plain ``encode_texts`` method so the same logic is usable — and testable —
without MTEB installed.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from prism.backends import EmbeddingBackend, build_backend
from prism.backends.base import l2_normalize
from prism.config import PipelineConfig, load_config
from prism.query.hyde import build_sketcher
from prism.query.preprocess import QueryType, analyze_query
from prism.snippet.multiview import build_views
from prism.snippet.preprocess import chunk_snippet, clean_snippet


class PrePostPipelineEncoder:
    """Encoder with query/snippet pre- and post-processing around a backend.

    Usable two ways:
      * ``encode_texts(texts, is_query=...)`` — direct, framework-free.
      * ``encode(inputs, task_metadata=..., prompt_type=...)`` — the MTEB 2.x
        encoder protocol. Enabled by mixing in MTEB's ``AbsEncoder`` via
        :func:`as_mteb_encoder`, so importing this module never requires mteb.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, backend: Optional[EmbeddingBackend] = None):
        self.config = load_config(config)
        self.backend = backend or build_backend(self.config)
        self.dim = self.backend.dim
        self._sketcher = (
            build_sketcher(self.config.query.hyde_cache_path)
            if self.config.query.hyde_enabled
            else None
        )

    # -- Query side ---------------------------------------------------------
    def _prepare_query(self, text: str) -> tuple[str, Optional[str], float]:
        """Return (query_text, sketch_or_None, hyde_weight_for_this_query)."""
        qcfg = self.config.query
        analysis = analyze_query(
            text,
            normalize=qcfg.normalize,
            extract=qcfg.extract_keywords,
            classify=qcfg.classify_type,
        )
        query_text = analysis.normalized or text or ""
        if not (qcfg.hyde_enabled and self._sketcher is not None):
            return query_text, None, 0.0

        sketch = self._sketcher.sketch(analysis)
        # Trust the sketch more for behavioral queries (the NL<->code gap is
        # widest there) and less for lookups that already name the symbol.
        weight = qcfg.hyde_weight
        if analysis.query_type == QueryType.LOOKUP:
            weight *= 0.5
        elif analysis.query_type in (QueryType.BEHAVIORAL, QueryType.GENERAL):
            weight = min(1.0, weight * 1.15)
        return query_text, sketch, weight

    def _encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [self._prepare_query(t) for t in texts]
        query_texts = [p[0] for p in prepared]
        base = self.backend.embed(query_texts, is_query=True)

        if not any(p[1] for p in prepared):
            return l2_normalize(base) if self.config.backend.normalize else base

        # HyDE: embed the code sketches and blend per-query.
        sketch_texts = [p[1] or "" for p in prepared]
        sketch_emb = self.backend.embed(sketch_texts, is_query=False)
        weights = np.asarray([p[2] for p in prepared], dtype=np.float32)[:, None]
        fused = (1.0 - weights) * base + weights * sketch_emb
        return l2_normalize(fused)

    # -- Snippet side -------------------------------------------------------
    def _encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        scfg = self.config.snippet
        cleaned = [clean_snippet(t) for t in texts]

        if not scfg.multiview_enabled:
            return self._encode_with_chunking(cleaned)

        # Multi-view: embed each view for every doc in one batched call per view
        # (so the backend batches efficiently), then fuse per doc.
        views = [build_views(c) for c in cleaned]
        weights = scfg.view_weights
        fused = np.zeros((len(texts), self.dim), dtype=np.float32)
        total_w = sum(w for w in weights.values() if w > 0) or 1.0
        for view_name, weight in weights.items():
            if weight <= 0:
                continue
            view_texts = [getattr(v, view_name) for v in views]
            emb = l2_normalize(self.backend.embed(view_texts, is_query=False))
            fused += (weight / total_w) * emb
        return l2_normalize(fused)

    def _encode_with_chunking(self, cleaned: Sequence[str]) -> np.ndarray:
        scfg = self.config.snippet
        # Fast path: nothing needs chunking.
        pieces = [chunk_snippet(c, scfg.max_chars, scfg.chunk_overlap) for c in cleaned]
        if all(len(p) == 1 for p in pieces):
            emb = self.backend.embed([p[0] for p in pieces], is_query=False)
            return l2_normalize(emb) if self.config.backend.normalize else emb

        # Flatten, embed once, then max-pool chunk vectors back per document.
        flat: List[str] = []
        spans: List[tuple[int, int]] = []
        for chunks in pieces:
            start = len(flat)
            flat.extend(chunks)
            spans.append((start, len(flat)))
        all_emb = l2_normalize(self.backend.embed(flat, is_query=False))
        out = np.zeros((len(cleaned), self.dim), dtype=np.float32)
        for i, (s, e) in enumerate(spans):
            out[i] = all_emb[s:e].max(axis=0)
        return l2_normalize(out)

    # -- Public, framework-free API ----------------------------------------
    def encode_texts(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Embed a list of strings as queries or documents."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode_queries(texts) if is_query else self._encode_documents(texts)


def _texts_from_batch(batch: Any) -> List[str]:
    """Pull the text list out of an MTEB batch (query/text/corpus shapes)."""
    if isinstance(batch, dict):
        text = batch.get("text")
        if text is None:
            text = batch.get("query")
        title = batch.get("title")
        if title is not None and text is not None:
            # BEIR convention: prepend a non-empty title to the body.
            return [
                (f"{t.strip()}\n{b}" if isinstance(t, str) and t.strip() else b)
                for t, b in zip(title, text)
            ]
        if text is not None:
            return list(text)
    # Fallback: a plain iterable of strings.
    return [str(x) for x in batch]


def as_mteb_encoder(config: Optional[PipelineConfig] = None, *, name: str = "prism-prepost-pipeline"):
    """Build a PrePostPipelineEncoder that also satisfies MTEB's encoder protocol.

    MTEB is imported here (not at module import time) so the offline pipeline and
    unit tests never require torch/mteb. The returned object subclasses MTEB's
    ``AbsEncoder`` and implements ``encode`` over a ``DataLoader`` of batches.
    """
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.model_meta import ModelMeta

    class _MTEBPrePostEncoder(AbsEncoder, PrePostPipelineEncoder):
        def __init__(self, cfg):
            PrePostPipelineEncoder.__init__(self, cfg)
            self.mteb_model_meta = ModelMeta.create_empty(
                overwrites=dict(name=name, revision=self.config.backend.model_name, loader=type(self))
            )

        def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
            from mteb.types import PromptType

            texts: List[str] = []
            for batch in inputs:
                texts.extend(_texts_from_batch(batch))
            is_query = prompt_type == PromptType.query
            return self.encode_texts(texts, is_query=is_query)

    return _MTEBPrePostEncoder(config)
