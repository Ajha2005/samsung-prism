"""A versioned vector index with cheap rebuilds.

P1 requires retrieving across multiple versions of a codebase, so the index has
to rebuild quickly when a version changes. The trick is a *content-addressed
embedding cache*: an embedding is keyed by the hash of the exact text that
produced it, so rebuilding for a new version only re-encodes the snippets that
actually changed. Unchanged snippets (the overwhelming majority between adjacent
versions) are served from cache for free.

The index is designed versioned from the start rather than bolted on, and it
persists to a single ``.npz`` + JSON manifest so a frozen checkout reloads the
same vectors without recomputation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

from prism.encoder import PrePostPipelineEncoder
from prism.telemetry.metrics import Timer


def content_hash(text: str) -> str:
    return hashlib.blake2b((text or "").encode("utf-8"), digest_size=16).hexdigest()


class VersionedIndex:
    """Embeddings for a corpus that evolves across versions.

    A *version* is a mapping ``{doc_id: text}``. Adding a version encodes only
    the docs whose content changed since anything already cached.
    """

    def __init__(self, encoder: PrePostPipelineEncoder):
        self.encoder = encoder
        self.dim = encoder.dim
        # content_hash -> embedding row (the reusable cache).
        self._emb_cache: Dict[str, np.ndarray] = {}
        # version -> {doc_id: content_hash}
        self.versions: Dict[str, Dict[str, str]] = {}
        # version -> {doc_id: text}
        self._texts: Dict[str, Dict[str, str]] = {}
        # Per-version rebuild timings, for telemetry / the P1 story.
        self.rebuild_seconds: Dict[str, float] = {}
        self.encoded_counts: Dict[str, int] = {}

    def add_version(self, version: str, corpus: Mapping[str, str]) -> float:
        """Add/replace a version's corpus. Returns the (cheap) rebuild time.

        Only snippets whose content hash is not already cached get encoded.
        """
        doc_hashes: Dict[str, str] = {}
        to_encode: List[str] = []
        to_encode_hashes: List[str] = []
        for doc_id, text in corpus.items():
            h = content_hash(text)
            doc_hashes[doc_id] = h
            if h not in self._emb_cache and h not in to_encode_hashes:
                to_encode.append(text)
                to_encode_hashes.append(h)

        with Timer() as t:
            if to_encode:
                new_emb = self.encoder.encode_texts(to_encode, is_query=False)
                for h, row in zip(to_encode_hashes, new_emb):
                    self._emb_cache[h] = row
        self.versions[version] = doc_hashes
        self._texts[version] = dict(corpus)
        self.rebuild_seconds[version] = t.seconds
        self.encoded_counts[version] = len(to_encode)
        return t.seconds

    def matrix_for(self, version: str) -> tuple[List[str], np.ndarray]:
        """Return (doc_ids, embedding matrix) for a version, from cache."""
        if version not in self.versions:
            raise KeyError(f"Unknown version: {version!r}")
        doc_ids = list(self.versions[version].keys())
        if not doc_ids:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        mat = np.vstack([self._emb_cache[self.versions[version][d]] for d in doc_ids])
        return doc_ids, mat.astype(np.float32)

    def search(self, version: str, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        doc_ids, mat = self.matrix_for(version)
        if not doc_ids:
            return []
        qvec = self.encoder.encode_texts([query], is_query=True)[0]
        scores = mat @ qvec
        k = min(top_k, len(doc_ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(doc_ids[i], float(scores[i])) for i in idx]

    def cache_hit_rate(self, version: str) -> float:
        """Fraction of a version's docs served from cache (0..1)."""
        total = len(self.versions.get(version, {}))
        if total == 0:
            return 0.0
        return 1.0 - self.encoded_counts.get(version, 0) / total

    # -- Persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        hashes = list(self._emb_cache.keys())
        if hashes:
            mat = np.vstack([self._emb_cache[h] for h in hashes])
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        np.savez_compressed(path / "embeddings.npz", matrix=mat, hashes=np.array(hashes, dtype=object))
        manifest = {
            "dim": self.dim,
            "versions": self.versions,
            "texts": self._texts,
            "rebuild_seconds": self.rebuild_seconds,
            "encoded_counts": self.encoded_counts,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self, path: str) -> "VersionedIndex":
        path = Path(path)
        data = np.load(path / "embeddings.npz", allow_pickle=True)
        hashes = list(data["hashes"])
        mat = data["matrix"]
        self._emb_cache = {h: mat[i] for i, h in enumerate(hashes)}
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.dim = manifest["dim"]
        self.versions = manifest["versions"]
        self._texts = manifest["texts"]
        self.rebuild_seconds = manifest.get("rebuild_seconds", {})
        self.encoded_counts = manifest.get("encoded_counts", {})
        return self
