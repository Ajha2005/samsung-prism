"""A deterministic, download-free embedding backend.

This is *not* the competition model. It is a hashed bag-of-features vectorizer
that turns text into a fixed-dimension vector using only the standard library
and numpy. Cosine similarity between two vectors approximates weighted token +
character-n-gram overlap, which is a real (if modest) retrieval signal.

Why it exists:
  * CI and unit tests run end-to-end with no network and no model download.
  * When the sentence-transformers stack or HuggingFace is unreachable, the
    pipeline degrades to this instead of crashing, so `one command -> a number`
    always holds.

It is deterministic for a fixed input, so eval is reproducible.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List

import numpy as np

from prism.backends.base import l2_normalize

# Split code/text into word-ish tokens: identifiers, numbers, and symbols.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def _stable_hash(feature: str) -> int:
    """A process-independent hash (Python's built-in hash is salted per run)."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def tokenize(text: str) -> List[str]:
    """Tokenize with light camelCase / snake_case splitting for code identifiers."""
    raw = _TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for tok in raw:
        out.append(tok)
        # Expand snake_case and camelCase so `getUserName` shares signal with
        # `user`, `name`, etc. (helps identifier-oriented code queries).
        for part in re.split(r"[_]+|(?<=[a-z0-9])(?=[A-Z])", tok):
            part = part.lower()
            if part and part != tok:
                out.append(part)
    return out


def _char_ngrams(token: str, n: int = 3) -> Iterable[str]:
    if len(token) < n:
        return ()
    padded = f"#{token}#"
    return (padded[i : i + n] for i in range(len(padded) - n + 1))


class HashingBackend:
    """Feature-hashing embedder with signed hashing and sublinear TF."""

    def __init__(self, dim: int = 1024, normalize: bool = True, char_ngrams: bool = True):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.normalize = normalize
        self.char_ngrams = char_ngrams

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
            if self.char_ngrams:
                for ng in _char_ngrams(tok):
                    key = "$" + ng  # namespace char-ngram features
                    counts[key] = counts.get(key, 0) + 1
        for feature, count in counts.items():
            h = _stable_hash(feature)
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            # Sublinear term frequency dampens the effect of very frequent tokens.
            weight = 1.0 + math.log(count)
            vec[idx] += sign * weight
        return vec

    def embed(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t or "") for t in texts])
        if self.normalize:
            matrix = l2_normalize(matrix)
        return matrix.astype(np.float32)
