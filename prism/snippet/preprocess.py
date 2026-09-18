"""Snippet pre-processing: clean and chunk."""

from __future__ import annotations

import re
from typing import List

_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_BLANK_RE = re.compile(r"\n{3,}")


def clean_snippet(code: str) -> str:
    """Light, structure-preserving cleanup.

    Normalizes line endings, strips trailing whitespace, and collapses runs of
    blank lines. Case and indentation are preserved — they are meaningful in
    code.
    """
    if not code:
        return ""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = _TRAILING_WS_RE.sub("", code)
    code = _MANY_BLANK_RE.sub("\n\n", code)
    return code.strip()


def chunk_snippet(code: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Split a long snippet into overlapping windows at line boundaries.

    Returns ``[code]`` unchanged when it fits (the common case), so short
    snippets pay no cost. Chunk embeddings are max-pooled by the caller.
    """
    if max_chars <= 0 or len(code) <= max_chars:
        return [code]

    lines = code.split("\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        # +1 accounts for the newline we will rejoin with.
        if current_len + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            # Start the next window with a small overlap for context continuity.
            if overlap > 0:
                tail: List[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) + 1 > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                current = tail
                current_len = tail_len
            else:
                current = []
                current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [code]
