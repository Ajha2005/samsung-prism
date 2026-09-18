"""Query pre-processing: normalize, extract keywords, classify type.

Natural-language code queries carry two kinds of signal the raw string buries:
the *salient identifiers* (names likely to appear verbatim in code) and the
*intent* (does the query describe behavior, name an API, or ask about
structure?). Surfacing both lets downstream steps adapt — e.g. apply HyDE more
aggressively to behavioral queries, and feed keywords to the sparse (BM25)
signal.

All heuristics are deterministic and dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

_WS_RE = re.compile(r"\s+")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_RE = re.compile(r"[`'\"]([^`'\"]+)[`'\"]")
_CODEISH_RE = re.compile(r"[A-Za-z_]\w*\.[A-Za-z_]\w*|[A-Za-z_]\w*\(\)|[a-z]+_[a-z_]+|[a-z]+[A-Z]\w*")

# Very common English words that are never useful code keywords.
_STOPWORDS = frozenset(
    """
    a an the of to in on for and or is are be do does how what which where when why who
    that this these those with without using use used i you we it its as at by from into
    can could would should will shall may might must want need find get set write code
    function method way given return returns some any all my your our their he she they
    """.split()
)

# Words that signal the query is describing *behavior* rather than naming things.
_BEHAVIOR_MARKERS = frozenset(
    """
    compute calculate parse sort search reverse merge convert count check validate
    generate produce return find determine implement solve print output format encode
    decode filter transform traverse detect handle process build create remove insert
    update delete append add subtract multiply divide compare match
    """.split()
)


class QueryType(str, Enum):
    """Coarse query intent, used to steer the pipeline."""

    BEHAVIORAL = "behavioral"      # "reverse a linked list", "sort by frequency"
    LOOKUP = "lookup"              # names an API / function / symbol
    STRUCTURAL = "structural"      # "which files call X before Y"
    USAGE = "usage"                # "where is the Bluetooth deeplink used"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    """The structured result of analysing one query."""

    raw: str
    normalized: str
    keywords: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL

    def is_behavioral(self) -> bool:
        return self.query_type in (QueryType.BEHAVIORAL, QueryType.GENERAL)


def normalize_query(text: str) -> str:
    """Collapse whitespace and strip. Case is preserved (identifiers are case-sensitive)."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def extract_keywords(text: str, max_keywords: int = 12) -> List[str]:
    """Pull the salient tokens: quoted spans, code-ish tokens, and content words.

    Order-preserving and de-duplicated, so the most-mentioned distinctive terms
    lead. These feed both the sparse signal and the HyDE sketch.
    """
    keywords: List[str] = []
    seen = set()

    def _add(token: str) -> None:
        token = token.strip()
        low = token.lower()
        if not token or low in seen:
            return
        seen.add(low)
        keywords.append(token)

    # 1) Anything explicitly quoted or backticked is almost always a real symbol.
    for m in _QUOTED_RE.findall(text):
        for tok in _IDENTIFIER_RE.findall(m):
            _add(tok)
    # 2) Code-ish tokens (dotted calls, snake_case, camelCase) are strong signal.
    for m in _CODEISH_RE.findall(text):
        _add(m)
    # 3) Remaining content words, minus stopwords and pure numbers.
    for tok in _IDENTIFIER_RE.findall(text):
        if tok.lower() in _STOPWORDS or tok.isdigit():
            continue
        _add(tok)

    return keywords[:max_keywords]


def classify_query(text: str, keywords: List[str] | None = None) -> QueryType:
    """Heuristically classify query intent."""
    low = text.lower()
    tokens = set(_IDENTIFIER_RE.findall(low))

    # Structural: talks about files/calls/order between symbols.
    if re.search(r"\b(which|what)\b.*\b(file|files|module|class|function)\b", low) and (
        "call" in low or "calls" in low or "before" in low or "after" in low or "import" in low
    ):
        return QueryType.STRUCTURAL

    # Usage: "where is X used / called / referenced".
    if re.search(r"\bwhere\b.*\b(used|called|referenced|defined|declared|invoked)\b", low):
        return QueryType.USAGE

    # Lookup: short and dominated by a code-ish symbol, little prose.
    codeish = _CODEISH_RE.findall(text)
    word_count = len(_IDENTIFIER_RE.findall(low))
    if codeish and word_count <= 4:
        return QueryType.LOOKUP

    # Behavioral: contains an action verb describing what the code should do.
    if tokens & _BEHAVIOR_MARKERS:
        return QueryType.BEHAVIORAL

    return QueryType.GENERAL


def analyze_query(text: str, *, normalize: bool = True, extract: bool = True, classify: bool = True) -> QueryAnalysis:
    """Run the full query analysis, honoring which steps are enabled."""
    norm = normalize_query(text) if normalize else (text or "")
    keywords = extract_keywords(norm) if extract else []
    identifiers = [k for k in keywords if _IDENTIFIER_RE.fullmatch(k)]
    qtype = classify_query(norm, keywords) if classify else QueryType.GENERAL
    return QueryAnalysis(
        raw=text or "",
        normalized=norm,
        keywords=keywords,
        identifiers=identifiers,
        query_type=qtype,
    )
