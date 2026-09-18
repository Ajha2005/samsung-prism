"""Multi-view snippet representation.

A raw-text embedding blurs signal that matters for code retrieval: it under-
weights *structure* (two functions with the same shape but different names) and
*identifiers* (the exact API names a usage query is really after). We represent
each snippet as three complementary views and fuse their embeddings:

  * ``raw``         — cleaned source, as-is (semantics + comments).
  * ``ast``         — AST-normalized structural skeleton (names canonicalized).
  * ``identifiers`` — a bag of the distinctive identifiers and string literals.

Fusion happens at the embedding level (weighted sum of L2-normalized view
vectors, then renormalize) so each corpus item is still a single vector and the
representation stays a drop-in MTEB encoder.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List

from prism.snippet.ast_normalize import normalize_code, _PRESERVE
from prism.snippet.preprocess import clean_snippet

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_RE = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""")


@dataclass
class SnippetViews:
    raw: str
    ast: str
    identifiers: str

    def as_dict(self) -> Dict[str, str]:
        return {"raw": self.raw, "ast": self.ast, "identifiers": self.identifiers}


def _split_identifier(token: str) -> List[str]:
    """Split camelCase / snake_case into sub-words for a richer bag."""
    parts = re.split(r"[_]+|(?<=[a-z0-9])(?=[A-Z])", token)
    return [p for p in parts if p]


def extract_identifier_bag(code: str, max_terms: int = 128) -> str:
    """Build the identifier/token-bag view.

    Prefers real identifiers from a Python parse (function names, attributes,
    call targets, string literals). Falls back to a regex scan for non-Python or
    unparseable code. Sub-words of compound identifiers are included so
    ``getUserName`` contributes ``user`` and ``name`` too.
    """
    terms: List[str] = []
    seen = set()

    def _add(tok: str) -> None:
        for piece in [tok, *_split_identifier(tok)]:
            low = piece.lower()
            if len(piece) < 2 or low in seen:
                continue
            seen.add(low)
            terms.append(piece)

    parsed_ok = False
    try:
        tree = ast.parse(code)
        parsed_ok = True
    except (SyntaxError, ValueError):
        tree = None

    if parsed_ok and tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                _add(node.id)
            elif isinstance(node, ast.Attribute):
                _add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _add(node.name)
            elif isinstance(node, ast.arg):
                _add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for tok in _IDENT_RE.findall(node.value):
                    _add(tok)
    else:
        # Regex fallback: tokens inside string literals first, then all idents.
        for span in _STRING_RE.finditer(code):
            for tok in _IDENT_RE.findall(span.group(0)):
                _add(tok)
        for tok in _IDENT_RE.findall(code):
            _add(tok)

    # Keep distinctive names first; drop pure language boilerplate.
    filtered = [t for t in terms if t not in _PRESERVE]
    ordered = filtered + [t for t in terms if t in _PRESERVE]
    return " ".join(ordered[:max_terms])


def build_views(code: str, language: str | None = None) -> SnippetViews:
    """Produce the three snippet views."""
    raw = clean_snippet(code)
    ast_view = normalize_code(raw, language=language)
    id_bag = extract_identifier_bag(raw)
    # Guard against empty views (e.g. unparseable): fall back to raw so a view
    # never contributes a zero vector that dilutes the fusion.
    if not ast_view:
        ast_view = raw
    if not id_bag:
        id_bag = raw
    return SnippetViews(raw=raw, ast=ast_view, identifiers=id_bag)
