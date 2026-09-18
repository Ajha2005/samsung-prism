"""Query -> code compilation (code-flavored HyDE).

Natural-language queries and code live in different "languages," so embedding
them in the same space is lossy. HyDE (Hypothetical Document Embeddings) closes
that gap by embedding a *hypothetical answer* instead of the question. Here the
hypothetical answer is a short pseudo-code sketch of what the matching code
would look like, so we embed code-against-code.

Two sketch sources, both compliant with "no LLM in the ranking loop":

  * :class:`TemplateSketcher` — a deterministic, dependency-free transform that
    turns the query analysis into a Python sketch. Runs at eval time, no model.
  * :class:`CachedSketcher` — looks up a precomputed sketch (e.g. one an LLM
    wrote *offline*, before eval) keyed by the query text, falling back to the
    template sketcher on a miss.

The eval-time cost is a string build and one extra embed, never a generation
call, so retrieval stays faster than generation as the rules require.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from prism.query.preprocess import QueryAnalysis, QueryType, analyze_query

_NON_IDENT_RE = re.compile(r"[^A-Za-z0-9_]+")


def _snake(tokens: List[str], fallback: str = "solution") -> str:
    parts = [_NON_IDENT_RE.sub("_", t).strip("_").lower() for t in tokens]
    parts = [p for p in parts if p]
    name = "_".join(parts[:4]) if parts else fallback
    if name and name[0].isdigit():
        name = "f_" + name
    return name or fallback


class Sketcher(Protocol):
    def sketch(self, analysis: QueryAnalysis) -> str: ...


class TemplateSketcher:
    """Deterministically compile a query into a plausible Python code sketch.

    The sketch is intentionally generic scaffolding — a function whose name and
    body echo the query's keywords and intent — so its embedding lands in the
    code region of the space near real answers, without pretending to be a
    correct program.
    """

    def sketch(self, analysis: QueryAnalysis) -> str:
        kws = analysis.keywords or []
        func = _snake(kws)
        # Prefer identifier-like keywords as fake parameters/locals.
        params = [k for k in kws if k.isidentifier() and k.lower() != func][:3]
        param_str = ", ".join(params) if params else "data"

        lines: List[str] = []
        lines.append(f"def {func}({param_str}):")
        # Docstring restates the query so lexical overlap survives too.
        doc = analysis.normalized.replace('"""', "'''")
        lines.append(f'    """{doc}"""')

        verbs = [k for k in kws if k.lower() in _VERB_HINTS]
        nouns = [k for k in kws if k.lower() not in _VERB_HINTS][:4]

        if analysis.query_type == QueryType.STRUCTURAL:
            lines.append("    result = []")
            for n in nouns:
                lines.append(f"    # inspect references to {n}")
                lines.append(f"    for node in ast.walk(tree):")
                lines.append(f"        if getattr(node, 'id', None) == {n!r}:")
                lines.append("            result.append(node)")
            lines.append("    return result")
        elif analysis.query_type in (QueryType.LOOKUP, QueryType.USAGE):
            call = nouns[0] if nouns else "target"
            lines.append(f"    return {call}({param_str})")
        else:  # behavioral / general
            acc = "result"
            lines.append(f"    {acc} = []")
            lines.append(f"    for item in {params[0] if params else 'data'}:")
            if verbs:
                lines.append(f"        # {' '.join(verbs)}")
            lines.append(f"        {acc}.append(item)")
            lines.append(f"    return {acc}")

        return "\n".join(lines)


class CachedSketcher:
    """Serve precomputed sketches from a JSON map, falling back to a template."""

    def __init__(self, cache_path: str, fallback: Optional[Sketcher] = None):
        self.fallback = fallback or TemplateSketcher()
        path = Path(cache_path)
        self._cache: Dict[str, str] = {}
        if path.exists():
            self._cache = json.loads(path.read_text(encoding="utf-8"))

    def sketch(self, analysis: QueryAnalysis) -> str:
        key = analysis.raw.strip()
        if key in self._cache:
            return self._cache[key]
        if analysis.normalized in self._cache:
            return self._cache[analysis.normalized]
        return self.fallback.sketch(analysis)


_VERB_HINTS = frozenset(
    """
    compute calculate parse sort search reverse merge convert count check validate
    generate return find determine solve print filter transform traverse detect
    handle process build create remove insert update delete append add
    """.split()
)


def build_sketcher(cache_path: Optional[str] = None) -> Sketcher:
    if cache_path:
        return CachedSketcher(cache_path)
    return TemplateSketcher()


def compile_query_to_code(query: str, sketcher: Optional[Sketcher] = None) -> str:
    """Convenience: analyze ``query`` and return its code sketch."""
    sketcher = sketcher or TemplateSketcher()
    return sketcher.sketch(analyze_query(query))
