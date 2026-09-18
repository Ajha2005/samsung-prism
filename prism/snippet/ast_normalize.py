"""AST-normalized code view.

Two snippets that do the same thing with different variable names should look
alike to a structural matcher. This module produces a canonical form where
locally-bound names (variables, arguments, user-defined functions/classes) are
renamed to positional placeholders (``VAR1``, ``FUNC1``, ...) while structure,
operators, control flow, and well-known API names are preserved.

  * Python -> uses the standard-library :mod:`ast` (no third-party parser, so it
    always works). CoIR ``AppsRetrieval`` is Python, so this covers the eval.
  * Anything else / parse failures -> a language-agnostic token normalizer that
    canonicalizes identifier-looking tokens the same way.

The result feeds the "ast" view in :mod:`prism.snippet.multiview`.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import re
from typing import Dict

# Names we never rename: they carry real API/structural signal.
_PRESERVE = frozenset(dir(builtins)) | frozenset(keyword.kwlist) | {
    "self",
    "cls",
    "range",
    "len",
    "print",
    "append",
    "sorted",
    "sum",
    "min",
    "max",
    "map",
    "filter",
    "int",
    "str",
    "list",
    "dict",
    "set",
    "tuple",
}


class _Canonicalizer(ast.NodeTransformer):
    """Rename bound identifiers to stable positional placeholders."""

    def __init__(self) -> None:
        self._names: Dict[str, str] = {}
        self._func_count = 0
        self._var_count = 0
        self._arg_count = 0

    def _rename(self, original: str, kind: str) -> str:
        if original in _PRESERVE:
            return original
        if original in self._names:
            return self._names[original]
        if kind == "func":
            self._func_count += 1
            placeholder = f"FUNC{self._func_count}"
        elif kind == "arg":
            self._arg_count += 1
            placeholder = f"ARG{self._arg_count}"
        else:
            self._var_count += 1
            placeholder = f"VAR{self._var_count}"
        self._names[original] = placeholder
        return placeholder

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._rename(node.name, "func")
        # Drop docstrings so prose doesn't pollute the structural view.
        _strip_docstring(node)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = self._rename(node.name, "func")
        _strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._rename(node.arg, "arg")
        node.annotation = None  # annotations are noise for structural matching
        return node

    def visit_Name(self, node: ast.Name):
        node.id = self._rename(node.id, "var")
        return node


def _strip_docstring(node) -> None:
    body = getattr(node, "body", None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        node.body = body[1:] or [ast.Pass()]


def normalize_python(code: str) -> str:
    """Return the canonical structural form of Python ``code``.

    Raises nothing: on syntax errors it returns "" so callers can fall back.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    # Strip a module-level docstring too.
    _strip_docstring(tree)
    tree = _Canonicalizer().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # pragma: no cover - ast.unparse is very robust on 3.9+
        return ""


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMENT_RE = re.compile(r"(#.*?$)|(//.*?$)|(/\*.*?\*/)", re.MULTILINE | re.DOTALL)


def normalize_tokens(code: str) -> str:
    """Language-agnostic fallback: canonicalize identifiers at the token level."""
    code = _COMMENT_RE.sub(" ", code)
    mapping: Dict[str, str] = {}
    counter = 0

    def _sub(m: "re.Match[str]") -> str:
        nonlocal counter
        tok = m.group(0)
        if tok in _PRESERVE or keyword.iskeyword(tok):
            return tok
        if tok not in mapping:
            counter += 1
            mapping[tok] = f"VAR{counter}"
        return mapping[tok]

    normalized = _IDENT_RE.sub(_sub, code)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_code(code: str, language: str | None = None) -> str:
    """Normalize ``code``, trying a Python AST first then the token fallback."""
    code = code or ""
    if language in (None, "python", "py"):
        normalized = normalize_python(code)
        if normalized:
            return normalized
    return normalize_tokens(code)
