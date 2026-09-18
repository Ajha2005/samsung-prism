"""Snippet cleaning, chunking, AST normalization, and multi-view."""

from prism.snippet.ast_normalize import normalize_code, normalize_python, normalize_tokens
from prism.snippet.multiview import build_views, extract_identifier_bag
from prism.snippet.preprocess import chunk_snippet, clean_snippet


def test_clean_strips_trailing_ws_and_blank_runs():
    code = "def f():   \n\n\n\n    return 1   \n"
    cleaned = clean_snippet(code)
    assert "   \n" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_short_snippet_is_single_piece():
    assert chunk_snippet("x = 1", max_chars=100) == ["x = 1"]


def test_chunk_long_snippet_overlaps():
    code = "\n".join(f"line_{i} = {i}" for i in range(200))
    chunks = chunk_snippet(code, max_chars=200, overlap=40)
    assert len(chunks) > 1
    # Reassembly covers all lines.
    joined = "\n".join(chunks)
    assert "line_0 " in joined and "line_199 " in joined


def test_ast_normalize_is_rename_invariant():
    a = normalize_python("def add(x, y):\n    return x + y")
    b = normalize_python("def plus(first, second):\n    return first + second")
    assert a == b  # variable/function renames collapse to the same skeleton
    assert a  # non-empty


def test_ast_normalize_strips_docstring():
    normalized = normalize_python('def f():\n    """docstring"""\n    return 1')
    assert "docstring" not in normalized


def test_ast_normalize_falls_back_on_syntax_error():
    # Not valid Python -> token fallback still returns something non-empty.
    out = normalize_code("function foo(a){ return a+1; }", language="javascript")
    assert out
    assert "VAR" in out or "function" in out


def test_token_fallback_preserves_keywords():
    out = normalize_tokens("for item in things: print(item)")
    assert "for" in out and "in" in out and "print" in out


def test_identifier_bag_extracts_names():
    bag = extract_identifier_bag("def get_user_name(user):\n    return user.name")
    # snake_case / attribute names surface, sub-words too.
    assert "user" in bag
    assert "name" in bag


def test_build_views_three_nonempty():
    v = build_views("def is_palindrome(s):\n    return s == s[::-1]")
    assert v.raw and v.ast and v.identifiers
    assert "palindrome" in v.identifiers
