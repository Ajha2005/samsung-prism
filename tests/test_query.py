"""Query pre-processing, classification, and HyDE sketching."""

from prism.query.hyde import TemplateSketcher, compile_query_to_code
from prism.query.preprocess import (
    QueryType,
    analyze_query,
    classify_query,
    extract_keywords,
    normalize_query,
)


def test_normalize_collapses_whitespace():
    assert normalize_query("  find   two\tsums\n ") == "find two sums"
    assert normalize_query("") == ""


def test_extract_keywords_prefers_symbols_and_drops_stopwords():
    kws = extract_keywords("how do I call os.urandom to get bytes")
    assert "os.urandom" in kws
    assert "how" not in kws and "do" not in kws and "to" not in kws


def test_classify_behavioral():
    assert classify_query("reverse a linked list") == QueryType.BEHAVIORAL


def test_classify_usage():
    assert classify_query("where is the deeplink used") == QueryType.USAGE


def test_classify_structural():
    q = "which files call parse before validate"
    assert classify_query(q) == QueryType.STRUCTURAL


def test_classify_lookup_short_symbol():
    assert classify_query("os.path.join()") == QueryType.LOOKUP


def test_analyze_respects_flags():
    a = analyze_query("Sort A List", normalize=False, extract=False, classify=False)
    assert a.normalized == "Sort A List"
    assert a.keywords == []
    assert a.query_type == QueryType.GENERAL


def test_template_sketch_is_codeish_and_deterministic():
    q = "compute the greatest common divisor of two numbers"
    s1 = compile_query_to_code(q)
    s2 = compile_query_to_code(q)
    assert s1 == s2  # deterministic (no LLM, safe at eval time)
    assert s1.startswith("def ")
    assert '"""' in s1  # docstring restates the query


def test_sketcher_handles_empty_query():
    assert TemplateSketcher().sketch(analyze_query("")).startswith("def ")
