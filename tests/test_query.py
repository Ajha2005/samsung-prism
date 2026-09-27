"""Query pre-processing, classification, and HyDE sketching."""

from prism.query.hyde import TemplateSketcher, compile_query_to_code
from prism.query.preprocess import (
    QueryType,
    analyze_query,
    classify_query,
    extract_keywords,
    front_load_keywords,
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


def test_front_load_puts_keywords_first_and_preserves_body():
    q = "You are given an array. Compute the maximum contiguous subarray sum."
    out = front_load_keywords(analyze_query(q), max_keywords=4)
    # Preamble comes before the original text.
    body_idx = out.find("You are given")
    assert body_idx > 0
    # A distinctive keyword ended up in the preamble.
    preamble = out[:body_idx]
    assert "subarray" in preamble.lower() or "contiguous" in preamble.lower()


def test_front_load_preserves_full_body():
    q = "Reverse a singly linked list in place using O(1) memory."
    out = front_load_keywords(analyze_query(q), max_keywords=3)
    # Nothing is dropped from the original text.
    assert q in out


def test_front_load_empty_query_is_empty():
    assert front_load_keywords(analyze_query(""), max_keywords=5) == ""


def test_front_load_zero_max_returns_original():
    q = "Sort an array of integers ascending."
    a = analyze_query(q)
    assert front_load_keywords(a, max_keywords=0) == a.normalized


def test_front_load_skips_keyword_already_at_start():
    # 'sort' is already the first token; front-loading it again would be wasteful.
    q = "sort the elements of a list into ascending order"
    a = analyze_query(q)
    out = front_load_keywords(a, max_keywords=6)
    # 'sort' should not appear twice back-to-back at the very start.
    assert not out.lower().startswith("sort, sort")
    assert not out.lower().startswith("sort. sort")
