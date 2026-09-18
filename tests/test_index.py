"""Versioned index (cheap rebuild) and evolutionary encoder."""

import numpy as np

from prism.config import BackendConfig, PipelineConfig
from prism.encoder import PrePostPipelineEncoder
from prism.index.evolutionary import EvolutionaryEncoder, stable_core_and_delta
from prism.index.versioned import VersionedIndex, content_hash


def _encoder():
    cfg = PipelineConfig(name="t", backend=BackendConfig(kind="hashing", hashing_dim=1024))
    return PrePostPipelineEncoder(cfg)


def test_content_hash_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_versioned_rebuild_only_encodes_changed():
    idx = VersionedIndex(_encoder())
    v1 = {"a": "def a(): return 1", "b": "def b(): return 2", "c": "def c(): return 3"}
    idx.add_version("v1", v1)
    assert idx.encoded_counts["v1"] == 3

    v2 = dict(v1)
    v2["b"] = "def b():\n    return 22  # changed"
    idx.add_version("v2", v2)
    assert idx.encoded_counts["v2"] == 1  # only 'b' re-encoded
    assert abs(idx.cache_hit_rate("v2") - 2 / 3) < 1e-9


def test_versioned_matrix_and_search():
    idx = VersionedIndex(_encoder())
    idx.add_version("v1", {"rev": "def reverse(x): return x[::-1]", "add": "def add(a,b): return a+b"})
    ids, mat = idx.matrix_for("v1")
    assert mat.shape[0] == 2 and mat.shape[1] == idx.dim
    hits = idx.search("v1", "reverse a sequence", top_k=1)
    assert hits[0][0] == "rev"


def test_versioned_persistence_roundtrip(tmp_path):
    enc = _encoder()
    idx = VersionedIndex(enc)
    idx.add_version("v1", {"a": "def a(): return 1", "b": "def b(): return 2"})
    idx.save(str(tmp_path / "idx"))

    idx2 = VersionedIndex(enc).load(str(tmp_path / "idx"))
    ids1, m1 = idx.matrix_for("v1")
    ids2, m2 = idx2.matrix_for("v1")
    assert ids1 == ids2
    assert np.allclose(m1, m2)


def test_stable_core_and_delta_splits_lines():
    versions = [
        "def f(x):\n    return x",
        "def f(x):\n    log(x)\n    return x",
    ]
    core, delta = stable_core_and_delta(versions[1], versions)
    assert "return x" in core
    assert "log(x)" in delta


def test_evolutionary_delta_weight_zero_equals_naive():
    enc = _encoder()
    versions = ["def f(): return 1", "def f(): return 2  # v2"]
    evo = EvolutionaryEncoder(enc, delta_weight=0.0)
    evo_vecs = evo.encode_group(versions)
    from prism.snippet.preprocess import clean_snippet
    naive = enc.encode_texts([clean_snippet(v) for v in versions], is_query=False)
    assert np.allclose(evo_vecs, naive, atol=1e-6)


def test_evolutionary_vectors_are_unit_norm():
    enc = _encoder()
    versions = ["def f(): return 1", "def f(): return 2", "def f(): return 3"]
    vecs = EvolutionaryEncoder(enc, delta_weight=1.0).encode_group(versions)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
