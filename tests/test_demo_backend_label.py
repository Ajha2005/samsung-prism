"""The demo must say which embedder actually ran.

The demo builds its backend non-strictly, so a model that fails to load falls
back to the offline hashing backend. Without a label, a recording of the demo
could show hashing-backend results under the submitted model's name.
"""

import prism.encoder
from prism.backends.hashing import HashingBackend
from prism.cli import demo_main


def test_offline_backend_is_labelled(capsys):
    assert demo_main(["--backend", "hashing", "--query", "sort a list"]) == 0
    out = capsys.readouterr().out
    assert "embedding backend: offline HashingBackend" in out
    assert "WARNING" not in out
    # Latency is labelled by what it measures: the full query path vs ranking alone.
    assert "end to end (encode + rank)" in out
    assert "ranking step only" in out


def test_fallback_from_requested_model_is_flagged(capsys, monkeypatch):
    monkeypatch.setattr(prism.encoder, "build_backend", lambda config, strict=False: HashingBackend(dim=256))
    model = "sentence-transformers/all-mpnet-base-v2"
    assert demo_main(["--model", model, "--query", "sort a list"]) == 0
    out = capsys.readouterr().out
    assert f"WARNING: {model} did not load" in out
