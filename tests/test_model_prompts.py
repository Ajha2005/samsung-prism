"""Instruction-prefixed models (e5) need their prefixes on the right side.

e5 is trained with "query: " on queries and "passage: " on documents; swapping
or dropping them silently degrades retrieval. This follows config/e5_base.json
through build_backend to the exact strings the model receives, with a stub in
place of sentence-transformers so no download is needed.
"""

import sys
import types
from pathlib import Path

import numpy as np

from prism.backends import build_backend
from prism.config import load_config

E5_CONFIG = Path(__file__).resolve().parents[1] / "config" / "e5_base.json"


class _RecordingModel:
    def __init__(self, name, device=None, trust_remote_code=False):
        self.name = name
        self.max_seq_length = None
        self.seen = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


def test_e5_config_prefixes_queries_and_passages(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_RecordingModel))
    backend = build_backend(load_config(str(E5_CONFIG)), strict=True)

    backend.embed(["sort a list"], is_query=True)
    backend.embed(["def f(): pass"], is_query=False)

    assert backend.model.name == "intfloat/e5-base-v2"
    assert backend.model.max_seq_length == 512
    assert backend.model.seen == [["query: sort a list"], ["passage: def f(): pass"]]
