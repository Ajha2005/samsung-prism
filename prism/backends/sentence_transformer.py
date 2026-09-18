"""The competition embedding backend, built on sentence-transformers.

Runs CPU inference with a small code/text embedding model. The first use
downloads the model from HuggingFace; subsequent runs read the local cache, so a
frozen checkout reproduces the same vectors.

Kept deliberately thin: all query/snippet cleverness lives in the pipeline, so
this class only owns model loading, batching, and (optional) prompt prefixes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from prism.backends.base import l2_normalize


class SentenceTransformerBackend:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32,
        device: str = "cpu",
        max_seq_length: Optional[int] = 256,
        query_prompt: Optional[str] = None,
        passage_prompt: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        # Imported lazily so the rest of the package (and the offline backend)
        # works without torch/sentence-transformers installed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt

        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
        if max_seq_length:
            # Cap sequence length to keep CPU latency predictable.
            try:
                self.model.max_seq_length = max_seq_length
            except Exception:
                pass

        dim = self.model.get_sentence_embedding_dimension()
        if not dim:
            # Some models report None until first encode; probe once.
            dim = int(self.model.encode(["_"], convert_to_numpy=True).shape[1])
        self.dim = int(dim)

    def embed(self, texts: List[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prompt = self.query_prompt if is_query else self.passage_prompt
        payload = [f"{prompt}{t}" if prompt else t for t in texts]
        vectors = self.model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for consistency
            show_progress_bar=False,
        ).astype(np.float32)
        if self.normalize:
            vectors = l2_normalize(vectors)
        return vectors
