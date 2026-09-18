"""PRISM Theme 1 — Agentic Code Intelligence.

A CPU-friendly code-retrieval pipeline that ranks code snippets by relevance to
a natural-language query. The cleverness lives in the pre- and post-processing
around a small embedding model:

  * query pre-processing + query->code compilation (code-flavored HyDE)
  * multi-view snippet embedding (raw + AST-normalized + identifier bag)
  * hybrid dense + sparse (BM25) scoring
  * cross-version / evolutionary retrieval (stable-core + version-delta)

The public entry point is :class:`prism.encoder.PrePostPipelineEncoder`, an
MTEB-compatible encoder scored on the CoIR ``AppsRetrieval`` test split.
"""

from prism.config import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config", "__version__"]

__version__ = "1.0.0"
