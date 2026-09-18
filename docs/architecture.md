# Architecture

The system is judged by MTEB as an **encoder**: an object that maps queries and
code snippets into a shared vector space where cosine similarity ranks relevance.
All cleverness lives in the pre- and post-processing wrapped around a small,
CPU-friendly embedding model.

## Data flow

```
NL query ─▶ QueryAnalysis ─▶ [HyDE sketch] ─▶ backend.embed(is_query=True) ─▶ blend ─▶ q-vec
code     ─▶ clean ─▶ [multi-view: raw|ast|id-bag] ─▶ backend.embed ─▶ weighted fuse ─▶ d-vec
q-vec · d-vec ─▶ cosine (+ optional BM25 hybrid, + optional cross-version) ─▶ ranking
```

## Components

### `backends/` — text → vectors
A small `EmbeddingBackend` protocol (`embed(texts, is_query) -> (n, dim)`), with
two implementations:
- `SentenceTransformerBackend` — the competition model (CPU inference; downloads
  from HuggingFace once, then cached). Thin: only model loading, batching, and
  optional query/passage prompt prefixes.
- `HashingBackend` — deterministic feature-hashing embedder (signed hashing,
  sublinear TF, char n-grams). No network, no model. Powers offline tests/CI and
  is the graceful-degradation fallback if the model can't load.

`build_backend(config)` selects one and falls back to hashing with a warning if
the ST stack is unavailable — so a run never hard-fails on the environment.

### `encoder.py` — `PrePostPipelineEncoder`
The heart. Two faces:
- `encode_texts(texts, is_query=...)` — framework-free; used by the retriever,
  demo, and all tests (no MTEB needed to import the module).
- `as_mteb_encoder(config)` — mixes in MTEB 2.x's `AbsEncoder` and implements
  `encode(inputs: DataLoader, *, task_metadata, hf_split, hf_subset, prompt_type)`,
  routing `PromptType.query` → query path and `PromptType.document` → snippet
  path. This is what `mteb.evaluate` drives.

Query path: `analyze_query` → optional HyDE sketch → embed → per-query blend
(HyDE weight adapts to query type: more for behavioral, less for lookups).
Snippet path: clean → optional multi-view fuse (or chunk+max-pool for long
snippets) → normalize.

### `query/`
- `preprocess.py` — whitespace normalization, salient-keyword extraction (quoted
  spans, dotted/`snake`/`camel` tokens, content words minus stopwords), and a
  heuristic query-type classifier (behavioral / lookup / structural / usage).
- `hyde.py` — `TemplateSketcher` deterministically compiles a query into a
  plausible Python sketch (function whose name/body echo the keywords and
  intent). `CachedSketcher` serves precomputed offline sketches with a template
  fallback. No LLM call at eval time.

### `snippet/`
- `preprocess.py` — structure-preserving cleanup and line-boundary chunking with
  overlap for long snippets.
- `ast_normalize.py` — Python `ast`-based canonicalization: bound identifiers →
  `VAR/ARG/FUNC` placeholders, docstrings/annotations stripped, well-known API
  names preserved. Token-level fallback for non-Python / unparseable code.
- `multiview.py` — builds the raw / AST-normalized / identifier-bag views and is
  fused at the embedding level (weighted sum of L2-normalized view vectors →
  renormalize), keeping one vector per corpus item.

### `scoring/`
- `bm25.py` — `rank_bm25` with the same identifier-aware tokenizer as the
  backend, so dense and sparse see code the same way.
- `fusion.py` — Reciprocal Rank Fusion (scale-free, default) and a min-max linear
  fusion. Used by the standalone retriever's ranking.

### `index/`
- `versioned.py` — `VersionedIndex` with a **content-addressed embedding cache**:
  a version is `{doc_id: text}`; adding a version re-encodes only snippets whose
  content hash isn't cached. Persists to `.npz` + JSON. Reports rebuild time and
  cache-hit rate (the P1 story).
- `evolutionary.py` — `EvolutionaryEncoder`: `vector = normalize(base +
  delta_weight · delta)`, where `delta` embeds the lines distinctive to a version
  (classified against the group by a normalized key, embedded raw to keep
  identifier signal). `stable_core_and_delta` exposes the invariant/changed split.

### `telemetry/`
- `metrics.py` — trec_eval-compatible NDCG@k / MRR / P@k / Recall@k / MAP, plus
  `RunTelemetry` (latency, throughput, index cost) and a `Timer`. Verified to
  match MTEB to 1e-4.

### `eval/`
- `mteb_runner.py` — `run_apps_retrieval` (real leaderboard JSON) and `run_task`
  (any MTEB task).
- `synthetic_task.py` — a local `AbsTaskRetrieval` populated from in-memory
  corpus/queries/qrels, so the full MTEB path runs offline in CI.

## Why this shape

- MTEB scores a *vector encoder*, so every differentiator is designed to fold
  into the query or document **vector** (not a bespoke ranker MTEB can't see).
- The framework-free `encode_texts` core means the entire pipeline is unit- and
  integration-testable with no torch, no MTEB, and no network — fast feedback,
  reproducible everywhere.
- One `PipelineConfig` object fully describes an experiment, so an ablation is a
  one-field diff and the config travels inside the results JSON.
