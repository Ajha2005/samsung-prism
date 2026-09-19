# PRISM Theme 1 — Agentic Code Intelligence

A **CPU-friendly code-retrieval pipeline**: given a natural-language query and a
library of code snippets, it returns the snippets ranked by relevance. The task
is pure retrieval (no generation) and is scored by
[MTEB](https://github.com/embeddings-benchmark/mteb) on the CoIR
`AppsRetrieval` test split via **NDCG@10** and **MRR**.

The design bet: *win on a number, not a demo.* We wrap a small embedding model
and put the cleverness in the pre- and post-processing around it — where a
sharp, full-time solo build out-iterates raw model scale on a CPU-only,
leaderboard-scored task.

```
        ┌───────────────────────────┐        ┌────────────────────────────┐
 NL ───▶│ query pre-process          │        │ snippet pre-process         │◀── code
 query  │  normalize · keywords ·    │        │  clean · (multi-view:       │    snippets
        │  classify · →code (HyDE)   │        │  raw + AST-norm + id-bag)   │
        └────────────┬──────────────┘        └──────────────┬─────────────┘
                     │ embed (query)                        │ embed + fuse (doc)
                     ▼                                       ▼
                  query vector ───▶  score + rerank  ◀─── snippet vectors
                        (cosine, optional dense+BM25 hybrid, cross-version)
                                        │
                                        ▼
                                 ranked snippets
```

Everything is a **measurable layer on top of a baseline floor**: get a real
NDCG@10 on the board first, then add one differentiator at a time and keep it
only if the scoreboard says so.

---

## Quickstart

### Offline (no downloads — proves the whole pipeline runs)

```bash
pip install -r requirements.txt        # numpy + rank-bm25
pip install -e .
python -m pytest -q                    # 70 tests, all offline
python -m prism.cli demo --backend hashing
python -m prism.cli demo --backend hashing --versioned
python -m prism.cli ablation --backend hashing
```

The `hashing` backend is a deterministic, download-free embedder used for tests,
CI, and graceful degradation. It is **not** the competition model — it exists so
`one command → a number` always holds.

### Leaderboard run (needs network: model + CoIR dataset)

```bash
pip install -r requirements.txt -r requirements-eval.txt   # + mteb, sentence-transformers, torch(CPU)
pip install -e .

python -m prism.cli eval --config config/baseline.json \
    --output results/appsretrieval_results.json
```

`results/appsretrieval_results.json` is the leaderboard artifact (it carries the
NDCG@10 / MRR plus the full MTEB result and the exact config that produced it).

**Why baseline, not `config/submission.json`:** we measured both — see
`docs/ablation_log.md` for the full ablation. On the real AppsRetrieval split,
plain `all-MiniLM-L6-v2` (NDCG@10 = **0.0662**) beat both HyDE (0.0625) and
multi-view (0.0515); `config/submission.json` (both stacked) scored lower
still. Per this repo's own rule — keep only what wins — baseline is what's
submitted. Both differentiators remain implemented and independently
verified; they're worth re-measuring against a code-specialized backend
(`config/code_model.json`) where the bet they make is more likely to pay off.

`all-MiniLM-L6-v2` is a small *general-purpose* model with a 256-token cap — real
AppsRetrieval queries/solutions often run longer, and it has never seen code
during training, so treat its score as a floor, not a ceiling. `config/code_model.json`
swaps in `flax-sentence-embeddings/st-codesearch-distilroberta-base`, a model
fine-tuned specifically for code search, with a longer 512-token window:

```bash
python -m prism.cli eval --config config/code_model.json \
    --output results/appsretrieval_results.json
```

Some code-specialized models (e.g. Jina's code embedding models) ship custom
modeling code fetched from the Hub at load time and need `trust_remote_code`
to load — set it in the config or pass `--trust-remote-code` on the CLI, and
only for a model repo you trust, since it executes code from that repo. Treat
it as a last resort: that custom code is pinned to whatever `transformers`
internals existed when it was written, so it can break on a newer
`transformers` release with an `ImportError` from deep inside the library
(this happened with `jina-embeddings-v2-base-code` during this build, which is
why `config/code_model.json` no longer uses it). **A leaderboard eval never
silently substitutes a different backend when the requested model fails to
load — it raises instead**, so an eval that reports a score always used the
model you asked for; the demo/ablation commands still degrade gracefully to
the offline backend on a load failure, since those aren't scoring anything.

### Docker (one command, reproducible)

```bash
docker build -t prism-code-search .
docker run --rm prism-code-search                     # offline: tests + demo
docker run --rm -v "$PWD/results:/app/results" \      # leaderboard eval (needs network)
    prism-code-search python -m prism.cli eval --output results/appsretrieval_results.json
```

---

## The three differentiators

Built in order of proven payoff; each is an A/B against the baseline, kept only
if it wins on the scoreboard (`python -m prism.cli ablation`).

1. **Query → code compilation (code-flavored HyDE).** NL queries and code live in
   different "languages." We compile the query into a short pseudo-code sketch
   and embed code-against-code, closing the modality gap on behavioral queries.
   The default sketcher is a **deterministic, LLM-free** template (safe at eval
   time; retrieval stays faster than generation); an offline LLM sketch cache can
   be plugged in without any LLM in the ranking loop.
   → `prism/query/hyde.py`

2. **Multi-view snippet embedding.** Each snippet is represented by three views —
   raw code, an **AST-normalized** structural skeleton (identifier-invariant), and
   an **identifier/token bag** — fused at the embedding level so each corpus item
   stays a single vector (a drop-in MTEB encoder). Captures structure and
   identifier signal a single raw-text view blurs.
   → `prism/snippet/multiview.py`, `prism/snippet/ast_normalize.py`

3. **Cross-version / evolutionary retrieval (P1 + the bonus).** A
   content-addressed index rebuilds cheaply across code versions (only changed
   snippets are re-encoded). A **stable-core + version-delta** representation
   keeps near-identical versions distinguishable: `vector = normalize(base +
   delta_weight · delta)`, so `delta_weight=0` is exactly the naive vector (it can
   only match or beat the baseline) and the delta term pulls a version-specific
   query toward the right sibling.
   → `prism/index/versioned.py`, `prism/index/evolutionary.py`

---

## Evaluation & self-measurement

- **Scoreboard-first discipline.** `prism.telemetry.metrics` reimplements
  NDCG@k, MRR, precision@k, recall@k, and MAP with trec_eval semantics — verified
  to match MTEB's output to 1e-4 (`tests/test_mteb_integration.py`). So we can
  measure on our own local tasks and the numbers line up with the leaderboard.
- **Ablation log.** `python -m prism.cli ablation` writes a `config → NDCG@10 →
  MRR → kept/dropped` table (`results/ablation_log.md` + `.json`) against a live
  baseline — the experiment log, the regression guard, and half the slides.
- **Operational metrics.** The demo and retriever report precision@k, recall,
  query latency, and index build/rebuild cost — what the hands-on round asks for.

Illustrative offline ablation (deterministic `hashing` backend on the bundled
synthetic set — the real numbers come from `--apps`):

| Config | NDCG@10 | ΔNDCG | Verdict |
|---|---:|---:|:--:|
| baseline | 0.7370 | — | keep |
| +multiview | 0.7784 | +0.0414 | keep |
| +hyde | 0.7433 | +0.0062 | keep |
| +hybrid (rrf) | 0.7345 | −0.0026 | drop |
| +all | 0.7711 | +0.0340 | keep |

(That `hybrid` row dropping is the methodology working as intended — measure,
keep what wins, revert what doesn't. The real decision is remade on
`AppsRetrieval`, and it reversed two of these: **on the real split, both
`+multiview` and `+hyde` lose to baseline** — see `docs/ablation_log.md` for
the real numbers and why. This offline table is kept here to show the harness
working, not as a preview of the real verdict.)

---

## Repository layout

```
prism/
  encoder.py            PrePostPipelineEncoder — the MTEB-scored encoder
  pipeline.py           CodeRetriever — standalone retriever (demo + telemetry)
  config.py             one diffable config object per experiment
  backends/             sentence-transformers (real) + hashing (offline) embedders
  query/                preprocess.py, hyde.py
  snippet/              preprocess.py, ast_normalize.py, multiview.py
  scoring/              bm25.py, fusion.py (RRF + linear)
  index/                versioned.py (cheap rebuild), evolutionary.py (core+delta)
  telemetry/            metrics.py (trec_eval-compatible), ablation via prism/ablation.py
  eval/                 mteb_runner.py, synthetic_task.py (offline MTEB task)
  data.py               synthetic AppsRetrieval-like fixtures
  cli.py                prism-eval / prism-demo / prism-ablation
config/                 baseline.json (submitted), submission.json, code_model.json,
                        retrieval_tuned.json, multiview_trimmed.json — see docs/ablation_log.md
scripts/                thin CLI wrappers
tests/                  70 tests, offline; MTEB integration auto-skips if absent
docs/                   architecture.md, submission_checklist.md
Dockerfile, Makefile, requirements*.txt
```

## Design constraints (why it's built this way)

- **CPU-only, small model.** Innovation is forced into representation and
  pipeline design, not scale.
- **No LLM in the ranking loop.** Any LLM use is offline pre-processing; eval-time
  cost is a string build + an embed, never a generation call.
- **Index rebuilds cheaply.** Versioned from the start (content-addressed cache),
  so P1's cross-version requirement is a first-class feature, not a bolt-on.

## Limitations (honest scope)

- **HyDE and multi-view underperform plain baseline on `all-MiniLM-L6-v2`**, measured
  directly on real AppsRetrieval (see `docs/ablation_log.md`) — both bet on a
  capability (code structure awareness, LLM-quality query translation) this small
  general-purpose model doesn't have. Baseline is what's submitted; both remain
  implemented and worth re-measuring against a code-specialized backend.
- The evolutionary bonus is a measurable **prototype**. Its version-discrimination
  gain is realized with the semantic backend; on the lexical fallback it holds
  parity with naive. The cheap-rebuild P1 requirement is fully working.
- The offline `hashing` backend is lexical, so offline numbers under-represent
  what the real embedding model achieves on semantic/behavioral queries.
- Hybrid dense+BM25 scoring shapes the standalone retriever's ranking; MTEB scores
  the pure encoder, so hybrid is reported in ablations but off in the encoder
  submission unless folded into the vector.

See `docs/submission_checklist.md` for the deliverables checklist and the exact
release-tag / naming requirements.
