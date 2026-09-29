# PRISM Theme 1 — Agentic Code Intelligence

A **CPU-friendly code-retrieval pipeline**: given a natural-language query and a
library of code snippets, it returns the snippets ranked by relevance. The task
is pure retrieval (no generation) and is scored by
[MTEB](https://github.com/embeddings-benchmark/mteb) on the CoIR
`AppsRetrieval` test split via **NDCG@10** and **MRR**.

The design bet: *let the real split decide.* Every stage around the embedding
model is a switchable, measured layer, and a stage stays on only if it beats the
baseline on the real split. Those measurements picked the submitted config:
`intfloat/e5-base-v2` with its `query:` / `passage:` prefixes and a 512-token
window, plus structure-preserving snippet cleanup and chunking of long
snippets. HyDE, multi-view and dense+BM25 hybrid scoring (shown below) are
built and switchable by config, but they lost on the real split, so they're
off — see `docs/ablation_log.md`.

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

## Submission — PRISM GenAI Hackathon 2026, Theme 1 (Team Trace)

- **Result:** NDCG@10 = **0.1151**, MRR@10 = **0.0986** on the CoIR AppsRetrieval
  test split, from `config/e5_base.json`. The MTEB results JSON is attached to the
  [`PRISM_GENAI_HACKATHON_Y2026` release](https://github.com/Ajha2005/samsung-prism/releases/tag/PRISM_GENAI_HACKATHON_Y2026).
- **Deck:** [`submission/Thapar_Trace_Submission_ppt.pptx`](submission/Thapar_Trace_Submission_ppt.pptx)
  ([PDF](submission/Thapar_Trace_Submission_ppt.pdf))
- **Demo video:** [YouTube](https://youtu.be/qD0DrWegOMI)
- **Reproduce the result:** `make eval` (see *Leaderboard run* below).

---

## Quickstart

### Offline (no downloads — proves the whole pipeline runs)

```bash
pip install -r requirements.txt        # numpy + rank-bm25
pip install -e .
python -m pytest -q                    # 78 tests, all offline
python -m prism.cli demo --backend hashing
python -m prism.cli demo --backend hashing --versioned
python -m prism.cli ablation --backend hashing
```

The `hashing` backend is a deterministic, download-free embedder used for tests,
CI, and graceful degradation. It is **not** the competition model — it exists so
`one command → a number` always holds.

### Demo with the submitted model (needs network for the model download)

```bash
pip install -r requirements.txt -r requirements-eval.txt && pip install -e .
python -m prism.cli demo --config config/e5_base.json
python -m prism.cli demo --config config/e5_base.json \
    --query "given a list of meeting times, collapse the ones that overlap"
python -m prism.cli demo --config config/e5_base.json --versioned   # P1 + evolutionary bonus
```

`--query` takes any natural-language query (repeatable) and prints the ranked
snippets with per-query latency. With e5-base-v2 on CPU a query takes about
42 ms end to end (embedding the query plus ranking); the demo prints that figure
and, separately, the ranking step alone. It also prints which embedding model
actually loaded; if the model can't be downloaded it falls back to the offline
backend and prints a `WARNING` saying so.

### Leaderboard run (needs network: model + CoIR dataset)

```bash
pip install -r requirements.txt -r requirements-eval.txt   # + mteb, sentence-transformers, torch(CPU)
pip install -e .

python -m prism.cli eval --config config/e5_base.json \
    --output results/appsretrieval_results.json
```

`results/appsretrieval_results.json` is the leaderboard artifact — the raw MTEB
`task_result.to_dict()`, exactly as the problem statement's reference snippet
writes it. `make eval` runs the same command (about 1.5 h on a free Kaggle CPU).

**What's submitted: `config/e5_base.json` — NDCG@10 = 0.1151, MRR@10 = 0.0986**
on the real CoIR AppsRetrieval test split. It's `intfloat/e5-base-v2` (110M
params) with its `query:` / `passage:` prefixes and a 512-token input window,
CPU-only, with none of the other pre/post differentiators switched on. The score
matches the published CoIR result for e5-base-v2 on this task (11.5), an
external check that the MTEB wiring and the prefixes are right.

We measured ten configs on the real split and kept one — see
`docs/ablation_log.md`:

| Config | NDCG@10 | vs baseline |
|---|---:|---:|
| **e5-base-v2, query/passage prefixes (submitted)** | **0.1151** | **+73.9%** |
| mpnet, 512-token window | 0.0861 | +30.1% |
| mpnet, 384-token window | 0.0837 | +26.5% |
| baseline — all-MiniLM-L6-v2 | 0.0662 | — |
| + front-loaded keywords | 0.0646 | −2.4% |
| + HyDE | 0.0625 | −6% |
| + multi-view | 0.0515 | −22% |
| retrieval-tuned `multi-qa-MiniLM-L6-cos-v1` | 0.0484 | −27% |
| `microsoft/unixcoder-base` | 0.0434 | −34.5% |
| CodeSearchNet `st-codesearch-distilroberta-base` | 0.0333 | −50% |

The winner, `e5-base-v2`, is the same size as mpnet and uses the same 512-token
window and pre-processing; it's trained specifically for query → passage
retrieval, and it beat mpnet by a third at no extra CPU cost. Bigger
general-purpose models also beat the MiniLM baseline (and stretching mpnet's
window from 384 to 512 tokens added +2.9% on its own), while every
code-specialized model lost. Every dropped model config stays in `config/` and
re-runs the same way (the HyDE and multi-view rows re-run with
`python -m prism.cli ablation --apps`), e.g.:

```bash
python -m prism.cli eval --config config/baseline.json \
    --output results/baseline_appsretrieval_results.json
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
    prism-code-search python -m prism.cli eval --config config/e5_base.json \
    --output results/appsretrieval_results.json
```

---

## The three differentiators

Built in order of expected payoff; each is an A/B against the baseline, kept
only if it wins on the scoreboard (`python -m prism.cli ablation`). On the real
split HyDE and multi-view lost, so the submission runs with both off.

1. **Query → code compilation (code-flavored HyDE).** NL queries and code live in
   different "languages." We compile the query into a short pseudo-code sketch
   and embed code-against-code, to close the modality gap on behavioral queries.
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
   delta_weight · delta)`, so `delta_weight=0` is exactly the naive vector. The
   delta term is meant to pull a version-specific query toward the right
   sibling; measured so far it ties naive retrieval, even on e5-base-v2 (see
   Limitations).
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
  query latency (end to end, and the ranking step alone), and index
  build/rebuild cost — what the hands-on round asks for.

Illustrative offline ablation (deterministic `hashing` backend on the bundled
synthetic set — the real numbers come from `--apps`):

| Config | NDCG@10 | ΔNDCG | Verdict |
|---|---:|---:|:--:|
| baseline | 0.7275 | — | keep |
| +multiview | 0.7587 | +0.0312 | keep |
| +hyde | 0.7568 | +0.0293 | keep |
| +hybrid (rrf) | 0.7336 | +0.0061 | keep |
| +all | 0.7647 | +0.0372 | keep |

(Offline, every lever wins. The real decision is remade on `AppsRetrieval`, and
it reversed two of these: **on the real split, both
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
config/                 e5_base.json (submitted), mpnet_512.json, mpnet.json, unixcoder.json,
                        baseline.json, frontload.json, retrieval_tuned.json,
                        code_model.json — results in docs/ablation_log.md
tests/                  78 tests, offline; MTEB integration auto-skips if absent
docs/                   architecture.md, ablation_log.md
submission/             the deck (PPTX + PDF)
Dockerfile, Makefile, requirements*.txt
```

## Design constraints (why it's built this way)

- **CPU-only, small model.** Gains have to come from choosing and feeding a
  small model well, not from scale.
- **No LLM in the ranking loop.** Any LLM use is offline pre-processing; eval-time
  cost is a string build + an embed, never a generation call.
- **Index rebuilds cheaply.** Versioned from the start (content-addressed cache),
  so P1's cross-version requirement is a first-class feature, not a bolt-on.

## Limitations (honest scope)

- **None of the pre/post differentiators is in the submitted config.** HyDE,
  multi-view and keyword front-loading all lost to plain MiniLM on the real
  split; they haven't yet been re-measured on top of e5, where they may
  behave differently (see `docs/ablation_log.md`).
- **Larger retrieval models are untested.** e5-base-v2 is 110M params; bigger
  ones (e.g. e5-large) may score higher but cost roughly 3× the CPU time.
- The evolutionary bonus is a **prototype with no measured gain yet**: the
  stable-core + version-delta representation ties naive full-text retrieval on
  both the offline backend and e5-base-v2. The cheap-rebuild P1 requirement is
  fully working (in the demo, a one-snippet change re-encodes 1 of 18 snippets).
- The offline `hashing` backend is lexical, so offline numbers under-represent
  what the real embedding model achieves on semantic/behavioral queries.
- Hybrid dense+BM25 scoring shapes the standalone retriever's ranking; MTEB scores
  the pure encoder, so hybrid is reported in ablations but off in the encoder
  submission unless folded into the vector.
