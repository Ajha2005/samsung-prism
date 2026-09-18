# Ablation log

> The scoreboard is the product. One change, one measurement, keep or revert.
> This log is the experiment record, the regression guard, and half the deck.

## Method
- **Never change two things at once.** Each row differs from `baseline` by one
  feature (`PipelineConfig` is one diffable object; the config travels in the row).
- **Baseline stays live.** Every ΔNDCG is measured against `baseline`, not the
  previous row, so slow drift can't hide a regression.
- **Keep only what wins.** A feature is kept iff it beats baseline NDCG@10.
- Regenerate anytime:
  - offline (deterministic, no network): `python -m prism.cli ablation --backend hashing`
  - leaderboard (real CoIR AppsRetrieval): `python -m prism.cli ablation --apps`

## Offline illustration (deterministic `hashing` backend, bundled synthetic set)

This validates the *harness and the direction of each lever*, not the final
leaderboard magnitude (the lexical backend under-represents semantic gains).

| Config | Signature | NDCG@10 | MRR@10 | ΔNDCG vs base | Verdict |
|---|---|---:|---:|---:|:--:|
| baseline | `model=all-MiniLM-L6-v2` | 0.7370 | 0.6870 | — | ✅ keep |
| +multiview | `+ multiview` | 0.7784 | 0.7256 | +0.0414 | ✅ keep |
| +hyde | `+ hyde@0.5` | 0.7433 | 0.6963 | +0.0062 | ✅ keep |
| +hybrid_rrf | `+ hybrid:rrf` | 0.7345 | 0.6866 | −0.0026 | ↩︎ drop |
| +all | `+ hyde@0.5, multiview, hybrid:rrf` | 0.7711 | 0.7333 | +0.0340 | ✅ keep |

Reading it: multi-view is the strongest single lever here; HyDE helps modestly;
hybrid dense+BM25 *hurt on this set* and is dropped — exactly the decision the
methodology is meant to force. All decisions are remade on `AppsRetrieval`.

## Leaderboard results (fill in from `--apps`)

| Config | NDCG@10 | MRR | Latency | Kept? |
|---|---:|---:|---:|:--:|
| baseline (all-MiniLM-L6-v2) | _tbd_ | _tbd_ | _tbd_ | |
| + multi-view | _tbd_ | _tbd_ | _tbd_ | |
| + query→code (HyDE) | _tbd_ | _tbd_ | _tbd_ | |
| + candidate model B / C | _tbd_ | _tbd_ | _tbd_ | |

Experiments worth running (double as slide content): dense-only vs hybrid;
with/without HyDE; single- vs multi-view; 2–3 candidate models scored vs CPU
latency.
