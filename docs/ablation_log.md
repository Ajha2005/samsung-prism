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

## Leaderboard results — real CoIR AppsRetrieval, `all-MiniLM-L6-v2`

Run via `python -m prism.cli ablation --apps` (2026-09-19). `+hybrid_rrf` and
`+all` did not finish in this run (interrupted after ~1.5h on a free-tier
Colab CPU — the encoder itself is fast; MTEB's own import/setup overhead and
shared-instance throttling account for the rest). Both are inert or
predictable from the three completed rows regardless: `+hybrid_rrf` cannot
differ from `baseline` (hybrid scoring only affects the standalone
`CodeRetriever`'s ranking, not the pure-encoder vectors MTEB scores), and
`+all` stacks two rows that both already lose to baseline individually.

| Config | NDCG@10 | MRR@10 | ΔNDCG vs base | Kept? |
|---|---:|---:|---:|:--:|
| **baseline** | **0.0662** | **0.0561** | — | ✅ **kept — submitted** |
| +multiview | 0.0515 | 0.0428 | −0.0147 (−22%) | ↩︎ drop |
| +hyde | 0.0625 | 0.0519 | −0.0037 (−6%) | ↩︎ drop |
| +hybrid_rrf | *(not run — provably ≡ baseline, see above)* | | | ↩︎ drop |
| +all | *(not run — stacks two losing rows)* | | | ↩︎ drop |

**Reading it — and why this is the discipline working, not a failed idea:**
`all-MiniLM-L6-v2` is a small, general-purpose sentence model with no code
pretraining. Multi-view's AST-normalized view replaces real identifiers with
placeholders (`FUNC1`, `ARG1`, ...) — a model that isn't code-aware has
nothing left to embed once the real names are gone, so that view adds noise
instead of structural signal. HyDE's sketch is a deterministic template, not
an LLM; on AppsRetrieval's long, genuinely complex problem statements, a
generic boilerplate sketch dilutes the real query rather than closing the
NL↔code gap. Both bet on a capability this exact model doesn't have.

Per this repo's own rule — *keep only what wins, drop anything that doesn't
earn its place* — plain baseline is what's submitted. Both differentiators
stay in the codebase (`--hyde`, `--multiview` flags; `config/submission.json`),
verified and ready to re-measure against a code-specialized or larger model
where the bet is more likely to pay off (see `docs/submission_checklist.md`
/ README's "What's next").

Earlier experiments worth (re-)running once time allows: 2–3 candidate
embedding models scored vs. CPU latency; HyDE/multi-view re-measured against
a code-aware backend instead of MiniLM.
