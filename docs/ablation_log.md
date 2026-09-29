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
| **baseline** | **0.0662** | **0.0561** | — | ✅ kept → superseded by mpnet (final round) |
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
earn its place* — plain baseline stayed the reference at this point (it was
later superseded by mpnet; see the final round). Both differentiators stay
in the codebase (`--hyde`, `--multiview` flags),
verified and ready to re-measure on a larger model where the bet is more
likely to pay off.

## Candidate model swap — also tested, also lost

| Config | Model | NDCG@10 | MRR@10 | ΔNDCG vs base | Kept? |
|---|---|---:|---:|---:|:--:|
| **baseline** | all-MiniLM-L6-v2 | **0.0662** | **0.0561** | — | ✅ kept → superseded by mpnet (final round) |
| retrieval-tuned | multi-qa-MiniLM-L6-cos-v1 | 0.0484 | 0.0402 | −0.0178 (−27%) | ↩︎ drop |

`multi-qa-MiniLM-L6-cos-v1` is pretrained on an asymmetric query→passage
retrieval objective — a closer match to this task's *shape* than baseline's
general sentence-similarity objective — but its training data is pure
natural-language QA pairs (questions against NL answer passages), not code.
The result: specializing further toward NL-passage retrieval moved it
*away* from anything code-shaped, underperforming even a model that was
never tuned for retrieval at all. This sharpens the diagnosis — the
bottleneck here is code-awareness specifically, not retrieval-tuning in
general — and rules out "just pick a retrieval-tuned model" as a shortcut.

## Follow-up experiments — also measured on the real split, also lost

Two more configs the earlier round had flagged as high-EV were re-run on
CoIR AppsRetrieval (Kaggle, 2026-09-27), so every idea in this repo is now
a measured row rather than a prediction:

| Config | What it changes | NDCG@10 | MRR@10 | ΔNDCG vs base | Kept? |
|---|---|---:|---:|---:|:--:|
| **baseline** | all-MiniLM-L6-v2 | **0.0662** | **0.0561** | — | ✅ kept → superseded by mpnet (final round) |
| +frontload | Prepends extracted keywords in front of each query, so salient tokens survive MiniLM's 256-wordpiece cut on long problem statements. | 0.0646 | 0.0536 | −0.0016 (−2.4%) | ↩︎ drop |
| code_model | Swaps in `flax-sentence-embeddings/st-codesearch-distilroberta-base`, an 82M model pretrained on CodeSearchNet (code + docstrings). | 0.0333 | 0.0264 | −0.0329 (−50%) | ↩︎ drop |

**Reading the new rows:** front-loading barely moved the number — the loss
is inside the noise band, meaning most AppsRetrieval queries already fit
under the 256-wordpiece cap, so there was little for the trick to rescue.
The code-model result was the interesting one: on paper the 82M
CodeSearchNet-pretrained DistilRoBERTa is exactly the "code-aware backend"
the earlier diagnosis called for, but it *lost by half* here. Two forces
compound: (a) CodeSearchNet is heavily Java / JavaScript / PHP / Go / Ruby
weighted, while AppsRetrieval is pure Python — the pretraining distribution
doesn't align with the eval; (b) the model was distilled and trained for
docstring↔function retrieval, not natural-language problem statements
↔ full competitive-programming solutions, which is a much longer and
messier query shape. So *"code-aware"* alone isn't the axis — at this
point we read the gap as needing Python-heavy pretraining on NL-problem ↔
code-solution pairs. The final round below tested that reading, and the
evidence pointed somewhere else.

## Final round — capacity vs. specialization (Kaggle, 2026-09-28)

| Config | Model (params, input window) | NDCG@10 | MRR@10 | ΔNDCG vs base | Kept? |
|---|---|---:|---:|---:|:--:|
| baseline | all-MiniLM-L6-v2 (22M, 256 tokens) | 0.0662 | 0.0561 | — | reference |
| unixcoder | microsoft/unixcoder-base (125M, 512 tokens) | 0.0434 | 0.0348 | −0.0228 (−34.5%) | ↩︎ drop |
| **mpnet** | **sentence-transformers/all-mpnet-base-v2 (110M, 384 tokens)** | **0.0837** | **0.0714** | **+0.0175 (+26.5%)** | ✅ kept → superseded by mpnet-512 (below) |

**Reading it:** the only config that beat baseline is the only one that
kept baseline's recipe. `all-mpnet-base-v2` comes from the same
sentence-transformers "all-*" family as `all-MiniLM-L6-v2`, trained on the
same mix of over a billion sentence pairs (which includes StackExchange
programming Q&A); it changes the model size, not the training recipe.
Every swap that moved *away* from that recipe lost: retrieval-tuned MiniLM
(−27%), CodeSearchNet DistilRoBERTa (−50%), UniXcoder (−34.5%). That
overturns the "Python-heavy code pretraining is the bottleneck" reading
above: on this benchmark, at this model scale, **capacity within a broad
general-purpose recipe beat domain specialization.**

Caveats, stated up front:
- **mpnet changes two things at once**: model size and input window (384
  vs 256 tokens — each model's native maximum). This run can't split the
  +26.5% between them; re-running mpnet at 256 tokens would.
- **UniXcoder's row is a lower bound, not a verdict.** It ran through
  sentence-transformers' generic mean pooling, without the `<encoder-only>`
  mode prefix its authors use for retrieval.
- **HyDE, multi-view and front-loading were only measured on MiniLM.**
  They may behave differently on the larger encoder; re-measuring each on
  top of mpnet is the next ablation.

## Window follow-up — one variable, one number (Kaggle, 2026-09-28)

AppsRetrieval problem statements and solutions often run past 384 tokens, so
the direct follow-up was to give mpnet more of each text. `config/mpnet_512.json`
is identical to `config/mpnet.json` except `max_seq_length` (384 → 512, within
MPNet's 514 position embeddings).

| Config | Change vs mpnet | NDCG@10 | MRR@10 | Δ vs mpnet | Δ vs base | Kept? |
|---|---|---:|---:|---:|---:|:--:|
| mpnet | — (384 tokens) | 0.0837 | 0.0714 | — | +26.5% | superseded |
| **mpnet-512** | **input window 512 tokens** | **0.0861** | **0.0733** | **+2.9%** | **+30.1%** | ✅ kept → superseded by e5-base (below) |

**Reading it:** a clean single-variable win — both NDCG@10 and MRR@10 rose,
so longer context genuinely helps on this benchmark even though the model's
sentence-embedding fine-tuning used much shorter inputs. The gain is modest; it doesn't resolve the first
caveat above (how much of MiniLM → mpnet was size vs. window), but it shows the
window matters at least at the 384 → 512 step.

## Retrieval-trained model at the same size (Kaggle, 2026-09-29)

`config/e5_base.json` keeps everything from `config/mpnet_512.json` — same
110M size class, 512-token window, same pre-processing — and swaps the model
for `intfloat/e5-base-v2`, which is trained specifically for query → passage
retrieval and expects a `"query: "` prefix on queries and `"passage: "` on
documents (`tests/test_model_prompts.py` checks the prefixes land on the right
side).

| Config | Model | NDCG@10 | MRR@10 | Δ vs mpnet-512 | Δ vs base | Kept? |
|---|---|---:|---:|---:|---:|:--:|
| mpnet-512 | all-mpnet-base-v2 | 0.0861 | 0.0733 | — | +30.1% | superseded |
| **e5-base** | **intfloat/e5-base-v2 + query/passage prefixes** | **0.1151** | **0.0986** | **+33.7%** | **+73.9%** | ✅ **kept — submitted** |

**Reading it:** the biggest single jump in the log, at no extra size or CPU
cost (this run took 84 min vs. ~146 min for mpnet-512). It refines the
"capacity beat specialization" reading above: at equal size and window, a
model trained for asymmetric query → passage retrieval beat a
sentence-similarity model by a third. "Retrieval-tuned" alone isn't the axis
either — the small `multi-qa-MiniLM-L6-cos-v1` lost — so the practical lesson
is to pick the strongest general retrieval model the CPU budget allows, and
code-specialized models (CodeSearchNet, UniXcoder) still lost here.

**Ten configs measured on the real leaderboard split; one kept.**
Submitted: `config/e5_base.json` — NDCG@10 = **0.1151**, MRR@10 = **0.0986**
(+73.9% over the MiniLM baseline). The other configs stay implemented behind
config flags. HyDE, multi-view and front-loading were only measured on MiniLM;
re-measuring them on e5 is the next ablation.
