# Submission checklist (Theme 1)

> The deck states plainly: **not following the submission guideline leads to
> direct disqualification.** Treat this list as pass/fail. The file-naming rule
> is called out twice in the deck as a disqualifier — double-check every name
> against the Google Form before submitting.

## Hard deadline
- **Final submission: 30 Sep, 11:59 PM** (extended from 25 Sep — verify against
  the Google Form). Registration closed 16 Sep — confirm you are registered before
  anything else; the final link only goes to registered teams.
- After submission: top 15 announced 9 Oct · final demo 15 Oct · results 24 Oct.
  The hands-on round **re-runs your code** and evaluates P1 + the bonus live —
  keep the repo reproducible.

## Deliverables (pass/fail)

- [ ] Registered via the Google Form (team name format `CollegeName_TeamName`).
- [x] GitHub repo with a working prototype. *(this repo)*
- [x] `README.md` with reproducible setup + Docker files + run instructions.
- [x] Dockerfile with a one-command run.
- [ ] **Release tag named exactly `PRISM_GENAI_HACKATHON_Y2026`** on the final
      commit — the tagged commit is what gets judged.
- [ ] MTEB results JSON (`appsretrieval_results.json`) uploaded as a **release
      artifact**. Generate it with:
      `python -m prism.cli eval --config config/e5_base.json --output results/appsretrieval_results.json`
      *(needs network for the model + CoIR dataset; run in an environment with
      HuggingFace access — ~1.5 h on a free Kaggle CPU.)* **Use `config/e5_base.json`**
      — the real ablation (`docs/ablation_log.md`) measured ten configs;
      e5-base-v2 scored best (NDCG@10 0.1151 vs baseline 0.0662).
      The written file is verified to be exactly `task_result.to_dict()` (no
      wrapper) — matching the problem statement's own reference snippet
      byte-for-byte, so it parses however the screening pipeline expects.
- [ ] Everything referenced (PPT, demo video, docs) present in the tagged commit.
- [ ] PPT / PDF named `CollegeName_TeamName_Submission_ppt` covering: theme ID,
      project title, team details, problem statement in your own words, solution +
      architecture diagram, tools/tech stack, innovation highlights, results,
      limitations. *(architecture diagram + tech stack + innovation + limitations
      are in `README.md` and `docs/architecture.md`.)*
- [ ] Demo video ≤ 5 minutes (YouTube/Drive link) — show the solution working on
      real queries **and how fast it is**, not just the numbers.
      (`python -m prism.cli demo` prints per-query latency and throughput.)
- [ ] One submission per team via the Google Form by the deadline.

## What this repo already provides
- Reproducible pipeline + MTEB wiring verified end-to-end (offline synthetic task
  + trec_eval-matching metrics), **and the submission file format independently
  verified against the problem statement's own reference snippet.**
- Real leaderboard results, not just offline estimates: ten configs
  measured on the actual CoIR AppsRetrieval test split. Submitted:
  `config/e5_base.json` (NDCG@10 = **0.1151**, MRR@10 = **0.0986**, +73.9%
  over the MiniLM baseline). Full record and reasoning in `docs/ablation_log.md`.
- P1 cheap versioned rebuild (working) + evolutionary-retrieval prototype.
- Operational telemetry (precision@k, recall, latency, index build/rebuild cost).
- 78 offline tests; Docker image whose default run is tests + demo.

## Human still needs to
1. Register / confirm registration on the Google Form.
2. ~~Run the leaderboard eval~~ — done on Kaggle: e5-base scored
   NDCG@10=0.1151, MRR@10=0.0986. Rename Kaggle's
   `e5_appsretrieval_results.json` (not the `.debug.json`) to
   `appsretrieval_results.json` and attach it to the release.
3. Record the ≤5-min demo video; put its link and the deck (PPT + PDF) in the
   repo.
4. Re-create the release tag `PRISM_GENAI_HACKATHON_Y2026` on that final
   commit (the current tag predates the deck, the video link and e5-base).
5. Submit via the Google Form.
