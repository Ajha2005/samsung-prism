"""Command-line entry points: eval, demo, ablation.

Installed as ``prism-eval`` / ``prism-demo`` / ``prism-ablation`` (see
pyproject), and also runnable as ``python -m prism.cli <cmd>``.

Defaults are chosen so every command runs *offline* with the deterministic
backend; flags switch to the real sentence-transformers model + CoIR dataset for
the leaderboard run.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from prism.config import (
    BackendConfig,
    PipelineConfig,
    QueryConfig,
    ScoringConfig,
    SnippetConfig,
    load_config,
)


def _backend_config(args) -> BackendConfig:
    return BackendConfig(
        kind=args.backend,
        model_name=args.model,
        hashing_dim=args.hashing_dim,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )


def _build_config(args, name: str) -> PipelineConfig:
    if getattr(args, "config", None):
        return load_config(args.config)
    return PipelineConfig(
        name=name,
        backend=_backend_config(args),
        query=QueryConfig(hyde_enabled=getattr(args, "hyde", False)),
        snippet=SnippetConfig(multiview_enabled=getattr(args, "multiview", False)),
        scoring=ScoringConfig(hybrid_enabled=getattr(args, "hybrid", False)),
    )


def _add_common_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["sentence_transformer", "hashing"], default="sentence_transformer")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--hashing-dim", type=int, default=2048, dest="hashing_dim")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        dest="trust_remote_code",
        help="Allow loading models that ship custom modeling code (needed by some code-specialized models).",
    )
    p.add_argument("--config", default=None, help="Path to a JSON/YAML pipeline config (overrides flags).")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def eval_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prism-eval", description="Run the CoIR AppsRetrieval MTEB eval.")
    _add_common_backend_args(p)
    p.add_argument("--hyde", action="store_true")
    p.add_argument("--multiview", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--output", default="results/appsretrieval_results.json")
    args = p.parse_args(argv)

    from prism.eval.mteb_runner import run_apps_retrieval

    config = _build_config(args, name="submission")
    print(f"[prism-eval] config: {config.describe()}", file=sys.stderr)
    print(f"[prism-eval] running AppsRetrieval (downloads model + dataset on first run)...", file=sys.stderr)
    payload = run_apps_retrieval(config, output_path=args.output)
    print(f"[prism-eval] NDCG@10 = {payload.get('main_score_ndcg_at_10')}")
    print(f"[prism-eval] MRR@10  = {payload.get('mrr_at_10')}")
    print(f"[prism-eval] wrote {args.output}")
    return 0


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def demo_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prism-demo", description="Interactive/CLI retrieval demo on real queries.")
    _add_common_backend_args(p)
    p.add_argument("--hyde", action="store_true")
    p.add_argument("--multiview", action="store_true")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--top-k", type=int, default=3, dest="top_k")
    p.add_argument("--query", action="append", default=None, help="Query to run (repeatable). Omit for a default set.")
    p.add_argument("--versioned", action="store_true", help="Run the cross-version (evolutionary) demo instead.")
    args = p.parse_args(argv)

    if args.versioned:
        return _demo_versioned(args)

    from prism.data import load_synthetic_corpus
    from prism.pipeline import CodeRetriever
    from prism.telemetry.metrics import evaluate_retrieval

    corpus, queries, qrels = load_synthetic_corpus()
    config = _build_config(args, name="demo")
    print(f"[prism-demo] indexing {len(corpus)} snippets | config: {config.describe()}")
    retriever = CodeRetriever(config).index(corpus)
    print(f"[prism-demo] index built in {retriever.telemetry.index_build_seconds*1000:.1f} ms")

    demo_queries = args.query or [
        "find two numbers that sum to a target",
        "detect if a string is a palindrome",
        "shortest path in an unweighted graph",
        "remove duplicate words and count them",
    ]
    import time

    for q in demo_queries:
        t0 = time.perf_counter()
        hits = retriever.search(q, top_k=args.top_k)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\nQ: {q}   ({dt:.1f} ms)")
        for rank, hit in enumerate(hits, 1):
            first_line = hit.text.strip().splitlines()[0]
            print(f"  {rank}. [{hit.doc_id}]  score={hit.score:.4f}   {first_line}")

    # Also report leaderboard-style quality on the labeled synthetic set.
    results = retriever.batch_search(queries, top_k=min(100, len(corpus)))
    m = evaluate_retrieval(qrels, results, k_values=(1, 5, 10))
    print(
        f"\n[prism-demo] on {len(queries)} labeled queries: "
        f"NDCG@10={m['ndcg_at_10']:.4f}  MRR@10={m['mrr_at_10']:.4f}  "
        f"Recall@10={m['recall_at_10']:.4f}  P@1={m['precision_at_1']:.4f}"
    )
    print(f"[prism-demo] mean query latency: {retriever.telemetry.mean_query_latency_ms:.2f} ms "
          f"({retriever.telemetry.queries_per_second:.0f} q/s)")
    return 0


def _demo_versioned(args) -> int:
    import numpy as np

    from prism.data import load_synthetic_corpus, load_versioned_eval
    from prism.encoder import PrePostPipelineEncoder
    from prism.index.evolutionary import EvolutionaryEncoder, stable_core_and_delta
    from prism.index.versioned import VersionedIndex
    from prism.snippet.preprocess import clean_snippet
    from prism.telemetry.metrics import evaluate_retrieval

    config = _build_config(args, name="versioned-demo")
    config.versioning.enabled = True
    encoder = PrePostPipelineEncoder(config)

    # 1) Cheap versioned rebuild (P1): only changed snippets get re-encoded. ----
    corpus, _, _ = load_synthetic_corpus()
    index = VersionedIndex(encoder)
    t_full = index.add_version("v1", corpus)
    v2 = dict(corpus)
    changed = next(iter(v2))  # tweak one snippet
    v2[changed] = v2[changed] + "\n    # v2: added a guard clause\n"
    t_rebuild = index.add_version("v2", v2)
    speedup = (t_full / t_rebuild) if t_rebuild > 0 else float("inf")
    print("[prism-demo] Cross-version retrieval — P1 + evolutionary bonus\n")
    print("1) Cheap versioned index rebuild (only changed snippets re-encoded):")
    print(f"     v1: encoded {index.encoded_counts['v1']}/{len(corpus)} snippets  in {t_full*1000:6.1f} ms")
    print(f"     v2: encoded {index.encoded_counts['v2']}/{len(v2)} snippets  in {t_rebuild*1000:6.1f} ms "
          f"(cache hit {index.cache_hit_rate('v2')*100:.0f}%, ~{speedup:.0f}x faster rebuild)")

    # 2) The stable-core + version-delta decomposition, shown on one group. -----
    groups, queries, qrels = load_versioned_eval()
    gid = next(iter(groups))
    versions = groups[gid]
    print(f"\n2) stable-core + version-delta decomposition (group '{gid}'):")
    core0, _ = stable_core_and_delta(versions[0], versions)
    print(f"     shared core : {core0.strip().splitlines()[0] if core0.strip() else '(none)'}")
    for i, v in enumerate(versions):
        _, delta = stable_core_and_delta(v, versions)
        d = delta.strip().splitlines()[0] if delta.strip() else "(identical to core)"
        print(f"     v{i} delta    : {d}")

    # 3) Version-specific retrieval: does the RIGHT version rank #1? ------------
    delta_weight = max(1.0, config.versioning.delta_weight)
    evo = EvolutionaryEncoder(encoder, delta_weight=delta_weight)
    doc_ids: List[str] = []
    naive_rows, evo_rows = [], []
    for g, vs in groups.items():
        ev = evo.encode_group(vs)
        nv = encoder.encode_texts([clean_snippet(x) for x in vs], is_query=False)
        for i in range(len(vs)):
            doc_ids.append(f"{g}@v{i}")
            naive_rows.append(nv[i])
            evo_rows.append(ev[i])

    def _retrieve(mat):
        results = {}
        qvecs = encoder.encode_texts([queries[q] for q in queries], is_query=True)
        for qid, qvec in zip(queries, qvecs):
            scores = mat @ qvec
            results[qid] = {doc_ids[i]: float(scores[i]) for i in range(len(doc_ids))}
        return results

    naive_m = evaluate_retrieval(qrels, _retrieve(np.vstack(naive_rows)), k_values=(1, 3))
    evo_m = evaluate_retrieval(qrels, _retrieve(np.vstack(evo_rows)), k_values=(1, 3))
    print(f"\n3) Version-specific retrieval ({len(queries)} targeted queries, delta_weight={delta_weight}):")
    print(f"     {'representation':22s} {'P@1':>6s} {'NDCG@3':>8s}")
    print(f"     {'naive full-text':22s} {naive_m['precision_at_1']:6.3f} {naive_m['ndcg_at_3']:8.3f}")
    print(f"     {'stable-core + delta':22s} {evo_m['precision_at_1']:6.3f} {evo_m['ndcg_at_3']:8.3f}")
    if config.backend.kind == "hashing":
        print("\n   note: delta_weight=0 reduces to naive; the discrimination gain is realized with the")
        print("   semantic backend (--backend sentence_transformer). On the lexical fallback it holds parity.")
    return 0


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def ablation_main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prism-ablation", description="Run the ablation sweep and log a table.")
    _add_common_backend_args(p)
    p.add_argument("--apps", action="store_true", help="Score on real AppsRetrieval instead of the offline set.")
    p.add_argument("--md", default="results/ablation_log.md")
    p.add_argument("--json", default="results/ablation.json", dest="json_path")
    args = p.parse_args(argv)

    from prism.ablation import apps_evaluator, offline_evaluator, run_ablation

    bc = lambda: _backend_config(args)
    configs = [
        ("baseline", PipelineConfig(name="baseline", backend=bc())),
        ("+multiview", PipelineConfig(name="multiview", backend=bc(), snippet=SnippetConfig(multiview_enabled=True))),
        ("+hyde", PipelineConfig(name="hyde", backend=bc(), query=QueryConfig(hyde_enabled=True, hyde_weight=0.5))),
        ("+hybrid_rrf", PipelineConfig(name="hybrid", backend=bc(), scoring=ScoringConfig(hybrid_enabled=True, fusion="rrf"))),
        ("+all", PipelineConfig(
            name="all", backend=bc(),
            snippet=SnippetConfig(multiview_enabled=True),
            query=QueryConfig(hyde_enabled=True, hyde_weight=0.5),
            scoring=ScoringConfig(hybrid_enabled=True, fusion="rrf"))),
    ]

    if args.apps:
        evaluator = apps_evaluator(output_dir="results/ablation")
    else:
        from prism.data import load_synthetic_corpus
        corpus, queries, qrels = load_synthetic_corpus()
        evaluator = offline_evaluator(corpus, queries, qrels)

    log = run_ablation(configs, evaluator)
    log.save(args.md, args.json_path)
    print(log.to_markdown())
    print(f"\n[prism-ablation] wrote {args.md} and {args.json_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m prism.cli {eval|demo|ablation} [options]", file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    dispatch = {"eval": eval_main, "demo": demo_main, "ablation": ablation_main}
    if cmd not in dispatch:
        print(f"unknown command: {cmd!r} (choose from eval, demo, ablation)", file=sys.stderr)
        return 2
    return dispatch[cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
