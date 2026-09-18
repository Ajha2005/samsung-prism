"""A local, HuggingFace-free MTEB retrieval task.

Used to verify the whole MTEB integration end-to-end without network access: it
carries an in-memory corpus/queries/qrels in MTEB's "v1" dict format and
converts them to the internal v2 dataset on load. Running the real
``mteb.evaluate`` against this task exercises exactly the same code path as the
CoIR ``AppsRetrieval`` run — dense search, similarity, and the official trec_eval
metrics — so a green run here means the encoder plugs into MTEB correctly.
"""

from __future__ import annotations

from typing import Dict, Mapping

from mteb import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval


def build_synthetic_task(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    relevant_docs: Mapping[str, Mapping[str, int]],
    *,
    name: str = "PrismSyntheticCodeRetrieval",
    split: str = "test",
) -> AbsTaskRetrieval:
    """Build an :class:`AbsTaskRetrieval` preloaded with local data."""

    metadata = TaskMetadata(
        name=name,
        description="Local synthetic code-retrieval task for offline MTEB integration tests.",
        reference="https://github.com/embeddings-benchmark/mteb",
        dataset={"path": "local/synthetic", "revision": "1.0.0"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[split],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2026-01-01", "2026-01-01"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )

    corpus_v1 = {split: {doc_id: {"text": text, "title": ""} for doc_id, text in corpus.items()}}
    queries_v1 = {split: dict(queries)}
    qrels_v1 = {split: {q: dict(rels) for q, rels in relevant_docs.items()}}

    class _SyntheticRetrieval(AbsTaskRetrieval):
        metadata = None  # set on the instance below

        def load_data(self, *args, **kwargs) -> None:
            if self.data_loaded:
                return
            self.corpus = corpus_v1
            self.queries = queries_v1
            self.relevant_docs = qrels_v1
            self.convert_v1_dataset_format_to_v2(kwargs.get("num_proc"))
            self.data_loaded = True

    task = _SyntheticRetrieval.__new__(_SyntheticRetrieval)
    # Bypass registry lookups: set metadata directly, then init.
    type(task).metadata = metadata
    task.__init__()
    return task
