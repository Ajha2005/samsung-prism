"""Configuration for the retrieval pipeline.

Every knob that affects the leaderboard number lives here so an experiment is a
single, diffable config object. This is what makes "change one thing, measure,
keep or revert" mechanical rather than aspirational: the ablation runner mutates
one field at a time and records the config next to its NDCG@10 / MRR.

Configs load from a plain dict (or YAML/JSON file) and validate on
construction, so a typo fails loudly instead of silently disabling a feature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class BackendConfig:
    """Which embedding model turns text into vectors.

    ``kind`` selects the backend implementation:
      * ``"sentence_transformer"`` — the real competition backend (downloads a
        model from HuggingFace the first time; CPU inference).
      * ``"hashing"`` — a deterministic, dependency-light fallback that needs no
        network or model download. Used for offline tests and CI, and as a
        graceful degradation when the model cannot be loaded.
    """

    kind: str = "sentence_transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Candidate models we A/B in Phase 1 (score vs. CPU latency). Documented here
    # so the sweep is reproducible; see scripts/run_ablation.py.
    candidate_models: tuple = (
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "Salesforce/SFR-Embedding-Code-400M_R",
    )
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"
    # None leaves the model's own native max sequence length untouched — use
    # this for long-context models (e.g. code-specialized models with 2k-8k
    # native context) instead of clipping them down to the 256 default, which
    # is sized for small general models like all-MiniLM-L6-v2.
    max_seq_length: Optional[int] = 256
    # Dimensionality used by the hashing fallback backend.
    hashing_dim: int = 1024
    # Optional instruction prefixes some models expect (e.g. bge/e5 style).
    query_prompt: Optional[str] = None
    passage_prompt: Optional[str] = None
    # Some code-specialized models (e.g. jina-embeddings-v2-base-code) ship a
    # custom modeling class and require this to load via sentence-transformers.
    # Only set true for models you trust — it executes code from the model repo.
    trust_remote_code: bool = False

    def validate(self) -> None:
        if self.kind not in {"sentence_transformer", "hashing"}:
            raise ValueError(f"Unknown backend kind: {self.kind!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.hashing_dim <= 0:
            raise ValueError("hashing_dim must be positive")


@dataclass
class QueryConfig:
    """Query-side pre-processing and query->code compilation (HyDE)."""

    normalize: bool = True
    extract_keywords: bool = True
    classify_type: bool = True
    # Code-flavored HyDE: translate the NL query into a pseudo-code sketch and
    # embed code-against-code. ``hyde_weight`` blends the sketch embedding with
    # the raw-query embedding: 0.0 = pure query, 1.0 = pure sketch.
    hyde_enabled: bool = False
    hyde_weight: float = 0.5
    # Optional path to a precomputed offline sketch cache (query -> sketch).
    # Lets an LLM-generated sketch be used without any LLM in the eval loop.
    hyde_cache_path: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.hyde_weight <= 1.0:
            raise ValueError("hyde_weight must be in [0, 1]")


@dataclass
class SnippetConfig:
    """Snippet-side pre-processing and multi-view embedding."""

    normalize: bool = True
    # Multi-view: fuse embeddings of (raw code, AST-normalized code, id/token bag).
    multiview_enabled: bool = False
    view_weights: Dict[str, float] = field(
        default_factory=lambda: {"raw": 1.0, "ast": 0.6, "identifiers": 0.4}
    )
    # Long snippets: split into windows and pool. 0 disables chunking.
    max_chars: int = 4000
    chunk_overlap: int = 200

    def validate(self) -> None:
        if self.multiview_enabled and not self.view_weights:
            raise ValueError("multiview_enabled requires at least one view weight")
        if any(w < 0 for w in self.view_weights.values()):
            raise ValueError("view weights must be non-negative")
        if self.max_chars < 0:
            raise ValueError("max_chars must be >= 0")


@dataclass
class ScoringConfig:
    """Dense + sparse fusion at ranking time.

    Note: MTEB scores an *encoder* (vectors + cosine). Hybrid scoring is used by
    the standalone :class:`prism.pipeline.CodeRetriever` and the demo, where we
    control ranking directly. It is reported in ablations but does not change the
    pure-encoder MTEB submission unless ``embed_fusion`` folds a signal into the
    vector itself.
    """

    hybrid_enabled: bool = False
    # "rrf" (reciprocal rank fusion) or "linear" (weighted score sum).
    fusion: str = "rrf"
    dense_weight: float = 1.0
    sparse_weight: float = 0.5
    rrf_k: int = 60

    def validate(self) -> None:
        if self.fusion not in {"rrf", "linear"}:
            raise ValueError(f"Unknown fusion: {self.fusion!r}")


@dataclass
class VersioningConfig:
    """Cross-version / evolutionary retrieval (P1 + bonus)."""

    enabled: bool = False
    # Blend of stable-core and version-delta: emb = norm(core + delta_weight*delta).
    delta_weight: float = 0.5
    # How to pick the reference version a delta is measured against.
    reference: str = "first"  # "first" | "centroid"

    def validate(self) -> None:
        if self.reference not in {"first", "centroid"}:
            raise ValueError(f"Unknown reference: {self.reference!r}")
        if self.delta_weight < 0:
            raise ValueError("delta_weight must be >= 0")


@dataclass
class PipelineConfig:
    """Top-level config: one object fully describes an experiment."""

    name: str = "baseline"
    backend: BackendConfig = field(default_factory=BackendConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    snippet: SnippetConfig = field(default_factory=SnippetConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    # Where to cache the built index between runs (cheap rebuild for P1).
    cache_dir: str = ".prism_cache"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "PipelineConfig":
        self.backend.validate()
        self.query.validate()
        self.snippet.validate()
        self.scoring.validate()
        self.versioning.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        """A compact one-line signature for the ablation log."""
        parts = [f"model={Path(self.backend.model_name).name}"]
        if self.query.hyde_enabled:
            parts.append(f"hyde@{self.query.hyde_weight}")
        if self.snippet.multiview_enabled:
            parts.append("multiview")
        if self.scoring.hybrid_enabled:
            parts.append(f"hybrid:{self.scoring.fusion}")
        if self.versioning.enabled:
            parts.append(f"versioned@{self.versioning.delta_weight}")
        return f"{self.name} [" + ", ".join(parts) + "]"


def _from_dict(cls, data: Dict[str, Any]):
    """Recursively build a (nested) dataclass from a dict, ignoring extras."""
    if not is_dataclass(cls):
        return data
    # `from __future__ import annotations` makes f.type a string, so resolve the
    # real types (needed to know which fields are nested dataclasses).
    from typing import get_type_hints

    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            raise ValueError(f"Unknown config key {key!r} for {cls.__name__}")
        field_type = hints.get(key)
        if isinstance(value, dict) and is_dataclass(field_type):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(source: Optional[Any] = None) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a dict, a JSON/YAML path, or None.

    ``None`` returns the default baseline config.
    """
    if source is None:
        return PipelineConfig()
    if isinstance(source, PipelineConfig):
        return source.validate()
    if isinstance(source, dict):
        return _from_dict(PipelineConfig, source)
    # Treat as a path.
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # optional dependency
        except ImportError as exc:  # pragma: no cover - clear error path
            raise RuntimeError(
                "PyYAML is required to load YAML configs; use a .json file or "
                "`pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return _from_dict(PipelineConfig, data)
