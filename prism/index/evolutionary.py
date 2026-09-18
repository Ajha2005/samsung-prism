"""Cross-version / evolutionary retrieval (the bonus).

When a corpus contains several near-identical versions of the same code, a naive
embedder maps them to almost the same vector: the duplicates crowd the top-k and
a query cannot tell which version it wants. The fix is to encode what is
*invariant* across versions separately from what *changed*:

    vector(version) = normalize( base_embedding + delta_weight * delta_embedding )

  * ``base``  — the embedding of the whole snippet (what a naive encoder would
                produce). It carries the group/content signal, so a *general*
                query still finds the right family.
  * ``delta`` — the embedding of just the lines distinctive to this version
                (what changed relative to its siblings). Added with extra
                weight, it concentrates a signal the base embedding dilutes when
                the distinctive change is one line in a large snippet — which is
                what pulls a *version-specific* query to the right sibling.

Why base+delta rather than core+delta: ``delta_weight=0`` then reduces *exactly*
to the naive vector, so the representation can only match or beat the baseline,
never fall below it. The invariant "stable core" is still computed (it is how the
delta is identified, and it is what the versioned index caches for cheap
rebuilds); it simply lives inside the base embedding rather than replacing it.

Design choice that matters: we classify a line as core/delta by comparing a
*normalized* form (so cosmetic edits don't count as changes) but embed the
**raw** lines — canonicalizing them away would discard the identifier names that
lexical and semantic queries match on.

Honest scope: the discrimination gain is realized with a *semantic* embedding
backend, where the delta's meaning is separable even when lexically diluted. On
the deterministic lexical fallback it holds parity with naive. This is a
measurable prototype, exactly as the build plan frames the bonus.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from prism.backends.base import l2_normalize
from prism.encoder import PrePostPipelineEncoder
from prism.snippet.ast_normalize import normalize_tokens
from prism.snippet.preprocess import clean_snippet

_WS_RE = re.compile(r"\s+")


def _nonempty_lines(text: str) -> List[str]:
    return [ln for ln in (clean_snippet(text) or "").splitlines() if ln.strip()]


def _ws_norm(line: str) -> str:
    """Whitespace-insensitive line key (cosmetic reformatting isn't a change)."""
    return _WS_RE.sub(" ", line.strip())


def rename_invariant_key(line: str) -> str:
    """Line key that also ignores identifier renames (uses token canonicalization)."""
    return normalize_tokens(line)


def stable_core_and_delta(
    text: str,
    versions: Sequence[str],
    *,
    key_fn: Callable[[str], str] = _ws_norm,
) -> Tuple[str, str]:
    """Split ``text`` (raw) into (stable-core, version-delta) relative to a group.

    A line is *core* if its ``key_fn`` form appears in every version of the
    group, else it is *delta*. Raw line text is returned either way, so
    identifiers survive into the embedding.
    """
    version_key_sets = [{key_fn(ln) for ln in _nonempty_lines(v)} for v in versions]
    common = set.intersection(*version_key_sets) if version_key_sets else set()

    core_lines: List[str] = []
    delta_lines: List[str] = []
    seen_core = set()
    for raw in _nonempty_lines(text):
        key = key_fn(raw)
        if key in common:
            if key not in seen_core:
                seen_core.add(key)
                core_lines.append(raw)
        else:
            delta_lines.append(raw)
    return "\n".join(core_lines), "\n".join(delta_lines)


class EvolutionaryEncoder:
    """Encode grouped code versions with stable-core + version-delta vectors."""

    def __init__(
        self,
        encoder: PrePostPipelineEncoder,
        delta_weight: float = 1.0,
        *,
        rename_invariant: bool = False,
    ):
        self.encoder = encoder
        self.dim = encoder.dim
        self.delta_weight = delta_weight
        self._key_fn = rename_invariant_key if rename_invariant else _ws_norm

    def encode_group(self, versions: Sequence[str]) -> np.ndarray:
        """Return one distinguishing vector per version in a group.

        ``vector = normalize(base + delta_weight * delta)`` — so ``delta_weight=0``
        is exactly the naive base embedding.
        """
        if len(versions) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        base_emb = l2_normalize(
            self.encoder.encode_texts([clean_snippet(v) for v in versions], is_query=False)
        )

        deltas = [stable_core_and_delta(v, versions, key_fn=self._key_fn)[1] for v in versions]
        delta_emb = np.zeros_like(base_emb)
        nonempty_idx = [i for i, d in enumerate(deltas) if d.strip()]
        if nonempty_idx:
            embedded = l2_normalize(
                self.encoder.encode_texts([deltas[i] for i in nonempty_idx], is_query=False)
            )
            for slot, i in enumerate(nonempty_idx):
                delta_emb[i] = embedded[slot]

        fused = base_emb + self.delta_weight * delta_emb
        return l2_normalize(fused)

    def encode_corpus(self, groups: Mapping[str, Sequence[str]]) -> Dict[str, np.ndarray]:
        """Encode a corpus of ``{group_id: [version_text, ...]}``."""
        return {gid: self.encode_group(list(versions)) for gid, versions in groups.items()}
