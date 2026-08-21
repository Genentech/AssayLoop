"""LLM-guided Nearest-Neighbour model (LLMNN).

Implements the strategy from Gupta et al. (arXiv 2509.21403): the LLM
proposes ``n_centers`` cluster-center genes each step, then candidates
are scored by cosine similarity to the nearest center in a gene-
embedding space (default: DepMap/Achilles via PRESAGE).

The LLM handles exploration (picking diverse regions of biology);
the NN expansion handles exploitation (filling the batch from the
chosen regions deterministically).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ..data.gene_embeddings.base import GeneEmbeddingProvider
from ..data.gene_embeddings.ortholog import map_mgi_to_hgnc
from ..data.gene_embeddings.presage import (
    MissingPresageCache,
    MissingPresageSource,
    PresageGeneEmbedding,
)
from ..llm.client import LLMClientConfig, complete
from assaybench.llm.parse_genes import extract_gene_list, organism_suffix


_SYSTEM_PROMPT = (
    "You are a computational biologist guiding a sequential gene-screening "
    "experiment. Given the screen description and results from previous "
    "rounds (confirmed hits and non-hits), propose {n_centers} gene symbols "
    "as cluster centers for the next round of experiments.\n\n"
    "Each center should represent a distinct biological pathway, gene "
    "family, or functional module that you believe is enriched for hits "
    "in this screen. Choose diverse centers that cover different "
    "hypothesized mechanisms — do not cluster them in one pathway.\n\n"
    "Return ONLY a comma-separated list of exactly {n_centers} gene "
    "symbols, best candidates first. No prose, no numbers, no formatting."
)

_USER_TEMPLATE = (
    "=== Screen ===\n{screen_block}\n\n"
    "=== Already validated examples ===\n{examples}\n\n"
    "=== Task ===\n"
    "Propose exactly {n_centers} cluster-center gene symbols that are "
    "likely proximal to hits in this screen. These centers will be used "
    "to query nearest neighbours in an embedding space, so pick genes "
    "that sit at the heart of hit-enriched regions. {organism_suffix}"
)


def _trim_question(q: str) -> str:
    start = q.find("## Experimental Context")
    if start != -1:
        q = q[start:]
    end = q.find("## Required Output Format")
    if end != -1:
        q = q[:end]
    return q.strip()


def _screen_block(ctx: dict[str, Any]) -> str:
    q = (ctx.get("question") or "").strip()
    if q:
        return _trim_question(q)
    parts = []
    for label, key in [
        ("Phenotype", "phenotype"),
        ("Cell line", "cell_line"),
        ("Cell type", "cell_type"),
        ("Organism", "organism"),
        ("Library methodology", "library_methodology"),
        ("Condition", "condition_clause"),
    ]:
        v = ctx.get(key)
        if v:
            parts.append(f"  {label}: {v}")
    return "\n".join(parts) or "  (no metadata)"


def _format_examples(observations: list[Observation]) -> str:
    if not observations:
        return "  (none acquired yet)"
    hits = [str(o.candidate) for o in observations
            if isinstance(o.label, dict) and o.label.get("hit")]
    misses = [str(o.candidate) for o in observations
              if isinstance(o.label, dict) and not o.label.get("hit")]
    return (
        f"  Confirmed HITS ({len(hits)}):\n    "
        f"{', '.join(hits) or '(none)'}\n"
        f"  Confirmed NON-hits ({len(misses)}):\n    "
        f"{', '.join(misses) or '(none)'}"
    )


def _resolve_provider(
    source: str,
    organism: str | None = None,
) -> GeneEmbeddingProvider:
    normalizer = None
    if organism and "musculus" in str(organism).lower():
        normalizer = map_mgi_to_hgnc
    try:
        return PresageGeneEmbedding(source=source, gene_normalizer=normalizer)
    except (MissingPresageCache, MissingPresageSource) as exc:
        raise RuntimeError(
            f"LLMNN requires gene embeddings but PRESAGE source "
            f"{source!r} is unavailable: {exc}. "
            f"Run scripts/fetch_presage_cache.sh to download."
        ) from exc


class LLMNNModel(Model):
    """LLM-guided nearest-neighbour model.

    The LLM proposes ``n_centers`` cluster centers, then all candidates
    are scored by cosine similarity to their nearest center in the
    chosen gene-embedding space.

    Args:
        llm: LLM client config.
        embedding_source: PRESAGE source key (default ``"depmap"``
            for Achilles/DepMap CRISPR gene-effect embeddings).
        n_centers: cluster centers the LLM proposes per step.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        llm: LLMClientConfig | None = None,
        embedding_source: str = "depmap",
        n_centers: int = 5,
        seed: int = 0,
    ):
        self.llm = llm or LLMClientConfig()
        self.embedding_source = str(embedding_source)
        self.n_centers = int(n_centers)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._provider: GeneEmbeddingProvider | None = None
        self._last_meta: dict[str, Any] = {}

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_meta = {}

    def name(self) -> str:
        cfg = self.llm.resolved()
        return f"llmnn/{cfg.provider}:{cfg.model}/{self.embedding_source}"

    def coverage(self, candidates: list) -> float | None:
        if self._provider is None:
            return None
        _, mask = self._provider.embed_batch(candidates)
        return float(mask.mean())

    def _ensure_provider(
        self, task_context: dict[str, Any] | None,
    ) -> GeneEmbeddingProvider:
        if self._provider is None:
            organism = (task_context or {}).get("organism")
            self._provider = _resolve_provider(
                self.embedding_source, organism=organism,
            )
        return self._provider

    def _ask_llm_for_centers(
        self,
        observations: list[Observation],
        task_context: dict[str, Any],
    ) -> list[str]:
        """Call the LLM to propose cluster center genes."""
        ctx = task_context or {}
        system = _SYSTEM_PROMPT.format(n_centers=self.n_centers)
        user = _USER_TEMPLATE.format(
            screen_block=_screen_block(ctx),
            examples=_format_examples(observations),
            n_centers=self.n_centers,
            organism_suffix=organism_suffix(ctx),
        )
        text = complete(self.llm, prompt=user, system=system)
        return extract_gene_list(text)

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        if not candidates:
            return ModelPrediction(
                scores={}, uncertainty=None,
                metadata={"name": self.name()},
            )

        ctx = task_context or {}
        provider = self._ensure_provider(ctx)

        # --- Step 1: LLM proposes cluster centers ---
        try:
            raw_centers = self._ask_llm_for_centers(observations, ctx)
        except Exception as e:
            return ModelPrediction(
                scores={},
                uncertainty=None,
                metadata={
                    "name": self.name(),
                    "error": f"{type(e).__name__}: {e}",
                },
            )

        # --- Step 2: Embed cluster centers ---
        center_vecs: list[np.ndarray] = []
        center_names: list[str] = []
        for g in raw_centers[: self.n_centers]:
            v = provider.embed(g)
            if v is not None:
                center_vecs.append(v)
                center_names.append(g)
        if not center_vecs:
            # None of the LLM's centers are in the embedding space.
            return ModelPrediction(
                scores={},
                uncertainty=None,
                metadata={
                    "name": self.name(),
                    "n_valid_centers": 0,
                    "raw_centers": raw_centers[: self.n_centers],
                },
            )

        C = np.stack(center_vecs, axis=0).astype(np.float32)  # (nc, D)
        # L2-normalise centers for cosine similarity.
        c_norms = np.linalg.norm(C, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1
        C_n = C / c_norms

        # --- Step 3: Embed candidates ---
        X_cand, known_mask = provider.embed_batch(candidates)
        q_norms = np.linalg.norm(X_cand, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1
        X_n = (X_cand / q_norms).astype(np.float32)

        # --- Step 4: Score = max cosine similarity to any center ---
        # S shape: (n_candidates, n_centers)
        S = X_n @ C_n.T
        max_sim = S.max(axis=1)  # (n_candidates,)

        # Shift scores to [0, 1]: sim in [-1, 1] -> (sim + 1) / 2.
        norm_scores = (max_sim + 1.0) / 2.0

        scores: dict[Any, float] = {}
        for i, c in enumerate(candidates):
            if known_mask[i]:
                scores[c] = float(norm_scores[i])

        self._last_meta = {
            "n_valid_centers": len(center_names),
            "center_genes": center_names,
            "raw_centers": raw_centers[: self.n_centers],
            "embedding_source": self.embedding_source,
            "coverage": float(known_mask.mean()),
        }
        return ModelPrediction(
            scores=scores,
            uncertainty=None,
            metadata={"name": self.name(), **self._last_meta},
        )


__all__ = ["LLMNNModel"]
