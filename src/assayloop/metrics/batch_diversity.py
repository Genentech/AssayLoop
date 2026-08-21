"""Batch diversity as a :class:`~assaybench.core.Metric` the loop can carry.

The statistics themselves are defined in
:mod:`assaybench.benchmark.diversity` -- Vendi score, mean pairwise centred
cosine distance, pathway Jaccard -- and this module does not reimplement any of
them. What it adds is the plumbing the loop needs and the package deliberately
does not have: resolving the default embedding provider and gene sets from this
repo's configuration, and caching the random-draw baseline against a
:class:`~assayloop.data.gene_embeddings.GeneEmbeddingProvider` (assaybench's
functions calibrate against its own ``GeneEmbeddings``, which enumerates its
gene universe up front; a provider reads a 3.4 GB cache lazily and does not).

Three complementary views of "how spread out is this batch":

1. **Embedding geometry (de-anisotropised).** Mean pairwise cosine distance in
   the gene-embedding space, but with the global mean vector subtracted first.
   Text-style embeddings (GenePT/ada) are highly anisotropic — random gene
   pairs have cosine ~0.82, so raw ``(1-cos)/2`` collapses every batch into a
   tiny band near 0.085 and is essentially constant. Centering restores the
   full dynamic range. Reported relative to a random genome draw as well.

2. **Vendi score.** The exponentiated Shannon entropy of the (normalised)
   embedding-kernel eigenvalues — an *effective number of distinct genes* in
   the batch. 1 = all collinear, ``n`` = mutually orthogonal. Far more
   interpretable than a raw cosine distance.

3. **Pathway set-coverage.** Embedding-free: mean pairwise Jaccard overlap of
   each gene's MSigDB pathway/GO-term membership, reported as a diversity
   (``1 - overlap``) and relative to a random draw. Captures literal
   "do these genes hit the same pathways" rather than embedding geometry.

Both the embedding views and the pathway view need reference data that is
downloaded rather than shipped. If it is absent, this metric raises. It does
not substitute a stand-in provider and it does not quietly drop the keys:
either produces a table that looks complete and is not. To run without a view,
disable it explicitly -- pass ``gene_sets=None`` for the pathway view, or a
provider of your choosing for the embedding views.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assaybench.benchmark.diversity import (
    mean_pairwise_cosine_distance,
    pathway_diversity,
    vendi_score,
)
from assaybench.core.metric import Metric
from assaybench.core.types import ModelPrediction, Observation

from ..data.gene_embeddings import GeneEmbeddingProvider, PresageGeneEmbedding

# Sentinel so we only try to auto-load the default gene sets once.
_UNSET = object()


def _default_provider() -> GeneEmbeddingProvider:
    """The GenePT embeddings the reported Vendi scores are computed in.

    Raises ``MissingPresageCache`` / ``MissingPresageSource`` if the cache has
    not been downloaded. There is deliberately no fallback: one-hot vectors are
    mutually orthogonal, so they would yield a Vendi score of exactly ``n`` for
    every batch -- a number that is meaningless but indistinguishable, in the
    output, from a real one.
    """
    return PresageGeneEmbedding(source="GenePT_ada")


class BatchDiversity(Metric):
    """Diversity of the genes acquired in the current batch.

    Returns:

    - ``batch_diversity``               : mean pairwise (centered) cosine distance
    - ``batch_diversity_vs_random``     : that, divided by a random genome draw
                                          (~1 = as spread as random, <1 = focused)
    - ``batch_vendi``                   : effective number of distinct genes
    - ``batch_vendi_ratio``             : ``batch_vendi / n`` in (0, 1]
    - ``batch_diversity_n``             : embedded genes used
    - ``batch_pathway_diversity``       : ``1 - mean pairwise Jaccard`` of pathways
    - ``batch_pathway_coverage``        : distinct pathways spanned per gene
    - ``batch_pathway_overlap_vs_random``: pathway overlap / random-draw overlap
                                          (>1 = genes cluster in shared pathways)
    - ``batch_pathway_n``               : annotated genes used

    Keys are omitted only when the batch itself is too small to score (fewer
    than two embedded or annotated genes), never because the reference data is
    missing -- that raises. Pass ``gene_sets=None`` to turn the pathway view
    off on purpose.
    """

    def __init__(
        self,
        provider: GeneEmbeddingProvider | None = None,
        gene_sets: Any = _UNSET,  # ``None`` disables the pathway view

        *,
        baseline_sample: int = 1024,
        baseline_seed: int = 0,
    ):
        self._provider = provider
        # ``_UNSET`` => auto-load default gene sets lazily; ``None`` => disabled.
        self._gene_sets = gene_sets
        self._baseline_sample = baseline_sample
        self._baseline_seed = baseline_seed
        self._emb_baseline: float | None = None

    def name(self) -> str:
        return "batch_diversity"

    # -- providers -----------------------------------------------------------

    def _ensure_provider(self) -> GeneEmbeddingProvider:
        if self._provider is None:
            self._provider = _default_provider()
        return self._provider

    def _ensure_gene_sets(self):
        if self._gene_sets is _UNSET:
            from ..data.gene_sets import load_default_gene_sets

            self._gene_sets = load_default_gene_sets()
        return self._gene_sets

    # -- helpers -------------------------------------------------------------

    def _embedding_baseline(self, provider: GeneEmbeddingProvider, mean) -> float | None:
        """Mean pairwise distance of a random draw from the provider's universe.

        A property of the space, not of any batch, so it is computed once and
        reused. ``None`` when the provider cannot enumerate its universe --
        ``batch_diversity_vs_random`` is then omitted rather than faked.
        """
        if self._emb_baseline is not None:
            return self._emb_baseline
        sample = provider.sample_matrix(self._baseline_sample, seed=self._baseline_seed)
        if sample is None or sample.shape[0] < 2:
            return None
        self._emb_baseline = mean_pairwise_cosine_distance(
            np.asarray(sample, dtype=np.float64), center=mean
        )
        return self._emb_baseline

    # -- metric --------------------------------------------------------------

    def _embedding_metrics(self, genes: list[Any]) -> dict[str, float]:
        provider = self._ensure_provider()
        embs, mask = provider.embed_batch([str(g) for g in genes])
        covered = np.asarray(embs[mask], dtype=np.float64)
        n = int(mask.sum())
        if n < 2:
            return {}
        mean = provider.mean_embedding()
        diversity = mean_pairwise_cosine_distance(covered, center=mean)
        vendi = vendi_score(covered, center=mean)
        out: dict[str, float] = {
            "batch_diversity": diversity,
            "batch_diversity_n": float(n),
            "batch_vendi": vendi,
            "batch_vendi_ratio": vendi / n,
        }
        base = self._embedding_baseline(provider, mean)
        if base:
            out["batch_diversity_vs_random"] = diversity / base
        return out

    def _pathway_metrics(self, genes: list[Any]) -> dict[str, float]:
        gene_sets = self._ensure_gene_sets()
        if gene_sets is None:
            return {}
        return pathway_diversity([str(g) for g in genes], gene_sets)

    def score(
        self,
        observations: list[Observation],
        model_prediction: ModelPrediction | None,
        ground_truth: Any,
        candidates_remaining: list[Any],
        *,
        new_observations: list[Observation] | None = None,
        acquired_batch: list[Any] | None = None,
        n_requested: int | None = None,
    ) -> dict[str, float]:
        if acquired_batch is None or len(acquired_batch) < 2:
            return {}
        out: dict[str, float] = {}
        out.update(self._embedding_metrics(acquired_batch))
        out.update(self._pathway_metrics(acquired_batch))
        return out


__all__ = ["BatchDiversity"]
