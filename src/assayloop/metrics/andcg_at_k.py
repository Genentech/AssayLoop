"""Secondary metric: adjusted-nDCG@K of the internal model's prediction.

Wraps ``assaybench.benchmark.metrics.RankingMetrics`` (public API).
Measures how well the model ranks the UNACQUIRED candidates against the
ground-truth relevance scores. Tracks the model's improvement as more
data is acquired.
"""

from __future__ import annotations

from typing import Any

from assaybench.core.metric import Metric
from assaybench.core.types import ModelPrediction, Observation


class AnDCGAtK(Metric):
    """Adjusted-nDCG@K over the model's current prediction on remaining
    candidates.

    Args:
        k_values: list of K cutoffs (default ``[10, 50, 100]``).
        organism: passes through to ``RankingMetrics.evaluate`` so the
            mouse symbol table is used for mouse screens. When ``None``
            (default), bridges this from ``ground_truth.organism`` if
            available; falls back to "Homo sapiens".
    """

    def __init__(self, k_values: list[int] | None = None, organism: str | None = None):
        self.k_values = k_values or [10, 50, 100]
        self.organism = organism
        self._ranking_metrics = None  # lazy

    def _get_rm(self):
        if self._ranking_metrics is None:
            from assaybench.benchmark.metrics import RankingMetrics  # public API

            self._ranking_metrics = RankingMetrics(
                k_values=self.k_values,
                metric_groups=["adjusted_ndcg", "ndcg"],
                use_gene_mapper=False,  # we already pass canonical symbols
            )
        return self._ranking_metrics

    def name(self) -> str:
        return "andcg_at_k"

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
        out: dict[str, float] = {}

        if model_prediction is None or not candidates_remaining:
            return out

        # Build relevance-score table from ground truth, restricted to
        # remaining candidates (acquired genes are no longer scored —
        # the model is judged on what it still has to find).
        rel_scores: list[float] = []
        gt_genes: list[str] = []
        # Lookup of gene -> relevance from ScreenRecord
        gene_to_rel: dict[str, float] = {}
        genes = getattr(ground_truth, "genes", None)
        rels = getattr(ground_truth, "relevance_scores", None)
        if genes and rels:
            gene_to_rel = dict(zip(genes, rels))
        elif isinstance(ground_truth, dict):
            gene_to_rel = dict(
                zip(
                    ground_truth.get("relevance_genes") or [],
                    ground_truth.get("relevance_scores") or [],
                )
            )

        for c in candidates_remaining:
            if c in gene_to_rel:
                gt_genes.append(c)
                rel_scores.append(float(gene_to_rel[c]))

        if not gt_genes:
            return out

        # Predicted ranking on the remaining candidates: sort by score desc.
        scores = model_prediction.scores or {}
        predicted = sorted(
            candidates_remaining,
            key=lambda g: -float(scores.get(g, float("-inf"))),
        )

        organism = self.organism or getattr(ground_truth, "organism", None) or "Homo sapiens"

        try:
            results = self._get_rm().evaluate(
                predicted_genes=predicted,
                ground_truth_genes=gt_genes,
                relevance_scores=rel_scores,
                organism=organism,
            )
        except Exception:  # noqa: BLE001
            # Surface that we tried but failed so persisted results don't
            # silently look "successful with missing nDCG keys".
            out["andcg_at_k__error"] = 1.0
            return out

        for k in self.k_values:
            key = f"adjusted_ndcg@{k}"
            if key in results:
                out[key] = float(results[key])
            key2 = f"ndcg@{k}"
            if key2 in results:
                out[key2] = float(results[key2])
        return out


__all__ = ["AnDCGAtK"]
