"""Option 1 metric: AnDCG@K on the model's predicted ranking.

Wraps ``assaybench.benchmark.metrics.RankingMetrics`` (public API) to
report adjusted-nDCG@K against the target screen's ground-truth
relevance scores.
"""

from __future__ import annotations

from typing import Any

from assaybench.core.metric import Metric
from assaybench.core.types import ModelPrediction, Observation
from ...tasks import ScreenRecord


class Option1AnDCG(Metric):
    """Adjusted-nDCG@K of the LLM ranker against the target screen."""

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [10, 50, 100]
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from assaybench.benchmark.metrics import RankingMetrics  # public API
            self._ranker = RankingMetrics(
                k_values=self.k_values,
                metric_groups=["adjusted_ndcg", "ndcg"],
                use_gene_mapper=False,
            )
        return self._ranker

    def name(self) -> str:
        return "option1_andcg_at_k"

    def score(
        self,
        observations: list[Observation],
        model_prediction: ModelPrediction | None,
        ground_truth: Any,
        candidates_remaining: list[Any],
        *,
        new_observations: list[Observation] | None = None,
        acquired_batch: list[Any] | None = None,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        if model_prediction is None:
            return out
        target: ScreenRecord = ground_truth
        if not isinstance(target, ScreenRecord):
            return out

        ranked = (model_prediction.metadata or {}).get("ranked_genes")
        if not ranked:
            scores = model_prediction.scores or {}
            ranked = [g for g, _ in sorted(scores.items(), key=lambda x: -x[1])]
        if not ranked:
            return out

        try:
            res = self._get_ranker().evaluate(
                predicted_genes=ranked,
                ground_truth_genes=list(target.genes),
                relevance_scores=[float(x) for x in target.relevance_scores],
                organism=target.organism or "Homo sapiens",
            )
        except Exception:
            return out

        for k in self.k_values:
            for key in (f"adjusted_ndcg@{k}", f"ndcg@{k}"):
                if key in res:
                    out[key] = float(res[key])
        # Count hits in top-K.
        hit_set = {g for g, h in zip(target.genes, target.hits) if h}
        for k in self.k_values:
            out[f"hits_in_top_{k}"] = float(sum(1 for g in ranked[:k] if g in hit_set))
        return out


__all__ = ["Option1AnDCG"]
