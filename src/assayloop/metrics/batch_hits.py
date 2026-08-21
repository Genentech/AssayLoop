"""Batch-hit metrics: counts and rates computed on the current suggested batch."""

from __future__ import annotations

from typing import Any

from assaybench.core.metric import Metric
from assaybench.core.types import ModelPrediction, Observation


def _hit_flag(obs: Observation) -> int:
    label = obs.label
    if isinstance(label, dict):
        return int(bool(label.get("hit", False)))
    if isinstance(label, (bool, int)):
        return int(bool(label))
    return 0


def random_expected_batch_hits(
    remaining_hits: float, batch_size: float, remaining_candidates: float
) -> float:
    """Hits a uniformly-random pick of ``batch_size`` from ``remaining_candidates``
    (containing ``remaining_hits`` hits) finds in expectation."""
    if remaining_candidates <= 0:
        return 0.0
    return float(remaining_hits) * float(batch_size) / float(remaining_candidates)


def batch_random_adjusted_hit_rate(
    n_hits: float, remaining_hits: float, batch_size: float, remaining_candidates: float
) -> float:
    """Per-step enrichment: realized hits / random-expected hits for the batch.

    ``1.0`` = no better than random, ``> 1`` = enriched. Shared by the
    :class:`BatchHits` metric and the RL trainer's per-step reward so both use an
    identical definition.
    """
    exp = random_expected_batch_hits(remaining_hits, batch_size, remaining_candidates)
    return float(n_hits / exp) if exp > 0 else 0.0


class BatchHits(Metric):
    """Metrics computed on the current acquisition batch only.

    Returns:
        batch_n_hits       : number of hits in the batch
        batch_size_actual  : number of candidates in the batch
        batch_hit_rate     : batch_n_hits / batch_size_actual
    """

    def name(self) -> str:
        return "batch_hits"

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
        
        if new_observations is None:
            return {}
        n = len(new_observations)
        n_hits = sum(_hit_flag(o) for o in new_observations)

        #random baseline computation
        n_hits_total = sum(ground_truth.hits)
        n_hits_history = sum(_hit_flag(o) for o in observations)
        n_hits_previous = n_hits_history - n_hits
        len_candidates_remaining = len(candidates_remaining) + len(acquired_batch)
        n_hits_potential = n_hits_total - n_hits_previous
        random_expected = random_expected_batch_hits(
            n_hits_potential, n, len_candidates_remaining
        )

        return {
            "batch_n_hits": float(n_hits),
            "batch_size_actual": float(n),
            "batch_hit_rate": float(n_hits / n) if n > 0 else 0.0,
            "batch_random_expected_n_hits": float(random_expected),
            "batch_random_expected_hit_rate": float(random_expected / n) if n > 0 else 0.0,
            "batch_random_adjusted_hit_rate": batch_random_adjusted_hit_rate(
                n_hits, n_hits_potential, n, len_candidates_remaining
            ),
        }


__all__ = [
    "BatchHits",
    "random_expected_batch_hits",
    "batch_random_adjusted_hit_rate",
]
