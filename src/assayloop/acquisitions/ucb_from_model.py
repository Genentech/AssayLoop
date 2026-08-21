"""UCB-style acquisition: score + beta * sqrt(uncertainty)."""

from __future__ import annotations

import math
import random
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord


class UCBFromModel(AcquisitionFunction):
    """Upper-confidence-bound acquisition.

    score(c) = model.scores[c] + beta * sqrt(model.uncertainty[c])

    Falls back to greedy-from-scores if uncertainty isn't provided, and
    to random if no prediction at all.

    Args:
        beta: exploration coefficient.
        seed: RNG seed.
    """

    def __init__(self, *, beta: float = 1.0, seed: int = 0):
        self.beta = float(beta)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        return f"ucb_beta{self.beta}"

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_trace = {}

    def last_trace(self) -> dict[str, Any]:
        return self._last_trace

    def suggest(
        self,
        history: list[StepRecord],
        candidates: list[Any],
        batch_size: int,
        model_prediction: ModelPrediction | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        k = min(batch_size, len(candidates))
        if not candidates:
            return []
        if model_prediction is None or not model_prediction.scores:
            self._last_trace = {"reason": "no_prediction", "fallback": "random"}
            return self._rng.sample(candidates, k)

        scores = model_prediction.scores
        unc = model_prediction.uncertainty or {}

        def ucb(c: Any) -> float:
            s = float(scores.get(c, 0.0))
            u = float(unc.get(c, 0.0))
            return s + self.beta * math.sqrt(max(u, 0.0))

        keyed = [(-ucb(c), self._rng.random(), c) for c in candidates]
        keyed.sort()
        picked = [c for _, _, c in keyed[:k]]

        self._last_trace = {
            "beta": self.beta,
            "top": [
                {
                    "gene": c,
                    "score": float(scores.get(c, 0.0)),
                    "unc": float(unc.get(c, 0.0)),
                    "ucb": float(ucb(c)),
                }
                for c in picked[: min(10, len(picked))]
            ],
        }
        return picked


__all__ = ["UCBFromModel"]
