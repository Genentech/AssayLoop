"""Random baseline acquisition."""

from __future__ import annotations

import random
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord


class RandomAcquisition(AcquisitionFunction):
    """Uniformly sample ``batch_size`` candidates without replacement."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "random"

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def suggest(
        self,
        history: list[StepRecord],
        candidates: list[Any],
        batch_size: int,
        model_prediction: ModelPrediction | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        k = min(batch_size, len(candidates))
        return self._rng.sample(candidates, k)


__all__ = ["RandomAcquisition"]
