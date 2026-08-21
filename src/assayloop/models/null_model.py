"""NullModel — uniform scores; for use with random / LLM acquisitions."""

from __future__ import annotations

from typing import Any

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation


class NullModel(Model):
    """Returns a uniform score of 0 for every candidate.

    Pair with `RandomAcquisition` or `LLMSingleAcquisition` when you
    don't want an internal classical model in the loop.
    """

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        return ModelPrediction(
            scores={c: 0.0 for c in candidates},
            uncertainty={c: 1.0 for c in candidates},  # uniform high uncertainty
            metadata={"name": "null"},
        )

    def name(self) -> str:
        return "null"


__all__ = ["NullModel"]
