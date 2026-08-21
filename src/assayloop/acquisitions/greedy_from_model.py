"""Greedy top-K acquisition: pick the highest-scoring candidates."""

from __future__ import annotations

import random
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord


class GreedyFromModel(AcquisitionFunction):
    """Take the top ``batch_size`` candidates by ``model_prediction.scores``.

    Falls back to random when no model prediction is provided (e.g. paired
    with NullModel — in that case use RandomAcquisition instead).

    Args:
        seed: RNG seed for tie-breaking and the no-prediction fallback.
        epsilon: with probability ``epsilon`` per step, sample
            uniformly at random instead of greedily (an ε-greedy
            schedule helps escape early-step traps).
    """

    def __init__(self, *, seed: int = 0, epsilon: float = 0.0):
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        return "greedy" if self.epsilon == 0.0 else f"epsilon_greedy_eps{self.epsilon}"

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

        if self.epsilon > 0:
            n_explore = sum(1 for _ in range(k) if self._rng.random() < self.epsilon)
        else:
            n_explore = 0

        # Greedy by score with random tie-break, over the candidates the model
        # actually scored. Models with partial coverage (LLMNN scores only genes
        # that have an embedding) must not have their gaps filled by chance.
        # No-op for models that score every candidate.
        scored = [(c, scores[c]) for c in candidates if c in scores]
        keyed = [(-s, self._rng.random(), c) for c, s in scored]
        keyed.sort()
        greedy = [c for _, _, c in keyed[: max(0, k - n_explore)]]

        if n_explore > 0:
            scored_set = {c for c, _ in scored}
            remaining = [c for c in candidates if c in scored_set and c not in set(greedy)]
            self._rng.shuffle(remaining)
            greedy.extend(remaining[:n_explore])

        self._last_trace = {
            "top_scores": [
                {"gene": c, "score": float(scores.get(c, 0.0))}
                for c in greedy[: min(10, len(greedy))]
            ],
            "n_explore": n_explore,
            "n_greedy": k - n_explore,
        }
        return greedy


__all__ = ["GreedyFromModel"]
