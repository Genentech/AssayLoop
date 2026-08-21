"""GLM -> transformer handoff acquisition.

Replays GLM-5.1's recorded per-round acquisitions for the first ``n`` AL rounds,
then delegates to a base (greedy-from-model) acquisition for the remaining rounds.

The inner loop reveals the GLM genes' *true* labels (from the screen) and feeds
them to the amortized-ranker model as context, so the transformer continues
exactly from GLM's warm start. Because everything runs through the standard
:class:`~assaybench.core.SequentialLoop`, the resulting per-step metrics
(``hits_auc``, ``n_hits_vs_random`` with the requested-budget denominator,
batch hits, ...) match GLM's own dashboard runs and ``eval-ranker``.
"""

from __future__ import annotations

from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord


class GlmHandoffAcquisition(AcquisitionFunction):
    """First ``n_warm`` rounds = GLM's recorded picks; the rest = ``base``.

    Args:
        rounds: GLM's per-round acquired gene symbols (round 1 first). Symbols
            not in the live candidate pool are simply dropped by the inner loop
            (counted as shortfall), matching GLM's own under-supplied rounds.
        n_warm: number of leading rounds to take from ``rounds``.
        base: acquisition used once GLM has handed off (typically greedy).
    """

    def __init__(
        self,
        rounds: list[list[str]],
        n_warm: int,
        base: AcquisitionFunction,
    ):
        self.rounds = [list(r) for r in (rounds or [])]
        self.n_warm = max(0, int(n_warm))
        self.base = base
        self._step = 0
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        return f"glm_handoff_n{self.n_warm}+{self.base.name()}"

    def reset(self) -> None:
        self._step = 0
        self._last_trace = {}
        try:
            self.base.reset()
        except NotImplementedError:
            pass

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
        self._step += 1
        # GLM phase: replay the recorded round (intersected with the live pool).
        if self._step <= self.n_warm and self._step <= len(self.rounds):
            cand = set(candidates)
            picked = [g for g in self.rounds[self._step - 1] if g in cand]
            self._last_trace = {
                "source": "glm_warm_start",
                "round": self._step,
                "n_picked": len(picked),
                "n_recorded": len(self.rounds[self._step - 1]),
            }
            return picked[:batch_size]
        # Transformer phase: delegate to the base (greedy) acquisition.
        picked = self.base.suggest(
            history=history,
            candidates=candidates,
            batch_size=batch_size,
            model_prediction=model_prediction,
            task_context=task_context,
        )
        self._last_trace = dict(self.base.last_trace() or {})
        self._last_trace["source"] = "transformer"
        return picked


__all__ = ["GlmHandoffAcquisition"]
