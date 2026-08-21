"""Option 1 acquisition: pick the next training screen to add to context.

Two concrete acquisitions are provided:

- :class:`RandomScreenAcquisition` -- baseline: uniform sampling.
- :class:`LLMScreenAcquisition`    -- LLM picks the most informative
  candidate screen via a single prompt.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord
from ...llm.client import LLMClientConfig, complete
from ...tasks import ScreenRecord


def _parse_screen_id(text: str) -> str | None:
    # Prefer a JSON object with "screen_id".
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "screen_id" in obj:
                return str(obj["screen_id"]).strip()
        except json.JSONDecodeError:
            pass
    # Fallback: first non-empty line.
    for line in text.splitlines():
        line = line.strip().strip(",.;:")
        if line:
            return line
    return None


class RandomScreenAcquisition(AcquisitionFunction):
    """Uniform sampling of candidate screens."""

    def __init__(self, *, seed: int = 0):
        self.seed = seed
        self._rng = random.Random(seed)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        return "random_screen"

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
        if not candidates:
            return []
        k = min(batch_size, len(candidates))
        picked = self._rng.sample(candidates, k)
        self._last_trace = {"strategy": "random", "n": k}
        return picked


class LLMScreenAcquisition(AcquisitionFunction):
    """LLM picks the next training screen to add to context."""

    def __init__(
        self,
        *,
        llm: LLMClientConfig | None = None,
        candidate_summary_chars: int = 200,
    ):
        self.llm = llm or LLMClientConfig()
        self.candidate_summary_chars = int(candidate_summary_chars)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        cfg = self.llm.resolved()
        return f"llm_screen_acq/{cfg.provider}:{cfg.model}"

    def reset(self) -> None:
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
        if not candidates:
            return []
        if batch_size != 1:
            # The LLM is asked for a single best candidate per call;
            # for larger batches we iterate (cheap because LLMScreen
            # batch_size is normally 1).
            picked: list[Any] = []
            already: set[str] = set()
            remaining = list(candidates)
            for _ in range(min(batch_size, len(candidates))):
                one = self._pick_one(history + [], remaining, model_prediction, task_context, exclude=already)
                if one is None:
                    break
                picked.append(one)
                already.add(one.dataset_name)
                remaining = [c for c in remaining if c.dataset_name not in already]
            return picked

        one = self._pick_one(history, list(candidates), model_prediction, task_context, exclude=set())
        return [one] if one is not None else []

    def _pick_one(
        self,
        history: list[StepRecord],
        candidates: list[ScreenRecord],
        model_prediction: ModelPrediction | None,
        task_context: dict[str, Any] | None,
        exclude: set[str],
    ) -> ScreenRecord | None:
        candidates = [c for c in candidates if c.dataset_name not in exclude]
        if not candidates:
            return None

        ctx = task_context or {}
        target_block = ""
        target = ctx.get("target") or {}
        if target:
            target_block = (
                f"  phenotype: {target.get('phenotype')}\n"
                f"  cell_line: {target.get('cell_line')}\n"
                f"  organism: {target.get('organism')}\n"
                f"  library_methodology: {target.get('library_methodology')}\n"
                f"  condition: {target.get('condition_clause')}\n"
            )

        all_observations = []
        for r in history:
            all_observations.extend(r.new_observations)
        if all_observations:
            sel_lines = [
                f"  - [{o.candidate.dataset_name}] {o.candidate.phenotype}"
                for o in all_observations if isinstance(o.candidate, ScreenRecord)
            ]
            selected_block = "\n".join(sel_lines)
        else:
            selected_block = "  (none selected yet)"

        cand_lines = [
            f"  - [{c.dataset_name}] phenotype={c.phenotype!s:60.60s}  "
            f"cell_line={c.cell_line!s:40.40s}  organism={c.organism}"
            for c in candidates[:200]  # keep prompt bounded
        ]
        cand_block = "\n".join(cand_lines)

        user = (
            f"TARGET SCREEN:\n{target_block}\n"
            f"SCREENS ALREADY SELECTED:\n{selected_block}\n\n"
            f"CANDIDATE SCREENS ({len(cand_lines)} of {len(candidates)} shown):\n"
            f"{cand_block}\n\n"
            f"Select the single MOST informative candidate screen to add. "
            f"Respond with JSON only: "
            f'{{"screen_id": "<dataset_name>", "rationale": "<one sentence>"}}'
        )
        system = (
            "You are picking training screens to add to a few-shot context "
            "that helps predict CRISPR screen hit genes for the target. "
            "Pick the candidate most biologically related to the target."
        )

        try:
            text = complete(self.llm, prompt=user, system=system)
        except Exception as e:  # noqa: BLE001
            self._last_trace = {"error": f"{type(e).__name__}: {e}", "fallback": "first"}
            return candidates[0]

        sid = _parse_screen_id(text)
        for c in candidates:
            if c.dataset_name == sid:
                self._last_trace = {
                    "picked": c.dataset_name,
                    "raw_response_preview": text[:300],
                }
                return c
        self._last_trace = {
            "fallback": "first",
            "raw_response_preview": text[:300],
        }
        return candidates[0]


__all__ = ["RandomScreenAcquisition", "LLMScreenAcquisition"]
