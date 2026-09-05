"""Option 1 task: pick training screens to improve hit prediction on a
target screen.

One inner-loop run = one target internal screen + a candidate pool of
other internal screens. Acquisition picks ``batch_size`` (default 1)
training screens per step; ``reveal()`` returns the full
:class:`ScreenRecord` of each picked screen as an
:class:`Observation`.

The eval-target is iterated externally by
:func:`assayloop.examples.assaybench.run.run_option1`, which spins up
one such task per target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from assaybench.core.task import Task
from assaybench.core.types import Observation
from ...tasks import ScreenRecord, load_screens


@dataclass
class Option1Context:
    target: ScreenRecord
    pool: list[ScreenRecord]


class AssayBenchScreenSelectionTask(Task):
    """Per-target screen-selection task (Option 1).

    Candidates are ``ScreenRecord`` objects (other internal screens);
    the acquired set is the running few-shot pool for the model.
    """

    def __init__(
        self,
        target: ScreenRecord,
        pool: list[ScreenRecord],
        *,
        seed: int = 0,
    ):
        self.target = target
        self.pool: list[ScreenRecord] = [p for p in pool if p.dataset_name != target.dataset_name]
        self._acquired: dict[str, Observation] = {}
        self.seed = seed
        self._rng = random.Random(seed)

    def candidates(self) -> list[ScreenRecord]:
        return [s for s in self.pool if s.dataset_name not in self._acquired]

    def reveal(self, batch: list[ScreenRecord]) -> list[Observation]:
        observations: list[Observation] = []
        for s in batch:
            if not isinstance(s, ScreenRecord):
                continue
            if s.dataset_name in self._acquired:
                continue
            obs = Observation(
                candidate=s,
                label={
                    "dataset_name": s.dataset_name,
                    "phenotype": s.phenotype,
                    "n_hits": s.total_hits,
                },
                metadata={"organism": s.organism},
            )
            self._acquired[s.dataset_name] = obs
            observations.append(obs)
        return observations

    def ground_truth(self) -> ScreenRecord:
        return self.target

    def context(self) -> dict[str, Any]:
        return {
            "task": "option1_screen_selection",
            "target": self.target.context(),
            "pool_size": len(self.pool),
        }

    def task_id(self) -> str:
        return f"option1/{self.target.dataset_name}"

    def total_positives(self) -> int:
        return self.target.total_hits

    def reset(self) -> None:
        self._acquired = {}
        self._rng = random.Random(self.seed)


def build_pool(
    *,
    target_set: str = "paper_test",
    dataset_names: list[str] | None = None,
) -> list[ScreenRecord]:
    """Resolve the candidate pool of screens.

    Every screen in ``target_set`` is available as both a target and a
    candidate (the task filters out the target). Pass ``dataset_names`` for an
    explicit list.
    """
    return load_screens(dataset_names=dataset_names, target_set=target_set)


__all__ = [
    "Option1Context",
    "AssayBenchScreenSelectionTask",
    "build_pool",
]
