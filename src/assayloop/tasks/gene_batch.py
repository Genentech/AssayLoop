"""Per-screen gene-batch sequential design: the task the paper evaluates.

One ``Task`` = one CRISPR screen. Candidates are gene symbols. At each round
the acquisition picks ``batch_size`` (default 100) genes and ``reveal()``
returns the ground-truth ``hit`` flag and ``relevance_score`` for each.

This module wraps a :class:`~assaybench.ScreenRecord` in a ``Task``. The record
itself, and the loaders that produce one, live in :mod:`assaybench.data.screens`
-- what a benchmark screen *is* is part of the benchmark. Choosing *which*
screens to run on, under this repository's names for them, lives in
:mod:`assayloop.tasks.screen_sets`.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from assaybench.core.task import Task
from assaybench.core.types import Observation
from assaybench.data.screens import ScreenRecord, screen_from_example

log = logging.getLogger("assayloop.tasks.gene_batch")


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class AssayBenchGeneBatchTask(Task):
    """Per-screen gene-batch active learning: one screen, batches of genes.

    Args:
        screen: ScreenRecord to run on.
        warm_start_size: number of genes to "pre-reveal" before the loop
            starts. Default 0 — fully cold-start (the acquisition picks
            from scratch). For trace-collection warm-start variants, set
            this to ``batch_size`` (random first batch) or ``k*batch_size``
            (k random batches). The current implementation samples the
            warm-start genes uniformly at random.
        seed: RNG seed for warm-start sampling and tie-breaking.

    Extension point (not yet wired): a "gene-program" warm start that
    seeds the first batch from a named pathway/program rather than random
    genes. This would take an optional ``warm_start_genes: list[str]`` (or
    a pathway name resolved via :mod:`assayloop.data.gene_sets`
    ``GeneSetMembership``) and use it in place of the random sample below.
    """

    def __init__(
        self,
        screen: ScreenRecord,
        warm_start_size: int = 0,
        seed: int = 0,
        universe_genes: list[str] | None = None,
    ):
        self.screen = screen
        self.seed = seed
        self.warm_start_size = int(warm_start_size or 0)
        self._universe_genes = universe_genes
        self._rng = random.Random(seed)

        # Warn (loudly, once per screen) if the library has duplicate gene
        # symbols, because gene_to_index keeps only the last occurrence —
        # which would mean ``reveal()`` assigns all duplicates the same
        # label/score and silently mis-counts hits.
        n_unique = len(set(screen.genes))
        if n_unique != len(screen.genes):
            log.warning(
                "[%s] duplicate gene symbols detected: %d unique of %d total; "
                "reveal() will collapse duplicates to the last library index.",
                screen.dataset_name,
                n_unique,
                len(screen.genes),
            )

        self._gene_to_idx = screen.gene_to_index
        # Acquired state — gene symbol -> Observation.
        self._acquired: dict[str, Observation] = {}
        # Observations revealed in __init__ via warm-start; surfaced to the
        # inner loop via initial_observations() so model/acquisition/metrics
        # can see them. Insertion order matches the warm sample order.
        self._initial_observations: list[Observation] = []

        if self.warm_start_size > 0:
            warm = self._rng.sample(self.screen.genes, k=min(self.warm_start_size, len(self.screen.genes)))
            self._initial_observations = list(self.reveal(warm))

    # ---- Task interface ----

    def candidates(self) -> list[str]:
        pool = self._universe_genes if self._universe_genes is not None else self.screen.genes
        return [g for g in pool if g not in self._acquired]

    def reveal(self, batch: list[str]) -> list[Observation]:
        observations: list[Observation] = []
        for g in batch:
            if g in self._acquired:
                # Already acquired; skip (the runner clean-up should
                # prevent this but be defensive).
                continue
            idx = self._gene_to_idx.get(g)
            if idx is None:
                # Gene not in the library; record an Observation with
                # ``hit=False, score=nan`` and metadata flagging it.
                obs = Observation(
                    candidate=g,
                    label={"hit": False, "relevance_score": float("nan")},
                    metadata={"in_library": False},
                )
            else:
                hit = self.screen.hits[idx]
                score = self.screen.relevance_scores[idx]
                obs = Observation(
                    candidate=g,
                    label={"hit": bool(hit), "relevance_score": float(score)},
                    metadata={"in_library": True, "library_index": int(idx)},
                )
            self._acquired[g] = obs
            observations.append(obs)
        return observations

    def ground_truth(self) -> ScreenRecord:
        return self.screen

    def context(self) -> dict[str, Any]:
        return self.screen.context()

    def task_id(self) -> str:
        return f"option2/{self.screen.dataset_name}"

    def total_positives(self) -> int:
        return self.screen.total_hits

    def initial_observations(self) -> list[Observation]:
        return list(self._initial_observations)

    def reset(self) -> None:
        self._acquired = {}
        self._initial_observations = []
        self._rng = random.Random(self.seed)
        if self.warm_start_size > 0:
            warm = self._rng.sample(
                self.screen.genes,
                k=min(self.warm_start_size, len(self.screen.genes)),
            )
            self._initial_observations = list(self.reveal(warm))


def make_task(
    screen: ScreenRecord,
    *,
    warm_start_size: int = 0,
    seed: int = 0,
    universe_genes: list[str] | None = None,
) -> AssayBenchGeneBatchTask:
    return AssayBenchGeneBatchTask(
        screen=screen, warm_start_size=warm_start_size, seed=seed,
        universe_genes=universe_genes,
    )


__all__ = [
    "ScreenRecord",
    "screen_from_example",
    "AssayBenchGeneBatchTask",
    "make_task",
]
