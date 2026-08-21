"""Per-screen gene-batch sequential design: the task the paper evaluates.

One ``Task`` = one CRISPR screen. Candidates are gene symbols. At each round
the acquisition picks ``batch_size`` (default 100) genes and ``reveal()``
returns the ground-truth ``hit`` flag and ``relevance_score`` for each.

This module is deliberately screen-set agnostic: it turns an AssayBench row
into a :class:`ScreenRecord` and wraps a ``ScreenRecord`` in a ``Task``.
Choosing *which* screens to run on lives in :mod:`assayloop.tasks.screen_sets`.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from assaybench.core.task import Task
from assaybench.core.types import Observation

log = logging.getLogger("assayloop.tasks.gene_batch")


# ---------------------------------------------------------------------------
# ScreenRecord — one AssayBench row, normalised into the shape the loop wants.
# ---------------------------------------------------------------------------


@dataclass
class ScreenRecord:
    dataset_name: str
    split: str
    organism: str
    gene_symbol_convention: str  # e.g. "HGNC", "MGI"
    phenotype: str
    cell_line: str
    cell_type: str
    library_type: str
    library_methodology: str
    direction_str: str
    condition_clause: str
    num_genes: int
    genes: list[str]
    relevance_scores: list[float]
    hits: list[bool]
    question: str = ""             # pre-rendered ranking prompt
    description: str = ""
    contrast_label: str = ""
    reverse: bool = False
    cleaned_phenotype: str = ""
    # Continuous relevance score for EVERY gene (not just hits): non-hits,
    # which carry a masked ``relevance_score`` of 0, are filled with their
    # ``combined_scores`` value. This is the MSE regression target for the
    # amortized ranker. Empty when the source row lacked ``combined_scores``
    # (falls back to ``relevance_scores`` in that case — see
    # ``screen_from_example``).
    unmasked_relevance_scores: list[float] = field(default_factory=list)

    @property
    def total_hits(self) -> int:
        return int(sum(1 for h in self.hits if h))

    @property
    def gene_to_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genes)}

    def context(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "organism": self.organism,
            "gene_symbol_convention": self.gene_symbol_convention,
            "phenotype": self.phenotype,
            "cell_line": self.cell_line,
            "cell_type": self.cell_type,
            "library_type": self.library_type,
            "library_methodology": self.library_methodology,
            "direction_str": self.direction_str,
            "condition_clause": self.condition_clause,
            "num_genes": self.num_genes,
            "total_hits": self.total_hits,
            "description": self.description,
            "contrast_label": self.contrast_label,
            "question": self.question,
            "cleaned_phenotype": self.cleaned_phenotype,
        }


def screen_from_example(example: dict[str, Any]) -> ScreenRecord:
    """Normalise one AssayBench example dict into a ScreenRecord."""
    genes = list(example.get("relevance_genes") or [])
    rs = list(example.get("relevance_scores") or [])
    hits = list(example.get("hit") or [])
    # Pad/truncate parallel arrays defensively (some legacy rows have
    # shorter hit arrays than relevance arrays).
    n = len(genes)
    rs = rs + [0.0] * (n - len(rs)) if len(rs) < n else rs[:n]
    hits = hits + [False] * (n - len(hits)) if len(hits) < n else hits[:n]
    hits = [bool(h) for h in hits]

    org = example.get("organism") or "Homo sapiens"
    conv = example.get("gene_symbol_convention") or (
        "HGNC" if "sapiens" in str(org).lower() or not org else "MGI"
    )

    # Unmasked relevance scores: fill the masked (==0) non-hit entries with
    # the gene's ``combined_scores`` value so every gene has a continuous
    # target (mirrors assaybench dataset.py). Falls back to the masked
    # ``relevance_scores`` when the source row has no ``combined_scores``.
    combined = example.get("combined_scores")
    if combined is not None:
        combined = list(combined)
        combined = (
            combined + [0.0] * (n - len(combined))
            if len(combined) < n else combined[:n]
        )
        unmasked = [
            float(combined[i]) if float(rs[i]) == 0.0 else float(rs[i])
            for i in range(n)
        ]
    else:
        unmasked = [float(x) for x in rs]

    return ScreenRecord(
        dataset_name=example.get("dataset_name") or example.get("screen_name", "unknown"),
        split=example.get("split", ""),
        organism=org,
        gene_symbol_convention=conv,
        phenotype=example.get("phenotype", "") or "",
        cell_line=example.get("cell_line", "") or "",
        cell_type=example.get("cell_type", "") or "",
        library_type=example.get("library_type", "") or "",
        library_methodology=example.get("library_methodology", "") or "",
        direction_str=example.get("direction_str", "") or "",
        condition_clause=example.get("condition_clause", "") or "",
        num_genes=int(example.get("num_genes", n) or n),
        genes=genes,
        relevance_scores=[float(x) for x in rs],
        hits=hits,
        unmasked_relevance_scores=unmasked,
        question=example.get("question", "") or "",
        description=example.get("description", "") or "",
        contrast_label=example.get("contrast_label", "") or "",
        reverse=bool(example.get("reverse", False)),
        cleaned_phenotype=example.get("cleaned_phenotype", "") or "",
    )


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
