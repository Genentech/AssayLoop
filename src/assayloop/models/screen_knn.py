"""Nearest-screen model.

Ranks candidate genes by finding training screens whose revealed hit
pattern is closest to the current screen, then predicting unrevealed
genes via a Bayesian mix of a global prior and distance-weighted
screen evidence.
"""

from __future__ import annotations

import math
from typing import Any

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation


class ScreenKNNModel(Model):

    def __init__(
        self,
        *,
        training_set: str = "public_train",
        tau: float = 0.1,
        kappa: float = 100.0,
        filter_phenotype: bool = False,
        prior_only: bool = False,
    ):
        self._training_set = training_set
        self._tau = float(tau)
        self._kappa = float(kappa)
        self._filter_phenotype = bool(filter_phenotype)
        self._prior_only = bool(prior_only)

        self._screens: list | None = None
        self._prior: dict[str, float] = {}
        self._hit_lookup: list[dict[str, bool]] = []

    def _ensure_screens(self) -> None:
        if self._screens is not None:
            return
        from ..tasks import load_screens

        self._screens = load_screens(target_set=self._training_set)
        self._hit_lookup = []
        gene_hit_count: dict[str, int] = {}
        gene_total_count: dict[str, int] = {}
        for screen in self._screens:
            hit_map: dict[str, bool] = {}
            for gene, hit in zip(screen.genes, screen.hits):
                hit_map[gene] = hit
                gene_total_count[gene] = gene_total_count.get(gene, 0) + 1
                if hit:
                    gene_hit_count[gene] = gene_hit_count.get(gene, 0) + 1
            self._hit_lookup.append(hit_map)

        self._prior = {
            gene: gene_hit_count.get(gene, 0) / count
            for gene, count in gene_total_count.items()
        }

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return "screen_knn"

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        self._ensure_screens()
        assert self._screens is not None

        revealed: dict[str, float] = {}
        for obs in observations:
            label = obs.label
            if isinstance(label, dict):
                revealed[obs.candidate] = 1.0 if label.get("hit") else 0.0
            else:
                revealed[obs.candidate] = 1.0 if label else 0.0

        current_name = (task_context or {}).get("dataset_name", "")
        current_phenotype = (task_context or {}).get("cleaned_phenotype", "")

        n_revealed = len(revealed)
        if self._prior_only:
            lam = 0.0
        else:
            lam = n_revealed / (n_revealed + self._kappa) if n_revealed > 0 else 0.0

        weights: list[float] = []
        active_indices: list[int] = []
        for idx, screen in enumerate(self._screens):
            if screen.dataset_name == current_name:
                continue
            if self._filter_phenotype and current_phenotype and screen.cleaned_phenotype != current_phenotype:
                continue
            hit_map = self._hit_lookup[idx]
            overlap = [g for g in revealed if g in hit_map]
            if not overlap:
                weights.append(math.exp(-1.0 / self._tau))
                active_indices.append(idx)
                continue
            dist = sum((revealed[g] - (1.0 if hit_map[g] else 0.0)) ** 2 for g in overlap) / len(overlap)
            weights.append(math.exp(-dist / self._tau))
            active_indices.append(idx)

        scores: dict[Any, float] = {}
        for gene in candidates:
            p0 = self._prior.get(gene, 0.0)
            if not active_indices:
                scores[gene] = p0
                continue
            w_sum = 0.0
            w_hit = 0.0
            for i, idx in enumerate(active_indices):
                hit_map = self._hit_lookup[idx]
                if gene in hit_map:
                    w_sum += weights[i]
                    if hit_map[gene]:
                        w_hit += weights[i]
            if w_sum > 0:
                screen_evidence = w_hit / w_sum
                scores[gene] = (1.0 - lam) * p0 + lam * screen_evidence
            else:
                scores[gene] = p0

        return ModelPrediction(
            scores=scores,
            metadata={
                "name": self.name(),
                "n_training_screens": len(active_indices),
                "n_revealed": n_revealed,
                "lambda": lam,
            },
        )


__all__ = ["ScreenKNNModel"]
