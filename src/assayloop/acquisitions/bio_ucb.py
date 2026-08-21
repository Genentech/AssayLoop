"""BioUCB: UCB augmented with enrichment-analysis prior (BioBO, Li et al. 2026).

Wraps the standard UCB acquisition with a piBO prior (Hvarfner et al. 2022)
derived from pathway enrichment analysis on the current top-performing genes.
The prior decays as more data is collected (exponent beta_pibo / L_n), so the
acquisition converges to plain UCB asymptotically.
"""

from __future__ import annotations

import math
import random
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, Observation, StepRecord
from ..data.enrichment import enrichment_prior, load_pathway_db


class BioUCBFromModel(AcquisitionFunction):
    """UCB + piBO enrichment-analysis prior.

    Args:
        beta: UCB exploration coefficient (score + beta * sqrt(unc)).
        beta_pibo: piBO prior confidence (higher = trust EA more initially).
        pathway_source: MSigDB collection name (default ``h.all`` = Hallmark).
        temperature: EA prior temperature (lower = sharper prior).
        top_k_frac: fraction of labeled genes treated as "top" for EA.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        beta_pibo: float = 1.0,
        pathway_source: str = "h.all",
        temperature: float = 0.1,
        top_k_frac: float = 0.1,
        seed: int = 0,
    ):
        self.beta = float(beta)
        self.beta_pibo = float(beta_pibo)
        self.pathway_source = pathway_source
        self.temperature = float(temperature)
        self.top_k_frac = float(top_k_frac)
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._pathway_db = load_pathway_db(pathway_source)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        return f"bio_ucb_beta{self.beta}_pibo{self.beta_pibo}"

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_trace = {}

    def last_trace(self) -> dict[str, Any]:
        return self._last_trace

    @staticmethod
    def _label_value(obs: Observation) -> float:
        label = obs.label
        if isinstance(label, dict):
            if label.get("hit"):
                return 1.0
            return float(label.get("relevance_score", 0.0))
        return float(label) if label is not None else 0.0

    def _collect_observations(self, history: list[StepRecord]) -> list[Observation]:
        obs: list[Observation] = []
        for step in history:
            obs.extend(step.new_observations)
        return obs

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

        # Collect all observations and identify top genes for EA
        all_obs = self._collect_observations(history)
        L_n = len(all_obs)

        top_genes: list[str] = []
        n_sig_pathways = 0
        prior: dict[str, float] = {}

        if L_n > 0:
            sorted_obs = sorted(all_obs, key=self._label_value, reverse=True)
            n_top = max(1, int(len(sorted_obs) * self.top_k_frac))
            top_genes = [str(o.candidate) for o in sorted_obs[:n_top]]

            background_size = L_n + len(candidates)
            prior, ea_results = enrichment_prior(
                top_genes=top_genes,
                unlabeled_genes=[str(c) for c in candidates],
                pathway_db=self._pathway_db,
                background_size=background_size,
                temperature=self.temperature,
            )
            n_sig_pathways = len(ea_results)

        # piBO augmentation (Eq. 4, log-space): ucb(c) + (beta_pibo / L_n) * log(pi(c))
        exponent = self.beta_pibo / max(L_n, 1)

        def pi_ucb(c: Any) -> float:
            base = ucb(c)
            p = prior.get(str(c), 0.0)
            if p > 0:
                return base + exponent * math.log(p)
            return base + exponent * math.log(1e-30)

        keyed = [(-pi_ucb(c), self._rng.random(), c) for c in candidates]
        keyed.sort()
        picked = [c for _, _, c in keyed[:k]]

        self._last_trace = {
            "beta": self.beta,
            "beta_pibo": self.beta_pibo,
            "exponent": exponent,
            "n_labeled": L_n,
            "n_top_genes": len(top_genes),
            "n_sig_pathways": n_sig_pathways,
            "top": [
                {
                    "gene": c,
                    "score": float(scores.get(c, 0.0)),
                    "unc": float(unc.get(c, 0.0)),
                    "ucb": float(ucb(c)),
                    "prior": float(prior.get(str(c), 0.0)),
                    "pi_ucb": float(pi_ucb(c)),
                }
                for c in picked[:min(10, len(picked))]
            ],
        }
        return picked


__all__ = ["BioUCBFromModel"]
