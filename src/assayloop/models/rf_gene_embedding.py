"""Random-Forest classifier on a gene-embedding space.

Trains an sklearn RandomForestClassifier/Regressor per AL step on
acquired observations. Exposes:

- ``scores[gene]`` = predicted hit probability (classification)
                    or predicted relevance score (regression).
- ``uncertainty[gene]`` = ensemble variance across trees, used by UCB.

Fits in ~tens of milliseconds for the 100-200 acquired-gene scale we
operate at; per-step retraining is fine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ..data.gene_embeddings import GeneEmbeddingProvider, PresageGeneEmbedding
from ..data.gene_embeddings.ortholog import map_mgi_to_hgnc


def _default_provider(organism: str | None = None) -> GeneEmbeddingProvider:
    """The GenePT embeddings this baseline's reported numbers are computed in.

    Raises if the PRESAGE cache has not been downloaded. There is deliberately
    no one-hot fallback: one-hot features are mutually orthogonal, so the trees
    would split on gene identity alone -- a different baseline reported under
    the same name. Pass ``provider=`` explicitly to use something else.
    """
    normalizer = None
    if organism and "musculus" in str(organism).lower():
        normalizer = map_mgi_to_hgnc
    return PresageGeneEmbedding(source="genept", gene_normalizer=normalizer)


class RFGeneEmbedding(Model):
    """Random-Forest classifier/regressor on gene embeddings.

    Args:
        provider: GeneEmbeddingProvider. ``None`` -> PRESAGE default.
        n_estimators: number of trees (default 200).
        task_type: ``"classification"`` (hit/no-hit, default) or
            ``"regression"`` (continuous relevance_score).
        random_state: RNG seed for sklearn.
    """

    def __init__(
        self,
        provider: GeneEmbeddingProvider | None = None,
        *,
        n_estimators: int = 200,
        task_type: str = "classification",
        random_state: int = 0,
    ):
        self._provider = provider
        self._n_estimators = int(n_estimators)
        self._task_type = task_type
        self._random_state = int(random_state)
        self._clf = None

    def reset(self) -> None:
        self._clf = None

    def coverage(self, candidates: list) -> float | None:
        provider = self._ensure_provider(None)
        _, mask = provider.embed_batch(candidates)
        return float(mask.mean())

    def name(self) -> str:
        p = self._provider.name() if self._provider else "default"
        return f"rf_{self._task_type}_n{self._n_estimators}_{p}"

    def _ensure_provider(self, task_context: dict[str, Any] | None) -> GeneEmbeddingProvider:
        if self._provider is None:
            organism = (task_context or {}).get("organism")
            self._provider = _default_provider(organism=organism)
        return self._provider

    @staticmethod
    def _label_of(obs: Observation, kind: str) -> float | int:
        label = obs.label
        if isinstance(label, dict):
            if kind == "classification":
                return 1 if label.get("hit") else 0
            return float(label.get("relevance_score", 0.0))
        return int(bool(label)) if kind == "classification" else float(label or 0.0)

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        provider = self._ensure_provider(task_context)

        # Build training matrix.
        if not observations:
            return ModelPrediction(
                scores={c: 0.0 for c in candidates},
                uncertainty={c: 1.0 for c in candidates},
                metadata={"name": self.name(), "n_train": 0},
            )

        Xtr_full, train_mask = provider.embed_batch([o.candidate for o in observations])
        ytr_full = np.array([self._label_of(o, self._task_type) for o in observations])
        Xtr = Xtr_full[train_mask]
        ytr = ytr_full[train_mask]

        # If only one class present we can't train a classifier; fall back
        # to the empirical base rate.
        if self._task_type == "classification" and len(set(ytr.tolist())) < 2:
            base = float(np.mean(ytr)) if len(ytr) else 0.0
            return ModelPrediction(
                scores={c: base for c in candidates},
                uncertainty={c: 0.5 for c in candidates},
                metadata={"name": self.name(), "n_train": int(len(ytr)), "single_class": True},
            )

        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )

        if self._task_type == "classification":
            clf = RandomForestClassifier(
                n_estimators=self._n_estimators,
                n_jobs=-1,
                random_state=self._random_state,
                bootstrap=True,
            )
            clf.fit(Xtr, ytr)
            Xq, qmask = provider.embed_batch(candidates)
            # Mean per-tree predicted probabilities -> use that as score
            # and variance across trees -> uncertainty.
            per_tree_p = np.stack(
                [t.predict_proba(Xq)[:, list(t.classes_).index(1)] if 1 in t.classes_ else np.zeros(Xq.shape[0])
                 for t in clf.estimators_],
                axis=0,
            )
            scores = per_tree_p.mean(axis=0)
            unc = per_tree_p.var(axis=0)
        else:
            reg = RandomForestRegressor(
                n_estimators=self._n_estimators,
                n_jobs=-1,
                random_state=self._random_state,
                bootstrap=True,
            )
            reg.fit(Xtr, ytr)
            Xq, qmask = provider.embed_batch(candidates)
            per_tree_y = np.stack(
                [t.predict(Xq) for t in reg.estimators_], axis=0,
            )
            scores = per_tree_y.mean(axis=0)
            unc = per_tree_y.var(axis=0)

        out_scores: dict[Any, float] = {}
        out_unc: dict[Any, float] = {}
        base = float(np.mean(ytr))
        for i, c in enumerate(candidates):
            if qmask[i]:
                out_scores[c] = float(scores[i])
                out_unc[c] = float(unc[i])
            else:
                out_scores[c] = base
                out_unc[c] = 1.0

        return ModelPrediction(
            scores=out_scores,
            uncertainty=out_unc,
            metadata={
                "name": self.name(),
                "n_train": int(len(ytr)),
                "provider": provider.name(),
                "coverage_q": float(qmask.mean()),
            },
        )


__all__ = ["RFGeneEmbedding"]
