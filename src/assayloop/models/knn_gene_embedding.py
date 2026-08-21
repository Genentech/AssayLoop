"""KNN model over a gene-embedding space.

Score(gene) = distance-weighted mean hit-label of the K nearest
already-acquired neighbours under cosine distance.

Defaults to PRESAGE/GenePT embeddings. The provider is fully
swappable so the GA outer loop can sweep over knowledge sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ..data.gene_embeddings import (
    GeneEmbeddingProvider,
    PresageGeneEmbedding,
)
from ..data.gene_embeddings.ortholog import map_mgi_to_hgnc


def _default_provider(organism: str | None = None) -> GeneEmbeddingProvider:
    """The GenePT embeddings this baseline's reported numbers are computed in.

    Raises if the PRESAGE cache has not been downloaded
    (``scripts/fetch_presage_cache.sh``). There is deliberately no one-hot
    fallback: under one-hot every gene is equidistant from every other, so
    "nearest neighbour" degenerates to an arbitrary tie-break -- a different
    baseline reported under the same name. Pass ``provider=`` to smoke-test
    with something cheap on purpose.
    """
    normalizer = None
    if organism and "musculus" in str(organism).lower():
        normalizer = map_mgi_to_hgnc
    return PresageGeneEmbedding(source="genept", gene_normalizer=normalizer)


class KNNGeneEmbedding(Model):
    """Distance-weighted KNN on a gene-embedding space.

    Args:
        provider: a GeneEmbeddingProvider. If ``None``, defaults to
            PRESAGE/GenePT, which must be downloaded.
        k: number of neighbours (default 10).
        score_field: which observation label field to read as the
            target signal. Either ``"hit"`` (boolean -> ``{0, 1}``) or
            ``"relevance_score"`` (regress on continuous relevance).
            Default ``"hit"``.
        epsilon: tiny constant added to distances to avoid div-by-zero.
    """

    def __init__(
        self,
        provider: GeneEmbeddingProvider | None = None,
        *,
        k: int = 10,
        score_field: str = "hit",
        epsilon: float = 1e-6,
    ):
        self._provider = provider
        self._k = int(k)
        self._score_field = score_field
        self._epsilon = float(epsilon)
        # Per-screen cached embedding matrix (rebuilt on reset()).
        self._acquired_genes: list[str] = []
        self._acquired_X: np.ndarray | None = None
        self._acquired_y: np.ndarray | None = None

    def reset(self) -> None:
        self._acquired_genes = []
        self._acquired_X = None
        self._acquired_y = None

    def coverage(self, candidates: list) -> float | None:
        provider = self._ensure_provider(None)
        _, mask = provider.embed_batch(candidates)
        return float(mask.mean())

    def name(self) -> str:
        p = self._provider.name() if self._provider else "default"
        return f"knn_k{self._k}_{p}"

    def _ensure_provider(self, task_context: dict[str, Any] | None) -> GeneEmbeddingProvider:
        if self._provider is None:
            organism = (task_context or {}).get("organism")
            self._provider = _default_provider(organism=organism)
        return self._provider

    def _label_of(self, obs: Observation) -> float:
        label = obs.label
        if isinstance(label, dict):
            if self._score_field == "hit":
                return 1.0 if label.get("hit") else 0.0
            return float(label.get(self._score_field, 0.0))
        return 1.0 if label else 0.0

    def _build_train(self, observations: list[Observation], provider: GeneEmbeddingProvider) -> None:
        if not observations:
            self._acquired_genes = []
            self._acquired_X = None
            self._acquired_y = None
            return
        genes = [o.candidate for o in observations]
        X, mask = provider.embed_batch(genes)
        if not mask.any():
            self._acquired_genes = genes
            self._acquired_X = None
            self._acquired_y = None
            return
        X = X[mask]
        ys = np.array([self._label_of(o) for o in observations], dtype=np.float32)[mask]
        self._acquired_genes = [g for g, k in zip(genes, mask) if k]
        # L2-normalise for cosine.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._acquired_X = (X / norms).astype(np.float32)
        self._acquired_y = ys

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        provider = self._ensure_provider(task_context)
        self._build_train(observations, provider)

        scores: dict[Any, float] = {}
        uncertainty: dict[Any, float] = {}
        if self._acquired_X is None or self._acquired_y is None:
            base = float(np.mean([self._label_of(o) for o in observations])) if observations else 0.0
            for c in candidates:
                scores[c] = base
                uncertainty[c] = 1.0
            return ModelPrediction(scores=scores, uncertainty=uncertainty, metadata={"name": self.name(), "n_train": 0})

        Xq, qmask = provider.embed_batch(candidates)
        # L2-normalise queries.
        qnorms = np.linalg.norm(Xq, axis=1, keepdims=True)
        qnorms[qnorms == 0] = 1
        Xq_n = (Xq / qnorms).astype(np.float32)

        # Cosine similarity: (n_query, n_train)
        S = Xq_n @ self._acquired_X.T  # in [-1, 1]
        k = min(self._k, S.shape[1])
        # Top-K similarities per query.
        top_idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        # Gather similarities.
        rows = np.arange(S.shape[0])[:, None]
        top_sim = S[rows, top_idx]
        top_lbl = self._acquired_y[top_idx]

        # Distance-weighted vote: weight = 1 / (1 - sim + epsilon)
        weights = 1.0 / (1.0 - top_sim + self._epsilon)
        w_sum = weights.sum(axis=1)
        pred = (weights * top_lbl).sum(axis=1) / w_sum

        # Uncertainty: variance of top-K labels (small if neighbours agree).
        var = ((top_lbl - pred[:, None]) ** 2 * weights).sum(axis=1) / w_sum

        for i, c in enumerate(candidates):
            if qmask[i]:
                scores[c] = float(pred[i])
                uncertainty[c] = float(var[i])
            else:
                # Unknown gene in embedding: assign baseline label and high uncertainty.
                scores[c] = float(self._acquired_y.mean())
                uncertainty[c] = 1.0

        return ModelPrediction(
            scores=scores,
            uncertainty=uncertainty,
            metadata={
                "name": self.name(),
                "n_train": int(self._acquired_X.shape[0]),
                "provider": provider.name(),
                "coverage_q": float(qmask.mean()),
            },
        )


__all__ = ["KNNGeneEmbedding"]
