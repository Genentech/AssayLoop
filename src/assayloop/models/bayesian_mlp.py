"""Bayesian MLP with MC Dropout over gene embeddings.

Architecture (following Haystack / Gal & Ghahramani 2016):
    Linear(d -> 128) -> ReLU -> Dropout(0.2)
    Linear(128 -> 64) -> ReLU -> Dropout(0.2)
    Linear(64 -> 1) [-> Sigmoid if classification]

Classification: BCELoss, scores[g] = P(hit | g, D_t).
Regression:     MSELoss, scores[g] = predicted relevance score.

Uncertainty estimated via MC Dropout (n_mc_samples stochastic forward passes).
Defaults to GenePT-ada embeddings with the same fallback chain as KNN/RF models.
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


def _require_torch():
    """Import torch with a clear hint, since it's an optional ``mlp`` extra."""
    try:
        import torch  # noqa: F401

        return torch
    except ModuleNotFoundError as e:  # pragma: no cover - depends on install
        raise ModuleNotFoundError(
            "The bayesian_mlp model requires PyTorch, which is an optional "
            "dependency. Install it with `uv sync --extra mlp` "
            "(or `pip install torch`)."
        ) from e


def _default_provider(organism: str | None = None) -> GeneEmbeddingProvider:
    """The GenePT embeddings this baseline's reported numbers are computed in.

    Raises if the PRESAGE cache has not been downloaded. There is deliberately
    no one-hot fallback: one-hot features are mutually orthogonal, so the model
    would be a different (and far weaker) baseline reported under the same
    name. Pass ``provider=`` explicitly to use something else on purpose.
    """
    normalizer = None
    if organism and "musculus" in str(organism).lower():
        normalizer = map_mgi_to_hgnc
    return PresageGeneEmbedding(source="GenePT_ada", gene_normalizer=normalizer)


class BayesianMLPModel(Model):
    """MC-Dropout Bayesian MLP on gene embeddings.

    Args:
        provider: GeneEmbeddingProvider. ``None`` -> GenePT_ada with one-hot fallback.
        task_type: ``"classification"`` (hit/no-hit, BCELoss + Sigmoid, default) or
            ``"regression"`` (continuous relevance_score, MSELoss).
        hidden_dims: hidden layer sizes (default [128, 64]).
        dropout_p: dropout probability (default 0.2).
        n_mc_samples: stochastic forward passes at inference (default 50).
        lr: Adam learning rate (default 1e-3).
        max_epochs: max training epochs per AL step (default 100).
        patience: early-stopping patience in epochs (default 10).
        batch_size: mini-batch size for training (default 64).
        random_state: RNG seed.
    """

    def __init__(
        self,
        provider: GeneEmbeddingProvider | None = None,
        *,
        task_type: str = "classification",
        hidden_dims: list[int] | None = None,
        dropout_p: float = 0.2,
        n_mc_samples: int = 50,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 64,
        random_state: int = 0,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        self._task_type = task_type
        self._provider = provider
        self._hidden_dims = hidden_dims or [128, 64]
        self._dropout_p = float(dropout_p)
        self._n_mc_samples = int(n_mc_samples)
        self._lr = float(lr)
        self._max_epochs = int(max_epochs)
        self._patience = int(patience)
        self._batch_size = int(batch_size)
        self._random_state = int(random_state)

    def reset(self) -> None:
        pass

    def coverage(self, candidates: list) -> float | None:
        provider = self._ensure_provider(None)
        _, mask = provider.embed_batch(candidates)
        return float(mask.mean())

    def name(self) -> str:
        p = self._provider.name() if self._provider else "default"
        dims = "-".join(str(d) for d in self._hidden_dims)
        return f"bayesian_mlp_{self._task_type}_{dims}_p{self._dropout_p}_{p}"

    def _ensure_provider(self, task_context: dict[str, Any] | None) -> GeneEmbeddingProvider:
        if self._provider is None:
            organism = (task_context or {}).get("organism")
            self._provider = _default_provider(organism=organism)
        return self._provider

    def _label_of(self, obs: Observation) -> float:
        label = obs.label
        if isinstance(label, dict):
            if self._task_type == "classification":
                return 1.0 if label.get("hit") else 0.0
            return float(label.get("relevance_score", 0.0))
        return float(label) if label is not None else 0.0

    def _build_net(self, input_dim: int):
        import torch.nn as nn

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in self._hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(self._dropout_p)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        if self._task_type == "classification":
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _train(self, X: np.ndarray, y: np.ndarray):
        _require_torch()
        import torch
        import torch.nn as nn

        torch.manual_seed(self._random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xt = torch.from_numpy(X).float().to(device)
        yt = torch.from_numpy(y).float().unsqueeze(1).to(device)

        net = self._build_net(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self._lr)
        loss_fn = nn.BCELoss() if self._task_type == "classification" else nn.MSELoss()

        n = len(Xt)
        n_val = max(1, int(0.2 * n))
        perm = torch.randperm(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        Xtr, ytr = Xt[train_idx], yt[train_idx]
        Xvl, yvl = Xt[val_idx], yt[val_idx]

        best_val, best_state, wait = float("inf"), None, 0
        for _ in range(self._max_epochs):
            net.train()
            perm_b = torch.randperm(len(Xtr))
            for start in range(0, len(Xtr), self._batch_size):
                idx = perm_b[start : start + self._batch_size]
                optimizer.zero_grad()
                loss_fn(net(Xtr[idx]), ytr[idx]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(Xvl), yvl).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self._patience:
                    break

        if best_state is not None:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return net, device

    def _mc_predict(self, net, X: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
        import torch

        Xt = torch.from_numpy(X).float().to(device)
        net.train()  # keep dropout active
        with torch.no_grad():
            samples = torch.stack(
                [net(Xt).squeeze(1) for _ in range(self._n_mc_samples)], dim=0
            )
        mean = samples.mean(0).cpu().numpy()
        var = samples.var(0).cpu().numpy()
        return mean, var

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        provider = self._ensure_provider(task_context)
        if not observations:
            return ModelPrediction(
                scores={c: 0.0 for c in candidates},
                uncertainty={c: 1.0 for c in candidates},
                metadata={"name": self.name(), "n_train": 0},
            )

        X_tr_full, tr_mask = provider.embed_batch([o.candidate for o in observations])
        y_tr_full = np.array([self._label_of(o) for o in observations], dtype=np.float32)
        X_tr = X_tr_full[tr_mask].astype(np.float32)
        y_tr = y_tr_full[tr_mask]
        base = float(y_tr.mean()) if len(y_tr) else 0.0

        if len(X_tr) < 2:
            return ModelPrediction(
                scores={c: base for c in candidates},
                uncertainty={c: 1.0 for c in candidates},
                metadata={"name": self.name(), "n_train": int(len(X_tr))},
            )

        net, device = self._train(X_tr, y_tr)

        X_q, q_mask = provider.embed_batch(candidates)
        X_q = X_q.astype(np.float32)
        mean, var = self._mc_predict(net, X_q, device)

        scores: dict[Any, float] = {}
        uncertainty: dict[Any, float] = {}
        for i, c in enumerate(candidates):
            if q_mask[i]:
                scores[c] = float(mean[i])
                uncertainty[c] = float(var[i])
            else:
                scores[c] = base
                uncertainty[c] = 1.0

        return ModelPrediction(
            scores=scores,
            uncertainty=uncertainty,
            metadata={
                "name": self.name(),
                "n_train": int(len(X_tr)),
                "provider": provider.name(),
                "coverage_q": float(q_mask.mean()),
            },
        )


__all__ = ["BayesianMLPModel"]
