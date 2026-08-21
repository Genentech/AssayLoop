"""BPMF Model for sequential optimization.

Wraps a pre-trained BPMFResult (frozen gene factors V) and computes the
posterior of a new screen's latent embedding U given incrementally
revealed hits. Predicts P(hit) for unobserved genes via Φ(u · v_j).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from .. import config
from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation

log = logging.getLogger(__name__)

_TRAIN_HINT = (
    "Train one with `python -m assayloop.scripts.train_bpmf_gpu`, or point "
    "ASSAYLOOP_BPMF_CHECKPOINT at an existing bpmf_result.pkl."
)


class BPMFModel(Model):

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        n_posterior_samples: int = 100,
        n_gibbs: int = 20,
        seed: int = 42,
        device: str = "cpu",
    ):
        # No bundled checkpoint: default to ASSAYLOOP_BPMF_CHECKPOINT (which
        # itself defaults under output/), and fail naming it rather than
        # scoring on an untrained factorisation.
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.BPMF_CHECKPOINT
        self._n_posterior_samples = n_posterior_samples
        self._n_gibbs = n_gibbs
        self._seed = seed
        # "cpu" (default) -> exact numpy path. "cuda"/"cuda:N" -> batched torch
        # path (same math, statistically equivalent up to RNG/float differences).
        self._device = device

        self._loaded = False
        self._V_samples: np.ndarray | None = None
        self._V_t = None  # torch tensor (n_samples, n_genes, K) on device, GPU path
        self._gene_names: list[str] = []
        self._gene_to_idx: dict[str, int] = {}
        self._K: int = 0
        self._sigma_u: float = 1.0

    @property
    def _use_torch(self) -> bool:
        return not str(self._device).startswith("cpu")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        path = config.require_path(
            self._checkpoint_path,
            env_var="ASSAYLOOP_BPMF_CHECKPOINT",
            what="the BPMF checkpoint",
            hint=_TRAIN_HINT,
        )
        log.info("Loading BPMFResult from %s", path)
        with open(path, "rb") as f:
            result = pickle.load(f)

        self._gene_names = result.gene_names
        self._gene_to_idx = {g: i for i, g in enumerate(self._gene_names)}
        self._K = result.K
        self._sigma_u = result.sigma_u

        n_total = result.V_samples.shape[0]
        step = max(1, n_total // self._n_posterior_samples)
        self._V_samples = result.V_samples[::step]
        log.info(
            "BPMF model ready: %d V samples retained (of %d), K=%d, %d genes"
            " [device=%s]",
            self._V_samples.shape[0], n_total, self._K, len(self._gene_names),
            self._device,
        )
        if self._use_torch:
            import torch
            self._torch_device = torch.device(self._device)
            # float64 to match the CPU path: float32 Cholesky/solve of the
            # posterior precision breaks down (NaNs) once many observations
            # accumulate over acquisition steps. The heavy op (all-gene matvec)
            # is tiny, so fp64 is still fast on-GPU.
            self._V_t = torch.as_tensor(
                np.ascontiguousarray(self._V_samples), dtype=torch.float64,
                device=self._torch_device,
            )
            self._V_samples = None  # free host copy; GPU path uses _V_t
        self._loaded = True

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return "bpmf"

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        self._ensure_loaded()
        if self._use_torch:
            return self._predict_torch(observations, candidates)
        assert self._V_samples is not None

        rng = np.random.default_rng(self._seed)

        obs_indices: list[int] = []
        obs_labels: list[float] = []
        for obs in observations:
            gene = obs.candidate
            if gene not in self._gene_to_idx:
                continue
            idx = self._gene_to_idx[gene]
            label = obs.label
            if isinstance(label, dict):
                obs_labels.append(1.0 if label.get("hit") else 0.0)
            else:
                obs_labels.append(1.0 if label else 0.0)
            obs_indices.append(idx)

        cand_indices: list[int] = []
        cand_genes: list[str] = []
        for gene in candidates:
            if gene in self._gene_to_idx:
                cand_indices.append(self._gene_to_idx[gene])
                cand_genes.append(gene)

        cand_idx_arr = np.array(cand_indices, dtype=int)
        n_samples = self._V_samples.shape[0]
        K = self._K
        precision_u = 1.0 / self._sigma_u**2

        if not obs_indices:
            prior_prob = 0.5
            scores = {gene: prior_prob for gene in candidates}
            return ModelPrediction(scores=scores, metadata={"name": self.name(), "n_obs": 0})

        obs_idx_arr = np.array(obs_indices, dtype=int)
        y_obs = np.array(obs_labels, dtype=np.float64)

        probs_all = np.zeros((n_samples, len(cand_indices)), dtype=np.float64)

        for s in range(n_samples):
            V = self._V_samples[s]
            V_obs = V[obs_idx_arr]
            V_cand = V[cand_idx_arr]

            u = rng.normal(0, self._sigma_u, size=K)

            for _ in range(self._n_gibbs):
                mean_z = V_obs @ u
                z_obs = _sample_truncnorm(mean_z, y_obs, rng)
                u = _sample_u(V_obs, z_obs, precision_u, rng)

            probs_all[s] = ndtr(V_cand @ u)

        mean_probs = probs_all.mean(axis=0)
        std_probs = probs_all.std(axis=0)

        scores: dict[Any, float] = {}
        uncertainty: dict[Any, float] = {}
        for i, gene in enumerate(cand_genes):
            scores[gene] = float(mean_probs[i])
            uncertainty[gene] = float(std_probs[i])

        for gene in candidates:
            if gene not in scores:
                scores[gene] = 0.5
                uncertainty[gene] = 0.0

        return ModelPrediction(
            scores=scores,
            uncertainty=uncertainty,
            metadata={"name": self.name(), "n_obs": len(obs_indices), "n_samples": n_samples},
        )


    def _predict_torch(self, observations, candidates):
        """Batched GPU port of :meth:`predict` (same math, batched over the
        posterior samples). Statistically equivalent to the numpy path up to
        RNG stream and float32 rounding — validated by the parity check."""
        import torch

        dev = self._torch_device
        Vt = self._V_t                      # (S, n_genes, K) float32 on device
        S, _, K = Vt.shape
        dt = Vt.dtype
        prec = 1.0 / self._sigma_u ** 2

        obs_idx, obs_labels = [], []
        for obs in observations:
            g = obs.candidate
            if g not in self._gene_to_idx:
                continue
            label = obs.label
            hit = label.get("hit") if isinstance(label, dict) else label
            obs_labels.append(1.0 if hit else 0.0)
            obs_idx.append(self._gene_to_idx[g])

        cand_idx, cand_genes = [], []
        for g in candidates:
            if g in self._gene_to_idx:
                cand_idx.append(self._gene_to_idx[g])
                cand_genes.append(g)

        if not obs_idx:
            scores = {g: 0.5 for g in candidates}
            return ModelPrediction(scores=scores,
                                   metadata={"name": self.name(), "n_obs": 0})

        gen = torch.Generator(device=dev).manual_seed(int(self._seed))
        oi = torch.as_tensor(obs_idx, device=dev)
        ci = torch.as_tensor(cand_idx, device=dev)
        ypos = torch.as_tensor(obs_labels, device=dev, dtype=dt) == 1  # (n_obs,)
        V_obs = Vt[:, oi, :]                                            # (S, n_obs, K)

        eye = torch.eye(K, device=dev, dtype=dt)
        Lambda = V_obs.transpose(-1, -2) @ V_obs + prec * eye          # (S, K, K)
        L = torch.linalg.cholesky(Lambda)                              # (S, K, K)

        u = torch.randn(S, K, generator=gen, device=dev, dtype=dt) * self._sigma_u
        eps = 1e-8
        for _ in range(self._n_gibbs):
            mean_z = torch.einsum("sok,sk->so", V_obs, u)              # (S, n_obs)
            uu = torch.rand(mean_z.shape, generator=gen, device=dev, dtype=dt)
            bound = torch.special.ndtr(-mean_z)                        # Phi(-mean)
            lo = torch.where(ypos, bound, torch.zeros_like(bound))
            hi = torch.where(ypos, torch.ones_like(bound), bound)
            lo = lo.clamp(eps, 1 - eps); hi = hi.clamp(eps, 1 - eps)
            p = (lo + uu * (hi - lo)).clamp(eps, 1 - eps)
            z = mean_z + torch.special.ndtri(p)                        # (S, n_obs)
            rhs = torch.einsum("sok,so->sk", V_obs, z)                 # (S, K)
            mu = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
            noise = torch.randn(S, K, generator=gen, device=dev, dtype=dt)
            # match CPU: noise = L^{-T} z, so cov = L^{-T} L^{-1} = Lambda^{-1}.
            sol = torch.linalg.solve_triangular(
                L.transpose(-1, -2), noise.unsqueeze(-1), upper=True).squeeze(-1)
            u = mu + sol                                               # (S, K)

        # score all genes (avoids a large per-candidate gather), then index
        logits_all = torch.einsum("sgk,sk->sg", Vt, u)                # (S, n_genes)
        probs = torch.special.ndtr(logits_all).mean(dim=0)            # (n_genes,)
        probs_cand = probs[ci].detach().cpu().numpy()

        scores = {g: float(probs_cand[i]) for i, g in enumerate(cand_genes)}
        for g in candidates:
            if g not in scores:
                scores[g] = 0.5
        return ModelPrediction(
            scores=scores,
            metadata={"name": self.name(), "n_obs": len(obs_idx), "n_samples": S},
        )


def _sample_u(
    V_obs: np.ndarray, z_obs: np.ndarray, precision_u: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the screen latent u ~ N(Lambda^{-1} V_obsᵀ z, Lambda^{-1}).

    ``Lambda = V_obsᵀ V_obs + precision_u I`` is the posterior precision. With
    ``Lambda = L Lᵀ`` the noise term must be ``L^{-T} xi``, whose covariance is
    ``L^{-T} L^{-1} = Lambda^{-1}``. Solving against ``L`` instead yields
    ``L^{-1} L^{-T}`` -- a different matrix with the *same eigenvalues*, i.e.
    the right spread in the wrong orientation, which is why the error is easy
    to miss by eye. ``tests/test_bpmf_sampler.py`` pins the covariance.
    """
    K = V_obs.shape[1]
    Lambda = V_obs.T @ V_obs + precision_u * np.eye(K)
    mu = np.linalg.solve(Lambda, V_obs.T @ z_obs)
    L = np.linalg.cholesky(Lambda)
    return mu + np.linalg.solve(L.T, rng.standard_normal(K))


def _sample_truncnorm(
    mean: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample z from truncated N(mean, 1). y=1 -> (0, inf), y=0 -> (-inf, 0).

    Inverse-CDF sampling: the standardized truncation point at Z=0 is
    ``Phi(-mean)``, so y=1 uses ``p in [Phi(-mean), 1]`` and y=0 uses
    ``p in [0, Phi(-mean)]``, then ``z = mean + Phi^{-1}(p)``.
    """
    eps = 1e-8
    u = rng.random(mean.shape)
    pos = y == 1

    bound = ndtr(-mean)
    lo = np.where(pos, bound, 0.0)
    hi = np.where(pos, 1.0, bound)
    lo = np.clip(lo, eps, 1.0 - eps)
    hi = np.clip(hi, eps, 1.0 - eps)
    p = np.clip(lo + u * (hi - lo), eps, 1.0 - eps)
    return mean + ndtri(p)


__all__ = ["BPMFModel"]
