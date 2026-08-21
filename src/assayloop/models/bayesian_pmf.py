"""Bayesian Probit Matrix Factorization via Gibbs sampling.

Model:
    Y[i,j] in {0,1}  — hit indicator for screen i, gene j
    Z[i,j] = U[i] @ V[j] + eps,  eps ~ N(0,1)
    Y[i,j] = 1{Z[i,j] > 0}

Priors:
    U[i] ~ N(0, sigma_u^2 I_K)       (screen factors)
    V[j] ~ N(0, sigma_v^2 I_K)       (gene factors)

Gibbs steps:
    1. Z | Y, U, V  — truncated normal (Albert & Chib 1993)
    2. U | Z, V     — conjugate normal
    3. V | Z, U     — conjugate normal

Gene factors V can be frozen after fitting so only screen factors U
are re-sampled during BO.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

log = logging.getLogger(__name__)


@dataclass
class BPMFResult:
    U_samples: np.ndarray  # (n_samples, n_screens, K)
    V_samples: np.ndarray  # (n_samples, n_genes, K)
    log_lik: np.ndarray    # (n_iter,)
    gene_names: list[str]
    screen_names: list[str]
    K: int
    sigma_u: float
    sigma_v: float

    def posterior_mean_U(self) -> np.ndarray:
        return self.U_samples.mean(axis=0)

    def posterior_mean_V(self) -> np.ndarray:
        return self.V_samples.mean(axis=0)

    def predicted_probs(self, burn_in: int = 0) -> np.ndarray:
        U = self.U_samples[burn_in:]
        V = self.V_samples[burn_in:]
        M = np.einsum("sik,sjk->sij", U, V)
        return ndtr(M).mean(axis=0)


def _fast_truncnorm(mean: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample Z from truncated N(mean, 1) via inverse CDF.

    y=1 -> truncated to (0, inf),  y=0 -> truncated to (-inf, 0).

    For N(mean, 1) truncated to [a, b], inverse-CDF sampling draws
    ``p ~ U(Phi(a-mean), Phi(b-mean))`` then returns ``mean + Phi^{-1}(p)``.
    The single standardized truncation point (at Z=0) is ``Phi(-mean)``, so
    y=1 uses ``p in [Phi(-mean), 1]`` and y=0 uses ``p in [0, Phi(-mean)]``;
    the mean must be added back to the standardized draw.
    """
    eps = 1e-8
    bound = ndtr(-mean)                 # Phi(-mean): standardized truncation point
    u = rng.random(mean.shape)

    pos = y == 1
    lo = np.where(pos, bound, 0.0)
    hi = np.where(pos, 1.0, bound)

    lo = np.clip(lo, eps, 1.0 - eps)
    hi = np.clip(hi, eps, 1.0 - eps)
    p = lo + u * (hi - lo)
    p = np.clip(p, eps, 1.0 - eps)
    return mean + ndtri(p)


def _group_by_missing(mask: np.ndarray, axis: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group items by their MISSING indices (complement of observed)."""
    groups: dict[bytes, list[int]] = {}
    missing_cache: dict[bytes, np.ndarray] = {}
    n = mask.shape[axis]
    for i in range(n):
        if axis == 0:
            col = ~mask[i]
        else:
            col = ~mask[:, i]
        key = col.tobytes()
        if key not in missing_cache:
            missing_cache[key] = np.nonzero(col)[0]
        groups.setdefault(key, []).append(i)
    result = [(missing_cache[k], np.array(v)) for k, v in groups.items()]
    result.sort(key=lambda x: len(x[1]), reverse=True)
    return result


def gibbs_bpmf(
    Y: np.ndarray,
    *,
    K: int = 10,
    sigma_u: float = 1.0,
    sigma_v: float = 1.0,
    n_iter: int = 1000,
    burn_in: int = 500,
    thin: int = 1,
    seed: int = 42,
    mask: np.ndarray | None = None,
) -> BPMFResult:
    """Run Gibbs sampling for Bayesian probit matrix factorization.

    Args:
        Y: (n_screens, n_genes) binary matrix.
        K: latent dimension.
        sigma_u: prior std for screen factors.
        sigma_v: prior std for gene factors.
        n_iter: total Gibbs iterations.
        burn_in: iterations to discard.
        thin: keep every thin-th sample after burn-in.
        seed: RNG seed.
        mask: optional (n_screens, n_genes) bool, True = observed. If None,
              all entries treated as observed.

    Returns:
        BPMFResult with posterior samples and log-likelihood trace.
    """
    rng = np.random.default_rng(seed)
    n_screens, n_genes = Y.shape
    Y = Y.astype(np.float64)

    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    U = rng.normal(0, 0.1, size=(n_screens, K))
    V = rng.normal(0, 0.1, size=(n_genes, K))
    Z = np.zeros_like(Y)

    precision_u = 1.0 / sigma_u**2
    precision_v = 1.0 / sigma_v**2
    eye_K = np.eye(K)

    screen_groups = _group_by_missing(mask, axis=0)
    gene_groups = _group_by_missing(mask, axis=1)
    n_gene_groups = len(gene_groups)
    n_screen_groups = len(screen_groups)

    # Precompute per-gene and per-screen group assignments for vectorized sampling
    gene_group_idx = np.empty(n_genes, dtype=np.int32)
    for g, (_, idxs) in enumerate(gene_groups):
        gene_group_idx[idxs] = g

    screen_group_idx = np.empty(n_screens, dtype=np.int32)
    for g, (_, idxs) in enumerate(screen_groups):
        screen_group_idx[idxs] = g

    log.info(
        "BPMF Gibbs: %d screens × %d genes, K=%d, %d screen groups, %d gene groups",
        n_screens, n_genes, K, n_screen_groups, n_gene_groups,
    )

    # Precompute mask as float for masked matmul
    mask_f = mask.astype(np.float64)

    U_samples = []
    V_samples = []
    log_liks = []

    from tqdm.auto import tqdm
    pbar = tqdm(range(n_iter), desc="Gibbs sampling", unit="iter")
    for it in pbar:
        # --- Step 1: sample Z from truncated normals ---
        mean_Z = U @ V.T
        Z[mask] = _fast_truncnorm(mean_Z[mask], Y[mask], rng)
        Z[~mask] = mean_Z[~mask]

        # --- Step 2: sample U (screen factors) ---
        # rhs for ALL screens at once: V[obs_j].T @ Z[i, obs_j] = (V.T @ (Z * mask).T)
        # which equals V.T @ (mask * Z).T = (K, n_screens)
        VtV_full = V.T @ V + precision_u * eye_K
        rhs_U_all = V.T @ (Z * mask_f).T  # (K, n_screens)

        # Build Lam per screen group using complement trick
        Lams_s = np.empty((n_screen_groups, K, K))
        for g, (missing, _) in enumerate(screen_groups):
            if len(missing) == 0:
                Lams_s[g] = VtV_full
            else:
                V_miss = V[missing]
                Lams_s[g] = VtV_full - V_miss.T @ V_miss

        Ls_s = np.linalg.cholesky(Lams_s)
        eyes_batch = np.broadcast_to(eye_K, (n_screen_groups, K, K)).copy()
        Lam_invs_s = np.linalg.solve(Lams_s, eyes_batch)
        L_invs_s = np.linalg.solve(Ls_s, eyes_batch)

        # Vectorized mean + sampling: expand per-group to per-screen
        Lam_inv_per_screen = Lam_invs_s[screen_group_idx]  # (n_screens, K, K)
        L_inv_per_screen = L_invs_s[screen_group_idx]      # (n_screens, K, K)
        mus_U = np.einsum("ijk,ki->ij", Lam_inv_per_screen, rhs_U_all)  # (n_screens, K)
        raw_U = rng.standard_normal((n_screens, K))
        U = mus_U + np.einsum("ijk,ij->ik", L_inv_per_screen, raw_U)

        # --- Step 3: sample V (gene factors) ---
        UtU_full = U.T @ U + precision_v * eye_K
        rhs_V_all = U.T @ (Z * mask_f)  # (K, n_genes)

        # Build Lam per gene group using complement trick
        Lams_g = np.empty((n_gene_groups, K, K))
        for g, (missing, _) in enumerate(gene_groups):
            if len(missing) == 0:
                Lams_g[g] = UtU_full
            else:
                U_miss = U[missing]
                Lams_g[g] = UtU_full - U_miss.T @ U_miss

        Ls_g = np.linalg.cholesky(Lams_g)
        eyes_batch_g = np.broadcast_to(eye_K, (n_gene_groups, K, K)).copy()
        Lam_invs_g = np.linalg.solve(Lams_g, eyes_batch_g)
        L_invs_g = np.linalg.solve(Ls_g, eyes_batch_g)

        # Vectorized mean + sampling: expand per-group to per-gene
        Lam_inv_per_gene = Lam_invs_g[gene_group_idx]  # (n_genes, K, K)
        L_inv_per_gene = L_invs_g[gene_group_idx]      # (n_genes, K, K)
        mus_V = np.einsum("jkl,lj->jk", Lam_inv_per_gene, rhs_V_all)  # (n_genes, K)
        raw_V = rng.standard_normal((n_genes, K))
        # noise = L^{-T} z so cov = L^{-T} L^{-1} = Lam^{-1} (transpose the last
        # two axes of L_inv: "jlk" applies L^{-T}, not L^{-1}).
        V = mus_V + np.einsum("jlk,jl->jk", L_inv_per_gene, raw_V)

        # --- Log-likelihood (probit) on observed entries ---
        p = ndtr((U @ V.T)[mask])
        p = np.clip(p, 1e-10, 1 - 1e-10)
        y_obs = Y[mask]
        ll = float(np.sum(y_obs * np.log(p) + (1 - y_obs) * np.log(1 - p)))
        log_liks.append(ll)

        # --- Store samples ---
        if it >= burn_in and (it - burn_in) % thin == 0:
            U_samples.append(U.copy())
            V_samples.append(V.copy())

        phase = "burn-in" if it < burn_in else "sampling"
        pbar.set_postfix(phase=phase, ll=f"{ll:.0f}", samples=len(U_samples))

    pbar.close()
    return BPMFResult(
        U_samples=np.array(U_samples),
        V_samples=np.array(V_samples),
        log_lik=np.array(log_liks),
        gene_names=[],
        screen_names=[],
        K=K,
        sigma_u=sigma_u,
        sigma_v=sigma_v,
    )


def build_hit_matrix(
    screens: list,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the binary hit matrix from a list of ScreenRecords.

    Returns:
        Y: (n_screens, n_genes) binary matrix
        screen_names: list of screen dataset_names
        gene_names: sorted union of all genes across screens
    """
    gene_set: set[str] = set()
    for s in screens:
        gene_set.update(s.genes)
    gene_names = sorted(gene_set)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    n_screens = len(screens)
    n_genes = len(gene_names)
    Y = np.zeros((n_screens, n_genes), dtype=np.float64)
    mask = np.zeros((n_screens, n_genes), dtype=bool)

    screen_names = []
    for i, s in enumerate(screens):
        screen_names.append(s.dataset_name)
        for gene, hit in zip(s.genes, s.hits):
            j = gene_to_idx[gene]
            Y[i, j] = 1.0 if hit else 0.0
            mask[i, j] = True

    return Y, mask, screen_names, gene_names


__all__ = ["BPMFResult", "gibbs_bpmf", "build_hit_matrix"]
