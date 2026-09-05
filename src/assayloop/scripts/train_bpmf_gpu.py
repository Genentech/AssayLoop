#!/usr/bin/env python3
"""GPU (PyTorch) port of the BPMF Gibbs sampler, with a hyperparameter sweep.

A faithful, much faster reimplementation of the CPU Albert-Chib probit-MF Gibbs
sampler in :mod:`assayloop.models.bayesian_pmf` (which is left untouched). It
emits the same pickled ``BPMFResult`` so the output is drop-in for
``--init-gene-factors bpmf:<pkl>``, and adds:

  * a grid sweep over ``K`` x ``sigma_u`` x ``sigma_v`` (build the hit matrix
    once, fit each config on GPU),
  * a cheap held-out completion metric (probit LL + AUC on masked-out observed
    entries) to rank configs before paying for a full ranker run, and
  * a ``--validate`` / ``--validate-cpu`` mode that checks the port matches the
    current implementation (RNG-free linear-algebra equivalence + statistical
    parity vs an existing pkl + optional tiny CPU<->GPU run).

Usage::

    # single config
    uv run python -m assayloop.scripts.train_bpmf_gpu --target-set train \
        --K 10 --sigma-u 1 --sigma-v 1 --n-iter 2000 --burn-in 1000

    # sweep
    uv run python -m assayloop.scripts.train_bpmf_gpu --target-set train \
        --K 8,10,16 --sigma-u 0.5,1,2 --sigma-v 0.5,1,2

    # validate against the current implementation
    uv run python -m assayloop.scripts.train_bpmf_gpu --validate \
        --ref-pkl output/bpmf/bpmf_public_train_K15_.../bpmf_result.pkl
    uv run python -m assayloop.scripts.train_bpmf_gpu --validate-cpu \
        --subsample-screens 100 --subsample-genes 500 --n-iter 300 --burn-in 150
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from assayloop import config
from assayloop.models.bayesian_pmf import BPMFResult, build_hit_matrix


# --------------------------------------------------------------------------- #
# Probit link helpers (standard normal CDF / inverse CDF), torch.
# --------------------------------------------------------------------------- #
_SQRT2 = float(np.sqrt(2.0))


def _ndtr(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.special, "ndtr"):
        return torch.special.ndtr(x)
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(p)
    return _SQRT2 * torch.erfinv(2.0 * p - 1.0)


def _trunc_normal(mean: torch.Tensor, y: torch.Tensor,
                  gen: torch.Generator) -> torch.Tensor:
    """Sample Z ~ N(mean, 1) truncated to (0, inf) if y==1 else (-inf, 0).

    Inverse-CDF method, mirroring ``bayesian_pmf._fast_truncnorm``. For
    N(mean, 1) truncated at Z=0 the standardized truncation point is
    ``Phi(-mean)``: y=1 draws ``p ~ U(Phi(-mean), 1)``, y=0 draws
    ``p ~ U(0, Phi(-mean))``, and the mean is added back to ``Phi^{-1}(p)``.

    The clamp epsilon must be dtype-aware: in float32 ``1 - 1e-8`` rounds to
    exactly 1.0, so ndtri would return +inf and blow up the chain. Use 1e-8 for
    float64 (matches the CPU sampler) and 1e-6 for float32.
    """
    eps = 1e-8 if mean.dtype == torch.float64 else 1e-6
    u = torch.rand(mean.shape, generator=gen, device=mean.device, dtype=mean.dtype)
    pos = y > 0.5
    bound = _ndtr(-mean)                 # Phi(-mean): standardized truncation point
    lo = torch.where(pos, bound, torch.zeros_like(mean))
    hi = torch.where(pos, torch.ones_like(mean), bound)
    lo = lo.clamp(eps, 1.0 - eps)
    hi = hi.clamp(eps, 1.0 - eps)
    p = (lo + u * (hi - lo)).clamp(eps, 1.0 - eps)
    return mean + _ndtri(p)


# --------------------------------------------------------------------------- #
# Conjugate-normal factor update (vectorized per item; no Python group loop).
#
# This computes, for every screen i (or gene j), the SAME natural parameters as
# the CPU sampler's missing-pattern "complement trick":
#   Lam_i = sum_{j observed} V_j V_j^T + I/sigma^2     (== VtV_full - V_miss^T V_miss)
#   rhs_i = sum_{j observed} Z_ij V_j
# but as two dense GEMMs + a batched Cholesky, which is what makes GPU fast.
# --------------------------------------------------------------------------- #
def _natural_params(other: torch.Tensor, ZM: torch.Tensor, mask_f: torch.Tensor,
                    precision: float, *, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Lam, rhs) for the factor on ``axis``.

    axis=0 -> updating screen factors U (other = V, genes on dim 1).
    axis=1 -> updating gene  factors V (other = U, screens on dim 0).
    """
    K = other.shape[1]
    outer = (other.unsqueeze(2) * other.unsqueeze(1)).reshape(other.shape[0], K * K)
    if axis == 0:  # U[i] uses genes: Lam_i = sum_j mask[i,j] V_j V_j^T
        lam = (mask_f @ outer).reshape(-1, K, K)        # (n_screens, K, K)
        rhs = ZM @ other                                # (n_screens, K)
    else:          # V[j] uses screens: Lam_j = sum_i mask[i,j] U_i U_i^T
        lam = (mask_f.t() @ outer).reshape(-1, K, K)    # (n_genes, K, K)
        rhs = ZM.t() @ other                            # (n_genes, K)
    eye = torch.eye(K, device=other.device, dtype=other.dtype)
    lam = lam + precision * eye
    return lam, rhs


def _sample_factor(lam: torch.Tensor, rhs: torch.Tensor,
                   gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x ~ N(Lam^{-1} rhs, Lam^{-1}) per item; return (sample, mean).

    Matches the CPU sampler: mean = cholesky_solve(rhs, L); noise = L^{-T} z so
    cov(noise) = L^{-T} L^{-1} = Lam^{-1}.
    """
    L = torch.linalg.cholesky(lam)
    mu = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    z = torch.randn(rhs.shape, generator=gen, device=rhs.device, dtype=rhs.dtype)
    noise = torch.linalg.solve_triangular(
        L.transpose(-1, -2), z.unsqueeze(-1), upper=True).squeeze(-1)
    return mu + noise, mu


# --------------------------------------------------------------------------- #
# Core GPU Gibbs sampler.
# --------------------------------------------------------------------------- #
def gibbs_bpmf_torch(
    Y: np.ndarray,
    *,
    K: int = 10,
    sigma_u: float = 1.0,
    sigma_v: float = 1.0,
    n_iter: int = 2000,
    burn_in: int = 1000,
    thin: int = 2,
    seed: int = 42,
    mask: np.ndarray | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    n_save_samples: int | None = None,
    progress: bool = True,
) -> BPMFResult:
    """GPU Albert-Chib probit-MF Gibbs sampler (faithful port of gibbs_bpmf)."""
    device = torch.device(device)
    n_screens, n_genes = Y.shape
    if mask is None:
        mask = np.ones_like(Y, dtype=bool)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    Yt = torch.as_tensor(Y, dtype=dtype, device=device)
    mask_f = torch.as_tensor(mask, dtype=dtype, device=device)

    U = torch.randn(n_screens, K, generator=gen, device=device, dtype=dtype) * 0.1
    V = torch.randn(n_genes, K, generator=gen, device=device, dtype=dtype) * 0.1

    precision_u = 1.0 / sigma_u ** 2
    precision_v = 1.0 / sigma_v ** 2

    keep = [it for it in range(n_iter) if it >= burn_in and (it - burn_in) % thin == 0]
    if n_save_samples is not None and len(keep) > n_save_samples:
        sel = np.linspace(0, len(keep) - 1, n_save_samples).round().astype(int)
        keep = [keep[i] for i in np.unique(sel)]
    keep_set = set(keep)

    U_samples: list[np.ndarray] = []
    V_samples: list[np.ndarray] = []
    log_liks: list[float] = []

    rng = range(n_iter)
    if progress:
        try:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc=f"Gibbs(GPU) K={K} su={sigma_u} sv={sigma_v}", unit="it")
        except Exception:  # noqa: BLE001
            pass

    for it in rng:
        # Step 1: latent Z (truncated normals on observed; missing zeroed by mask).
        mean_Z = U @ V.t()
        ZM = _trunc_normal(mean_Z, Yt, gen) * mask_f

        # Step 2: screen factors U.
        lam_u, rhs_u = _natural_params(V, ZM, mask_f, precision_u, axis=0)
        U, _ = _sample_factor(lam_u, rhs_u, gen)

        # Step 3: gene factors V.
        lam_v, rhs_v = _natural_params(U, ZM, mask_f, precision_v, axis=1)
        V, _ = _sample_factor(lam_v, rhs_v, gen)

        # Probit log-likelihood on observed entries. Clamp epsilon is dtype-aware
        # (1-1e-10 rounds to 1.0 in float32 -> log(0)); mask via where so missing
        # entries contribute exactly 0 (avoids 0 * -inf = NaN).
        ll_eps = 1e-10 if dtype == torch.float64 else 1e-7
        M = U @ V.t()
        p = _ndtr(M).clamp(ll_eps, 1.0 - ll_eps)
        term = Yt * torch.log(p) + (1.0 - Yt) * torch.log(1.0 - p)
        ll = float(torch.where(mask_f > 0, term, torch.zeros_like(term)).sum())
        log_liks.append(ll)

        if it in keep_set:
            U_samples.append(U.detach().to("cpu", torch.float32).numpy())
            V_samples.append(V.detach().to("cpu", torch.float32).numpy())

        if progress and hasattr(rng, "set_postfix"):
            rng.set_postfix(ll=f"{ll:.0f}", samples=len(U_samples))

    return BPMFResult(
        U_samples=np.asarray(U_samples),
        V_samples=np.asarray(V_samples),
        log_lik=np.asarray(log_liks),
        gene_names=[],
        screen_names=[],
        K=K,
        sigma_u=sigma_u,
        sigma_v=sigma_v,
    )


# --------------------------------------------------------------------------- #
# Held-out completion metric.
# --------------------------------------------------------------------------- #
def _holdout_split(mask: np.ndarray, frac: float, seed: int):
    """Return (fit_mask, hold_rows, hold_cols) splitting off ``frac`` of observed."""
    if frac <= 0:
        return mask, None, None
    rng = np.random.default_rng(seed)
    flat = np.flatnonzero(mask.ravel())
    n_hold = int(frac * flat.size)
    if n_hold == 0:
        return mask, None, None
    pick = rng.choice(flat, size=n_hold, replace=False)
    rows, cols = np.divmod(pick, mask.shape[1])
    fit_mask = mask.copy()
    fit_mask[rows, cols] = False
    return fit_mask, rows, cols


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Dependency-free ROC-AUC via rank statistic (ties broken by sort order)."""
    n = y.size
    n_pos = float(y.sum())
    n_neg = float(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _holdout_metrics(res: BPMFResult, Y: np.ndarray, rows, cols) -> dict:
    """Probit LL + AUC of P(hit)=ndtr(U_mean.V_mean^T) on held-out entries."""
    if rows is None:
        return {"heldout_ll": None, "heldout_auc": None, "n_heldout": 0}
    U = res.posterior_mean_U().astype(np.float64)
    V = res.posterior_mean_V().astype(np.float64)
    score = np.einsum("ij,ij->i", U[rows], V[cols])
    from scipy.special import ndtr  # local; numpy side of the metric
    p = np.clip(ndtr(score), 1e-10, 1.0 - 1e-10)
    y = Y[rows, cols].astype(np.float64)
    ll = float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    return {"heldout_ll": ll, "heldout_auc": _auc(y, p), "n_heldout": int(y.size)}


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def _tier1_linalg_check(device, dtype) -> bool:
    """RNG-free: GPU grouped-GEMM natural params == textbook per-item reference."""
    print("\n[Tier 1] RNG-free linear-algebra equivalence")
    torch.manual_seed(0)
    n_s, n_g, K = 7, 5, 3
    g = torch.Generator(device=device).manual_seed(0)
    U = torch.randn(n_s, K, generator=g, device=device, dtype=dtype)
    V = torch.randn(n_g, K, generator=g, device=device, dtype=dtype)
    Z = torch.randn(n_s, n_g, generator=g, device=device, dtype=dtype)
    mask = (torch.rand(n_s, n_g, generator=g, device=device) > 0.3).to(dtype)
    ZM = Z * mask
    prec_v, prec_u = 0.5, 2.0

    lam_v, rhs_v = _natural_params(U, ZM, mask, prec_v, axis=1)
    lam_u, rhs_u = _natural_params(V, ZM, mask, prec_u, axis=0)

    Un, Vn, ZMn, Mn = (t.cpu().numpy().astype(np.float64) for t in (U, V, ZM, mask))
    eyeK = np.eye(K)
    ok = True
    # gene side
    for j in range(n_g):
        obs = Mn[:, j] > 0.5
        ref_lam = Un[obs].T @ Un[obs] + prec_v * eyeK
        ref_rhs = Un[obs].T @ ZMn[obs, j]
        ok &= np.allclose(lam_v[j].cpu().numpy(), ref_lam, atol=1e-4)
        ok &= np.allclose(rhs_v[j].cpu().numpy(), ref_rhs, atol=1e-4)
    # screen side
    for i in range(n_s):
        obs = Mn[i] > 0.5
        ref_lam = Vn[obs].T @ Vn[obs] + prec_u * eyeK
        ref_rhs = Vn[obs].T @ ZMn[i, obs]
        ok &= np.allclose(lam_u[i].cpu().numpy(), ref_lam, atol=1e-4)
        ok &= np.allclose(rhs_u[i].cpu().numpy(), ref_rhs, atol=1e-4)

    # sampler covariance: empirical cov of many draws ~ Lam^{-1}.
    big = torch.eye(K, device=device, dtype=dtype).unsqueeze(0) * 1.0 + 0.3
    lam_one = (big @ big.transpose(-1, -2))  # SPD (1,K,K)
    rhs_one = torch.zeros(1, K, device=device, dtype=dtype)
    draws = torch.stack([_sample_factor(lam_one, rhs_one, g)[0][0] for _ in range(20000)])
    emp_cov = np.cov(draws.cpu().numpy().T)
    want_cov = np.linalg.inv(lam_one[0].cpu().numpy())
    cov_ok = np.allclose(emp_cov, want_cov, atol=0.05)
    ok &= cov_ok
    print(f"  natural params match textbook reference: {ok and True}")
    print(f"  sampler covariance ~ Lam^-1 (atol 0.05): {cov_ok}")
    print(f"  Tier 1: {'PASS' if ok else 'FAIL'}")
    return ok


def _aligned_means(res, gene_names, screen_names):
    g_idx = {g: i for i, g in enumerate(res.gene_names)}
    s_idx = {s: i for i, s in enumerate(res.screen_names)}
    genes = [g for g in gene_names if g in g_idx]
    screens = [s for s in screen_names if s in s_idx]
    gi = np.array([g_idx[g] for g in genes])
    si = np.array([s_idx[s] for s in screens])
    return res.posterior_mean_U()[si], res.posterior_mean_V()[gi], genes, screens


def _tier2_parity(ref_pkl: Path, device, dtype, holdout_frac: float,
                  corr_subsample: int = 500_000, n_iter_override: int = 0) -> bool:
    """Statistical parity vs an existing pkl: predicted-prob corr + SV spectrum."""
    print(f"\n[Tier 2] statistical parity vs {ref_pkl}")
    from assayloop.tasks import load_screens

    with open(ref_pkl, "rb") as f:
        ref = pickle.load(f)
    cfg_path = ref_pkl.parent / "config.json"
    target_set = "public_train"
    n_iter, burn_in, thin, seed = 2000, 1000, 2, 42
    if cfg_path.exists():
        c = json.loads(cfg_path.read_text())
        target_set = c.get("target_set", target_set)
        n_iter, burn_in = c.get("n_iter", n_iter), c.get("burn_in", burn_in)
        thin, seed = c.get("thin", thin), c.get("seed", seed)
    K, sigma_u, sigma_v = int(ref.K), float(ref.sigma_u), float(ref.sigma_v)
    if n_iter_override > 0:
        n_iter = n_iter_override
        burn_in = min(burn_in, n_iter // 2)
        print(f"  NOTE: n_iter overridden to {n_iter} (smoke test; expect lower "
              f"corr than a full {ref.U_samples.shape[0]}-sample fit).")
    print(f"  ref config: target={target_set} K={K} su={sigma_u} sv={sigma_v} "
          f"n_iter={n_iter} burn_in={burn_in} thin={thin} seed={seed}")

    Y, mask, screen_names, gene_names = build_hit_matrix(load_screens(target_set=target_set))
    print(f"  rebuilt matrix {Y.shape}; ref names match: "
          f"genes={list(gene_names) == list(ref.gene_names)} "
          f"screens={list(screen_names) == list(ref.screen_names)}")

    t0 = time.time()
    gpu = gibbs_bpmf_torch(Y, K=K, sigma_u=sigma_u, sigma_v=sigma_v, n_iter=n_iter,
                           burn_in=burn_in, thin=thin, seed=seed, mask=mask,
                           device=device, dtype=dtype, progress=True)
    gpu.gene_names, gpu.screen_names = gene_names, screen_names
    print(f"  GPU refit in {time.time() - t0:.1f}s")

    Ur, Vr, genes, screens = _aligned_means(ref, gene_names, screen_names)
    Ug, Vg, _, _ = _aligned_means(gpu, genes, screens)

    # predicted-prob correlation over a random subsample of observed entries.
    g_to_local = {g: i for i, g in enumerate(genes)}
    s_to_local = {s: i for i, s in enumerate(screens)}
    gset = np.array([g in g_to_local for g in gene_names])
    sset = np.array([s in s_to_local for s in screen_names])
    sub = mask & sset[:, None] & gset[None, :]
    flat = np.flatnonzero(sub.ravel())
    rng = np.random.default_rng(0)
    pick = rng.choice(flat, size=min(corr_subsample, flat.size), replace=False)
    rows, cols = np.divmod(pick, mask.shape[1])
    # map global screen/gene rows to aligned-local indices
    gl = np.array([g_to_local[g] for g in gene_names if g in g_to_local])
    sl = np.array([s_to_local[s] for s in screen_names if s in s_to_local])
    glob_g_to_local = -np.ones(len(gene_names), dtype=int)
    glob_g_to_local[np.array([i for i, g in enumerate(gene_names) if g in g_to_local])] = np.arange(len(gl))
    glob_s_to_local = -np.ones(len(screen_names), dtype=int)
    glob_s_to_local[np.array([i for i, s in enumerate(screen_names) if s in s_to_local])] = np.arange(len(sl))
    lr, lc = glob_s_to_local[rows], glob_g_to_local[cols]
    from scipy.special import ndtr
    pr = ndtr(np.einsum("ij,ij->i", Ur[lr], Vr[lc]))
    pg = ndtr(np.einsum("ij,ij->i", Ug[lr], Vg[lc]))
    corr = float(np.corrcoef(pr, pg)[0, 1])

    # rotation-invariant geometry: singular-value spectra of V_mean.
    sv_r = np.linalg.svd(Vr - Vr.mean(0), compute_uv=False)
    sv_g = np.linalg.svd(Vg - Vg.mean(0), compute_uv=False)
    sv_cos = float(sv_r @ sv_g / (np.linalg.norm(sv_r) * np.linalg.norm(sv_g)))

    # held-out LL of both, on the same split.
    fit_mask, hr, hc = _holdout_split(mask, holdout_frac, seed=0)
    m_ref = _holdout_metrics(ref, Y, hr, hc) if hr is not None else {}
    m_gpu = _holdout_metrics(gpu, Y, hr, hc) if hr is not None else {}

    # held-out AUC parity is the decisive metric: it's a property of the fitted
    # model (not rotation/RNG dependent) and should match to MC error. The
    # predicted-prob Pearson corr is a softer diagnostic -- two INDEPENDENT Gibbs
    # chains (NumPy vs torch, float64 vs float32) estimate each entry's posterior
    # mean from a finite number of samples, so ~0.98-0.99 is expected, not 1.0.
    auc_delta = (abs(m_ref["heldout_auc"] - m_gpu["heldout_auc"])
                 if hr is not None else None)
    print(f"  predicted-prob Pearson corr (n={pick.size}): {corr:.4f}")
    print(f"  V singular-value spectrum cosine:            {sv_cos:.4f}")
    if hr is not None:
        print(f"  held-out LL  ref={m_ref['heldout_ll']:.4f}  gpu={m_gpu['heldout_ll']:.4f}")
        print(f"  held-out AUC ref={m_ref['heldout_auc']:.4f}  gpu={m_gpu['heldout_auc']:.4f}  "
              f"(|delta|={auc_delta:.4f})")
    ok = (sv_cos >= 0.99 and corr >= 0.98
          and (auc_delta is None or auc_delta <= 0.005))
    print(f"  Tier 2: {'PASS' if ok else 'WARN (inspect above)'}  "
          f"[criteria: corr>=0.98, sv_cos>=0.99, |AUC delta|<=0.005]")
    return ok


def _tier3_cpu_parity(target_set, device, dtype, n_screens_sub, n_genes_sub,
                      n_iter, burn_in, thin, seed) -> bool:
    """Tiny subsampled CPU<->GPU end-to-end parity (opt-in; minimal CPU)."""
    print("\n[Tier 3] tiny CPU<->GPU parity")
    from assayloop.models.bayesian_pmf import gibbs_bpmf
    from assayloop.tasks import load_screens

    Y, mask, _, _ = build_hit_matrix(load_screens(target_set=target_set))
    rng = np.random.default_rng(0)
    si = rng.choice(Y.shape[0], size=min(n_screens_sub, Y.shape[0]), replace=False)
    gi = rng.choice(Y.shape[1], size=min(n_genes_sub, Y.shape[1]), replace=False)
    Ys, ms = Y[np.ix_(si, gi)], mask[np.ix_(si, gi)]
    keep = ms.any(1)
    Ys, ms = Ys[keep], ms[keep]
    keep_g = ms.any(0)
    Ys, ms = Ys[:, keep_g], ms[:, keep_g]
    print(f"  subsampled matrix {Ys.shape}, observed {100*ms.mean():.1f}%")

    t0 = time.time()
    cpu = gibbs_bpmf(Ys, K=8, sigma_u=1.0, sigma_v=1.0, n_iter=n_iter,
                     burn_in=burn_in, thin=thin, seed=seed, mask=ms)
    t_cpu = time.time() - t0
    # Match the CPU sampler's float64 so this isolates algorithm parity (not
    # float32 rounding); production sweeps can still run float32 for speed.
    t0 = time.time()
    gpu = gibbs_bpmf_torch(Ys, K=8, sigma_u=1.0, sigma_v=1.0, n_iter=n_iter,
                           burn_in=burn_in, thin=thin, seed=seed, mask=ms,
                           device=device, dtype=torch.float64, progress=False)
    t_gpu = time.time() - t0

    from scipy.special import ndtr
    obs = ms
    pr = ndtr((cpu.posterior_mean_U() @ cpu.posterior_mean_V().T)[obs])
    pg = ndtr((gpu.posterior_mean_U() @ gpu.posterior_mean_V().T)[obs])
    corr = float(np.corrcoef(pr, pg)[0, 1])
    ll_cpu = float(cpu.log_lik[burn_in:].mean())
    ll_gpu = float(gpu.log_lik[burn_in:].mean())
    print(f"  CPU {t_cpu:.1f}s  vs  GPU {t_gpu:.1f}s  (speedup {t_cpu/max(t_gpu,1e-9):.1f}x)")
    print(f"  post-burn-in mean LL  cpu={ll_cpu:.1f}  gpu={ll_gpu:.1f}")
    print(f"  predicted-prob Pearson corr (observed): {corr:.4f}")
    rel = abs(ll_cpu - ll_gpu) / max(abs(ll_cpu), 1e-9)
    ok = corr >= 0.98 and rel <= 0.05
    print(f"  Tier 3: {'PASS' if ok else 'WARN (inspect above)'}")
    return ok


# --------------------------------------------------------------------------- #
# CLI / sweep driver.
# --------------------------------------------------------------------------- #
def _floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x != ""]


def _ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x != ""]


def _fmt(x: float) -> str:
    return ("%g" % x).replace(".", "p")


def _save_result(res: BPMFResult, out_dir: Path, cfg: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "bpmf_result.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(res, f)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    np.savetxt(out_dir / "gene_names.txt", res.gene_names, fmt="%s")
    np.savetxt(out_dir / "screen_names.txt", res.screen_names, fmt="%s")
    return pkl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-set", default="train")
    ap.add_argument("--K", default="10", help="Comma list of latent dims to sweep.")
    ap.add_argument("--sigma-u", default="1.0", help="Comma list of screen-prior stds.")
    ap.add_argument("--sigma-v", default="1.0", help="Comma list of gene-prior stds.")
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--n-save-samples", type=int, default=100,
                    help="Cap stored posterior samples (keeps pkls small).")
    ap.add_argument("--eval-holdout-frac", type=float, default=0.05,
                    help="Fraction of observed entries held out for the LL/AUC "
                         "selection metric (the fit uses the remainder). 0 = fit "
                         "on all data and report train LL only.")
    ap.add_argument("--output-root", default=str(config.OUTPUT_PATH / "bpmf"))
    ap.add_argument("--train-size", type=int, default=None,
                    help="Fit BPMF on a subsample of N screens from --target-set, "
                         "using the SAME selection the transformer uses "
                         "(assayloop.amortized.train._subsample). Default: all "
                         "screens. Pair with --subset-seed to match a specific "
                         "transformer run.")
    ap.add_argument("--subset-seed", type=int, default=None,
                    help="Seed for the --train-size screen subsample (must equal "
                         "the transformer run's --seed). Distinct from --seed, "
                         "which only seeds the Gibbs sampler / holdout split.")
    # validation
    ap.add_argument("--validate", action="store_true",
                    help="Run Tier 1 (RNG-free math) + Tier 2 (parity vs --ref-pkl).")
    ap.add_argument("--validate-cpu", action="store_true",
                    help="Also run Tier 3 (tiny subsampled CPU<->GPU parity).")
    ap.add_argument("--ref-pkl", default=None, help="Reference BPMFResult pkl for Tier 2.")
    ap.add_argument("--ref-n-iter", type=int, default=0,
                    help="Override Tier 2 refit n_iter (0 = use the ref's config; "
                         "set small, e.g. 300, for a quick CPU smoke test).")
    ap.add_argument("--subsample-screens", type=int, default=100)
    ap.add_argument("--subsample-genes", type=int, default=500)
    args = ap.parse_args(argv)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    if args.validate or args.validate_cpu:
        results = []
        results.append(("Tier1", _tier1_linalg_check(device, dtype)))
        if args.validate and args.ref_pkl:
            results.append(("Tier2", _tier2_parity(Path(args.ref_pkl), device, dtype,
                                                    args.eval_holdout_frac,
                                                    n_iter_override=args.ref_n_iter)))
        elif args.validate:
            print("\n[Tier 2] skipped: pass --ref-pkl <BPMFResult.pkl> to enable.")
        if args.validate_cpu:
            results.append(("Tier3", _tier3_cpu_parity(
                args.target_set, device, dtype, args.subsample_screens,
                args.subsample_genes, args.n_iter, args.burn_in, args.thin, args.seed)))
        print("\n=== validation summary ===")
        for name, ok in results:
            print(f"  {name}: {'PASS' if ok else 'CHECK'}")
        sys.exit(0 if all(ok for _, ok in results) else 1)

    # ---- sweep ----
    from assayloop.tasks import load_screens

    print(f"Loading screens target_set={args.target_set!r} ...")
    screens = load_screens(target_set=args.target_set)
    subset_tag = ""
    if args.train_size is not None:
        # Reproduce the transformer's EXACT training subset so BPMF embeddings are
        # computed on the same screens the ranker sees (amortized/train.py:530).
        from assayloop.amortized.train import _subsample
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
        n_before = len(screens)
        screens = _subsample(screens, args.train_size, subset_seed)
        print(f"Subsampled {len(screens)}/{n_before} screens "
              f"(train_size={args.train_size}, subset_seed={subset_seed}) to match "
              f"the transformer subset.")
        subset_tag = f"_d{args.train_size}_s{subset_seed}"
    Y, mask, screen_names, gene_names = build_hit_matrix(screens)
    print(f"Hit matrix: {Y.shape[0]} screens x {Y.shape[1]} genes "
          f"({100*mask.mean():.1f}% observed, hit rate {Y[mask].mean():.4f})")

    fit_mask, hr, hc = _holdout_split(mask, args.eval_holdout_frac, seed=args.seed)
    if hr is not None:
        print(f"Held out {hr.size} observed entries for the selection metric.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    grid = [(k, su, sv) for k in _ints(args.K)
            for su in _floats(args.sigma_u) for sv in _floats(args.sigma_v)]
    print(f"Sweep over {len(grid)} config(s) on {device} ({args.dtype}).")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rows = []
    for n, (K, su, sv) in enumerate(grid, 1):
        print(f"\n[{n}/{len(grid)}] K={K} sigma_u={su} sigma_v={sv}")
        t0 = time.time()
        res = gibbs_bpmf_torch(
            Y, K=K, sigma_u=su, sigma_v=sv, n_iter=args.n_iter, burn_in=args.burn_in,
            thin=args.thin, seed=args.seed, mask=fit_mask, device=device, dtype=dtype,
            n_save_samples=args.n_save_samples)
        res.gene_names, res.screen_names = gene_names, screen_names
        dt = time.time() - t0

        metrics = _holdout_metrics(res, Y, hr, hc)
        train_ll = float(res.log_lik[args.burn_in:].mean())
        target_label = Path(args.target_set).stem if "/" in args.target_set else args.target_set
        tag = f"bpmf_{target_label}_K{K}_su{_fmt(su)}_sv{_fmt(sv)}{subset_tag}_{ts}"
        cfg = {
            "target_set": args.target_set, "K": K, "n_iter": args.n_iter,
            "burn_in": args.burn_in, "thin": args.thin, "sigma_u": su, "sigma_v": sv,
            "train_size": args.train_size,
            "subset_seed": (args.subset_seed if args.subset_seed is not None
                            else (args.seed if args.train_size is not None else None)),
            "seed": args.seed, "device": str(device), "dtype": args.dtype,
            "n_screens": len(screen_names), "n_genes": len(gene_names),
            "n_posterior_samples": int(res.V_samples.shape[0]),
            "eval_holdout_frac": args.eval_holdout_frac,
            "observed_frac": float(mask.mean()), "hit_rate": float(Y[mask].mean()),
            "train_ll": train_ll, **metrics, "fit_seconds": dt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        pkl = _save_result(res, out_root / tag, cfg)
        ho = metrics["heldout_ll"]
        au = metrics["heldout_auc"]
        print(f"  done in {dt:.1f}s  train_ll={train_ll:.0f}"
              + (f"  heldout_ll={ho:.4f}  heldout_auc={au:.4f}" if ho is not None else "")
              + f"\n  -> {pkl}")
        rows.append({"K": K, "sigma_u": su, "sigma_v": sv, "train_ll": train_ll,
                     "heldout_ll": ho, "heldout_auc": au, "fit_seconds": round(dt, 1),
                     "path": str(pkl)})

    # ---- summary ----
    (out_root / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out_root / "sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_root / 'sweep_summary.json'}")

    rank = sorted(rows, key=lambda r: (r["heldout_auc"] is None, -(r["heldout_auc"] or 0)))
    print(f"\n{'K':>4} {'sigma_u':>8} {'sigma_v':>8} {'train_ll':>11} "
          f"{'heldout_ll':>11} {'heldout_auc':>12}")
    for r in rank:
        print(f"{r['K']:>4} {r['sigma_u']:>8g} {r['sigma_v']:>8g} {r['train_ll']:>11.0f} "
              f"{(r['heldout_ll'] if r['heldout_ll'] is not None else float('nan')):>11.4f} "
              f"{(r['heldout_auc'] if r['heldout_auc'] is not None else float('nan')):>12.4f}")


if __name__ == "__main__":
    main()
