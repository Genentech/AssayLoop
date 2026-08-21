"""Regression tests for the BPMF Gibbs sampler's two conditional draws.

Both draws were wrong at some point during development and both failures are
invisible in a spot check of the output -- the samples look like plausible
Gaussians either way. These tests pin the two moments that actually distinguish
correct from incorrect:

* :func:`_sample_truncnorm` must respect the sign implied by ``y`` and land on
  the analytic truncated-normal mean (an earlier version used ``Phi(mean)``
  instead of ``Phi(-mean)`` for the ``y=0`` bound and returned the draw in
  standardized space, dropping the ``+ mean``).
* :func:`_sample_u` must have covariance ``Lambda^{-1}`` and not its transpose-
  flipped cousin ``L^{-1} L^{-T}``, which has the *same eigenvalues* -- so any
  test that only checks the spread passes on the buggy code.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from assayloop.models.bpmf_model import _sample_truncnorm, _sample_u


def _random_design(rng, n_obs=40, K=6):
    """A V_obs whose posterior precision is well conditioned but not diagonal."""
    return rng.standard_normal((n_obs, K))


# --------------------------------------------------------------------------- #
# u | z  --  covariance must be Lambda^{-1}, orientation included.
# --------------------------------------------------------------------------- #
def test_u_draws_have_the_posterior_mean_and_covariance():
    rng = np.random.default_rng(0)
    K, precision_u = 6, 1.0
    V_obs = _random_design(rng, K=K)
    z_obs = rng.standard_normal(V_obs.shape[0])

    Lambda = V_obs.T @ V_obs + precision_u * np.eye(K)
    mu = np.linalg.solve(Lambda, V_obs.T @ z_obs)
    cov = np.linalg.inv(Lambda)

    draws = np.array([_sample_u(V_obs, z_obs, precision_u, rng)
                      for _ in range(200_000)])

    # Monte-Carlo error on the mean is ~sqrt(diag(cov)/n); 5 sigma is generous.
    tol = 5.0 * np.sqrt(np.diag(cov) / len(draws))
    assert np.all(np.abs(draws.mean(axis=0) - mu) < tol)

    emp = np.cov(draws, rowvar=False)
    rel = np.linalg.norm(emp - cov) / np.linalg.norm(cov)
    assert rel < 0.02, f"empirical covariance is off by {rel:.3f} relative"


def test_transposed_cholesky_solve_is_the_bug_this_test_catches():
    """The buggy noise term matches on eigenvalues but not on covariance.

    Guards the test above from being vacuous: it must be tight enough to
    separate ``L^{-T} xi`` (correct) from ``L^{-1} xi`` (the historical bug),
    which is only possible because the check is on the full matrix rather than
    on its spectrum.
    """
    rng = np.random.default_rng(1)
    K = 6
    V_obs = _random_design(rng, K=K)
    Lambda = V_obs.T @ V_obs + np.eye(K)
    L = np.linalg.cholesky(Lambda)

    good = np.linalg.inv(Lambda)                      # L^{-T} L^{-1}
    L_inv = np.linalg.inv(L)
    bad = L_inv @ L_inv.T                             # L^{-1} L^{-T}

    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(good)),
                               np.sort(np.linalg.eigvalsh(bad)), rtol=1e-10)
    rel = np.linalg.norm(bad - good) / np.linalg.norm(good)
    assert rel > 0.05, "bug is too subtle for the covariance tolerance above"


def test_u_is_pulled_toward_the_least_squares_fit():
    """Sanity check on the mean: more data -> u concentrates on the truth."""
    rng = np.random.default_rng(2)
    K = 4
    u_true = rng.standard_normal(K)
    V_obs = _random_design(rng, n_obs=4000, K=K)
    z_obs = V_obs @ u_true + rng.standard_normal(4000)

    draws = np.array([_sample_u(V_obs, z_obs, 1.0, rng) for _ in range(200)])
    assert np.linalg.norm(draws.mean(axis=0) - u_true) < 0.1


# --------------------------------------------------------------------------- #
# z | u  --  truncation side and mean.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mean", [-3.0, -0.5, 0.0, 0.5, 3.0])
def test_truncnorm_respects_the_sign_implied_by_y(mean):
    rng = np.random.default_rng(3)
    m = np.full(5000, mean)
    assert np.all(_sample_truncnorm(m, np.ones(5000), rng) > 0.0)
    assert np.all(_sample_truncnorm(m, np.zeros(5000), rng) < 0.0)


@pytest.mark.parametrize("mean", [-2.0, -0.5, 0.0, 1.0, 2.5])
@pytest.mark.parametrize("y", [0.0, 1.0])
def test_truncnorm_matches_the_analytic_truncated_mean(mean, y):
    """E[Z] for N(mean,1) truncated at 0.

    y=1 -> mean + phi(-mean)/(1-Phi(-mean));  y=0 -> mean - phi(-mean)/Phi(-mean).
    A sampler that forgot the ``+ mean`` shift, or that used ``Phi(mean)`` as the
    y=0 bound, misses this for every ``mean != 0``.
    """
    rng = np.random.default_rng(4)
    n = 400_000
    draws = _sample_truncnorm(np.full(n, mean), np.full(n, y), rng)

    phi = np.exp(-0.5 * mean ** 2) / np.sqrt(2.0 * np.pi)
    tail = ndtr(-mean)
    expected = mean + phi / (1.0 - tail) if y == 1.0 else mean - phi / tail

    # Truncated-normal variance is <= 1, so the standard error is <= 1/sqrt(n).
    assert abs(draws.mean() - expected) < 5.0 / np.sqrt(n)
