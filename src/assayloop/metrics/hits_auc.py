"""Per-step hit recovery: area under the cumulative-hits curve.

This is the loop's *training-time* trajectory signal, recorded at every step
and used as the RL reward. It is not the paper's reported nAUC -- that is
:func:`assaybench.benchmark.sequential.adjusted_nauc`, which is computed once
per trajectory on the effective-budget axis (forgiven picks advance neither
axis) and sampled once per round rather than once per gene. The two answer
different questions and are deliberately not unified; the paper's tables come
from the assaybench function.

X axis = fraction of library the policy was *budgeted* to acquire
    (``n_requested / library_size``, i.e. how many genes it should have
    sampled = cumulative batch size). When a policy under-supplies (returns
    fewer valid genes than requested), the unfilled budget slots find no
    hits, so the curve is extended flat to the budget point. This keeps two
    policies that requested the same number of genes comparable at the same x
    and penalises under-supply, instead of rewarding it via a smaller
    (inflated) random denominator. Falls back to ``n_acquired`` when the loop
    doesn't pass a requested budget (legacy callers).
Y axis = fraction of hits acquired (hits_so_far / total_hits).

Trapezoidal integration over the inner-loop trajectory.

Inputs needed from the ground truth (a `ScreenRecord`):

- ``len(screen.genes)`` -> library_size
- ``screen.total_hits``  -> total_hits

The metric is stateless: it reconstructs the trajectory from the
cumulative ``observations`` list (which the inner loop appends to in
acquisition order). Each Observation's label must contain ``"hit"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assaybench.benchmark.sequential import enrichment_factor_from_value
from assaybench.core.metric import Metric
from assaybench.core.types import ModelPrediction, Observation


def _hit_flag(obs: Observation) -> int:
    """Extract a 0/1 hit flag from an Observation.

    Supports labels shaped as:
      - dict with ``"hit"`` key (what the screen task emits);
      - bool/int (treated directly);
      - dict with no ``"hit"`` -> 0.
    """
    label = obs.label
    if isinstance(label, dict):
        return int(bool(label.get("hit", False)))
    if isinstance(label, (bool, int)):
        return int(bool(label))
    return 0


def _library_size_and_total_hits(ground_truth: Any) -> tuple[int, int]:
    """Best-effort extraction. Works for a ScreenRecord and
    falls back to ``ground_truth.total_hits`` / ``len(ground_truth.genes)``
    style attributes."""
    library_size = 0
    total_hits = 0

    # ScreenRecord
    genes = getattr(ground_truth, "genes", None)
    if genes is not None:
        library_size = len(genes)
    elif isinstance(ground_truth, dict):
        library_size = int(
            ground_truth.get("num_genes")
            or len(ground_truth.get("relevance_genes") or [])
            or 0
        )

    total_hits_attr = getattr(ground_truth, "total_hits", None)
    if isinstance(total_hits_attr, int):
        total_hits = total_hits_attr
    elif callable(total_hits_attr):
        try:
            total_hits = int(total_hits_attr())
        except Exception:
            pass
    elif isinstance(ground_truth, dict):
        hits = ground_truth.get("hit") or []
        total_hits = int(sum(1 for h in hits if h))

    return library_size, total_hits


def best_possible_auc(frac_acquired: float, total_hits: int, library_size: int) -> float:
    """AUC of the optimal acquisition curve at ``x = frac_acquired``.

    The optimal acquisition picks every hit before any miss, so the
    cumulative-hits curve climbs at slope ``L/H`` until ``x = H/L`` (all
    hits found), then flat at ``y = 1``.

    Returns the trapezoidal area under that piecewise-linear curve from
    ``x=0`` to ``x=frac_acquired``, which is the denominator used by
    ``hits_auc_normalized``.
    """
    if library_size <= 0 or total_hits <= 0 or frac_acquired <= 0:
        return 0.0
    hit_density = total_hits / library_size  # H/L; turnover point in x.
    x = float(frac_acquired)
    if x <= hit_density:
        # Still on the initial ramp y = x / hit_density.
        return (x * x) / (2.0 * hit_density)
    # Past the corner: triangle up to (H/L, 1) plus a rectangle from there.
    return hit_density / 2.0 + (x - hit_density)


def random_expected_auc(frac_acquired: float) -> float:
    """Expected AUC of a uniformly-random acquisition at ``x = frac_acquired``.

    Random picks n_acquired = x · L genes uniformly, so in expectation
    cumulative-hits = x · H and the curve is y(x') = x' for x' ∈ [0, x].
    Its trapezoidal AUC is x² / 2 -- the diagonal triangle below y = x.
    """
    if frac_acquired <= 0:
        return 0.0
    x = float(frac_acquired)
    return (x * x) / 2.0


def random_expected_n_hits(frac_acquired: float, total_hits: int) -> float:
    """Expected hits a uniformly-random acquisition has found by ``x``.

    Random samples a fraction ``x`` of the library; in expectation it
    captures the same fraction of the total hits.
    """
    if frac_acquired <= 0 or total_hits <= 0:
        return 0.0
    return float(frac_acquired) * float(total_hits)


def n_hits_vs_random_value(n_hits: float, frac_budget: float, total_hits: int) -> float:
    """Episodic enrichment ``n_hits / (frac_budget * total_hits)``.

    Mirrors the ``n_hits_vs_random`` field of :class:`HitsAUC` so the RL trainer
    can compute the same trajectory-level reward/metric off the AL loop.
    """
    denom = random_expected_n_hits(frac_budget, total_hits)
    return float(n_hits / denom) if denom > 0 else 0.0


def adjusted_nvr_value(
    n_hits: float,
    n_in_lib: int,
    n_out_lib_in_universe: int,
    domain_size: int,
    total_hits: int,
    budget: int,
) -> float:
    """Domain-adjusted NVR -- the paper's EF, from counts this repo already has.

    This is a thin argument adapter over
    :func:`assaybench.benchmark.sequential.enrichment_factor_from_value`, which
    is where the arithmetic lives. It exists because the callers here hold
    *counts* pulled out of a persisted ``result.json`` or an in-memory rollout,
    not the gene lists that :func:`~assaybench.benchmark.sequential.enrichment_factor`
    wants.

    Picks are classified into ``n_in_lib`` (in the screen's library),
    ``n_out_lib_in_universe`` (real genes outside this screen's library but in
    the candidate universe -- FORGIVEN), and ``n3 = budget - n_in_lib -
    n_out_lib_in_universe`` (out-of-universe or unfilled -- charged). The
    effective budget is ``eff = n_in_lib + n3``, which is algebraically
    ``budget - n_out_lib_in_universe`` and identical to
    :attr:`~assaybench.benchmark.sequential.AcquisitionCounts.effective_budget`.

    Both :func:`assayloop.scripts.full_genome_table._adj_nvr_from_run` and the
    RL evaluator call this, so the paper's headline number has one definition
    across the persisted-results path and the training path.
    """
    n3 = budget - n_in_lib - n_out_lib_in_universe
    return enrichment_factor_from_value(
        n_hits=n_hits,
        effective_budget=n_in_lib + n3,
        library_size=domain_size,
        total_hits=total_hits,
    )


class HitsAUC(Metric):
    """Per-step hit recovery. Returns per-call:

    - ``n_hits``        : cumulative hits acquired so far
    - ``total_hits``    : ground-truth total hits in the screen
    - ``frac_hits``     : cumulative-hits / total-hits
    - ``frac_acquired`` : n_acquired / library_size
    - ``hits_auc``      : trapezoidal AUC of the curve so far on [0,1]²
                          (i.e. the actual filled area; bounded above
                          by ``frac_acquired``).
    - ``hits_auc_best`` : AUC of the optimal-ordering curve clipped to the
                          same ``frac_acquired``. The denominator used by
                          ``hits_auc_normalized``.
    - ``hits_auc_random``    : Expected AUC of a uniformly-random
                          acquisition at the same ``frac_acquired``
                          (= ``x² / 2``). The denominator used by
                          ``hits_auc_vs_random``.
    - ``hits_auc_normalized`` : ``hits_auc / hits_auc_best``. ``1.0`` = perfect
                          ordering at the current budget; random expectation
                          = ``H/L``.
    - ``hits_auc_vs_random`` : ``hits_auc / hits_auc_random``. ``1.0`` = no
                          better than random, ``> 1`` = above random,
                          ``< 1`` = below random. Best-possible at small ``x``
                          is roughly ``L/H``.
    - ``n_hits_random``  : Expected hits under random acquisition at the
                          same ``frac_acquired`` (= ``x · H``). The
                          denominator used by ``n_hits_vs_random``.
    - ``n_hits_vs_random`` : ``n_hits / n_hits_random``. ``1.0`` = matches
                          the random hit count, ``> 1`` = enriched over
                          random, ``< 1`` = depleted.
    """

    def name(self) -> str:
        return "hits_auc"

    def score(
        self,
        observations: list[Observation],
        model_prediction: ModelPrediction | None,
        ground_truth: Any,
        candidates_remaining: list[Any],
        *,
        new_observations: list[Observation] | None = None,
        acquired_batch: list[Any] | None = None,
        n_requested: int | None = None,
    ) -> dict[str, float]:
        library_size, total_hits = _library_size_and_total_hits(ground_truth)
        # When candidates are expanded beyond the screen library (e.g.
        # full-genome evaluation), use the actual pool size so the random
        # baseline reflects the larger haystack.
        pool_size = len(observations) + len(candidates_remaining)
        if pool_size > library_size:
            library_size = pool_size
        if library_size <= 0 or total_hits <= 0:
            return {
                "hits_auc": 0.0,
                "hits_auc_best": 0.0,
                "hits_auc_random": 0.0,
                "hits_auc_normalized": 0.0,
                "hits_auc_vs_random": 0.0,
                "n_hits": 0.0,
                "n_hits_random": 0.0,
                "n_hits_vs_random": 0.0,
                "frac_hits": 0.0,
                "frac_acquired": 0.0,
                "frac_budget": 0.0,
            }

        n_acquired = len(observations)
        if n_acquired == 0:
            return {
                "n_hits": 0.0,
                "n_hits_random": 0.0,
                "n_hits_vs_random": 0.0,
                "total_hits": float(total_hits),
                "frac_hits": 0.0,
                "frac_acquired": 0.0,
                "frac_budget": 0.0,
                "hits_auc": 0.0,
                "hits_auc_best": 0.0,
                "hits_auc_random": 0.0,
                "hits_auc_normalized": 0.0,
                "hits_auc_vs_random": 0.0,
            }

        # Budget basis (x-axis): how many genes the policy *should* have
        # sampled = cumulative requested batch size. Falls back to the actual
        # count when the loop doesn't supply it (legacy callers). The budget
        # is at least ``n_acquired`` (you can't request fewer than you took)
        # and capped at the library size.
        budget = n_acquired if n_requested is None else int(n_requested)
        budget = min(max(budget, n_acquired), library_size)

        hit_flags = np.fromiter((_hit_flag(o) for o in observations), dtype=np.int32, count=n_acquired)
        cum_hits = np.cumsum(hit_flags)

        # Trajectory on the *budget* x-axis: prepend (0, 0); one point per
        # acquired candidate at (i/L, cum_hits[i]/H). When the policy
        # under-supplied (budget > n_acquired), the unfilled budget slots
        # found no hits, so the curve is extended FLAT to (budget/L, frac_hits).
        # This penalises under-supply instead of rewarding it (a shorter
        # curve no longer gets a smaller — and thus inflated — random
        # denominator).
        frac_hits = float(cum_hits[-1] / total_hits)
        xs_list = [0.0, *(np.arange(1, n_acquired + 1) / library_size)]
        ys_list = [0.0, *(cum_hits / total_hits)]
        if budget > n_acquired:
            xs_list.append(budget / library_size)
            ys_list.append(frac_hits)
        xs = np.asarray(xs_list, dtype=float)
        ys = np.asarray(ys_list, dtype=float)
        auc = float(np.trapezoid(ys, xs))

        frac_budget = float(budget / library_size)
        frac_acq = float(n_acquired / library_size)
        n_hits = float(cum_hits[-1])

        # Baselines are evaluated at the *budget* fraction so two policies
        # that requested the same number of genes are compared at the same x.
        best = best_possible_auc(frac_budget, total_hits, library_size)
        rand_auc = random_expected_auc(frac_budget)
        rand_n_hits = random_expected_n_hits(frac_budget, total_hits)

        return {
            "n_hits": n_hits,
            "n_hits_random": rand_n_hits,
            "n_hits_vs_random": float(n_hits / rand_n_hits) if rand_n_hits > 0 else 0.0,
            "total_hits": float(total_hits),
            "frac_hits": frac_hits,
            "frac_acquired": frac_acq,
            "frac_budget": frac_budget,
            "hits_auc": auc,
            "hits_auc_best": float(best),
            "hits_auc_random": float(rand_auc),
            "hits_auc_normalized": float(auc / best) if best > 0 else 0.0,
            "hits_auc_vs_random": float(auc / rand_auc) if rand_auc > 0 else 0.0,
        }


__all__ = [
    "HitsAUC",
    "best_possible_auc",
    "random_expected_auc",
    "random_expected_n_hits",
    "n_hits_vs_random_value",
    "adjusted_nvr_value",
]
