"""Effective Pathways (EP-B / EP-S / EP-D) must count genes, not annotations.

The reported statistic assigns each gene exactly *one* of its pathways, drawn
uniformly and averaged over draws, then rarefies to a fixed reference count.
The tempting alternative -- the fractional-weight plug-in, ``1/len(pathways)``
spread over all of a gene's pathways -- is the historical sunburst statistic and
is *not* interchangeable: it gives a bag of ``m`` genes a support ceiling of
``sum(k)`` rather than ``m``, so a method that happens to pick densely annotated
genes outscores one that picks a genuinely broader set. These tests pin the
ceiling, the rarefaction bookkeeping, and the ordering that follows from it.

Everything here uses a synthetic membership passed in explicitly, so no Reactome
GMT fetch is needed.
"""

from __future__ import annotations

import math

import pytest

from assayloop.metrics.effective_pathways import (
    M_BATCH,
    RETENTION,
    effective_n,
    effective_pathways,
    pathway_weights,
)


# ---------------------------------------------------------------------------
# Synthetic corpora
# ---------------------------------------------------------------------------

def _membership(n_genes: int, k: int, n_paths: int) -> dict[str, tuple[str, ...]]:
    """``n_genes`` genes, each in ``k`` pathways cycled out of ``n_paths``."""
    return {
        f"G{i}": tuple(f"P{(i * k + j) % n_paths}" for j in range(k))
        for i in range(n_genes)
    }


def _batches(genes, n_screens: int, n_batches: int, size: int):
    """Deal ``genes`` round-robin into ``list[screen][batch] -> list[gene]``."""
    it = iter(genes)
    return [
        [[next(it) for _ in range(size)] for _ in range(n_batches)]
        for _ in range(n_screens)
    ]


# ---------------------------------------------------------------------------
# effective_n
# ---------------------------------------------------------------------------

def test_effective_n_uniform_is_the_category_count():
    assert effective_n([1.0] * 7) == pytest.approx(7.0)


def test_effective_n_point_mass_is_one():
    assert effective_n([3.0]) == pytest.approx(1.0)
    assert effective_n([5.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_effective_n_is_scale_invariant():
    assert effective_n([2.0, 6.0]) == pytest.approx(effective_n([1.0, 3.0]))


def test_effective_n_empty_is_nan():
    assert math.isnan(effective_n([]))
    assert math.isnan(effective_n([0.0, 0.0]))


# ---------------------------------------------------------------------------
# The ceiling: EP-B counts genes, the plug-in counts annotations
# ---------------------------------------------------------------------------

def test_ep_batch_is_capped_by_the_reference_gene_count():
    """Densely annotated genes must not buy a batch more than M_BATCH pathways."""
    mem = _membership(n_genes=600, k=8, n_paths=4000)
    sb = _batches(list(mem), n_screens=4, n_batches=3, size=50)

    out = effective_pathways(sb, membership=mem)

    assert out["ep_batch"] <= M_BATCH + 1e-9
    # Every gene sits in 8 disjoint-ish pathways, so the fractional plug-in
    # sails past the ceiling the reported statistic respects.
    assert out["ep_batch_raw"] > M_BATCH


def test_unrarefied_batch_scope_is_the_plug_in():
    mem = _membership(n_genes=600, k=8, n_paths=4000)
    sb = _batches(list(mem), n_screens=4, n_batches=3, size=50)

    rare = effective_pathways(sb, membership=mem)
    plug = effective_pathways(sb, membership=mem, rarefy=False)

    assert plug["ep_batch"] == pytest.approx(rare["ep_batch_raw"])
    assert plug["ep_batch"] != pytest.approx(rare["ep_batch"])


# ---------------------------------------------------------------------------
# Ordering: broad picks beat narrow picks
# ---------------------------------------------------------------------------

def test_broad_picks_score_above_narrow_picks():
    """A method spread across many pathways must outscore a concentrated one."""
    broad = {f"G{i}": (f"P{i}",) for i in range(600)}
    narrow = {f"G{i}": (f"P{i % 3}",) for i in range(600)}
    sb = _batches([f"G{i}" for i in range(600)], n_screens=4, n_batches=3, size=50)

    ep_broad = effective_pathways(sb, membership=broad)["ep_batch"]
    ep_narrow = effective_pathways(sb, membership=narrow)["ep_batch"]

    assert ep_broad == pytest.approx(M_BATCH, rel=1e-6)  # every pick is unique
    assert ep_narrow < 3.01                              # only 3 pathways exist
    assert ep_broad > ep_narrow


# ---------------------------------------------------------------------------
# Rarefaction bookkeeping: drop, count, and refuse to report
# ---------------------------------------------------------------------------

def test_units_too_small_to_rarefy_are_dropped_and_counted():
    """Batches below the reference count are excluded, not silently compared."""
    mem = {f"G{i}": (f"P{i}",) for i in range(400)}
    small = M_BATCH - 5
    sb = _batches(list(mem), n_screens=4, n_batches=2, size=small)

    out = effective_pathways(sb, membership=mem)

    assert out["ep_n_batches"] == 8
    assert out["ep_n_dropped"] >= 8
    assert out["ep_batch_retention"] == 0.0
    # Nothing survived the filter, so there is no batch-scope number to print.
    assert out["ep_batch"] is None
    # The picks are still real genes, so they still feed the pooled scopes.
    assert out["ep_n_picks"] == 8 * small
    assert out["ep_dataset_raw"] is not None


def test_very_low_retention_suppresses_the_mean():
    """Below RETENTION, too few surviving units produce a reportable mean."""
    mem = {f"G{i}": (f"P{i}",) for i in range(400)}
    # One fifth of the batches can supply M_BATCH annotated genes. Every batch
    # is full by pick count, so only annotation depth separates them.
    unannotated = [f"X{i}" for i in range(1_000)]
    sb = []
    for s in range(4):
        rich = [f"G{s * 50 + i}" for i in range(50)]
        poor = []
        for b in range(4):
            offset = (s * 4 + b) * 45
            poor.append(
                [f"G{200 + s * 20 + b * 5 + i}" for i in range(5)]
                + unannotated[offset:offset + 45]
            )
        sb.append([rich, *poor])

    out = effective_pathways(sb, membership=mem)

    assert out["ep_batch_retention"] == pytest.approx(0.2)
    assert out["ep_batch_retention"] < RETENTION
    assert out["ep_batch"] is None
    # The drop count spans every scope: the 16 poor batches, plus the 4 screens
    # and the pooled dataset, none of which reach their own reference counts on
    # a corpus this small.
    assert out["ep_n_dropped"] == 21
    assert out["ep_batch_raw"] is not None


def test_annotated_fraction_tracks_unannotated_picks():
    mem = {f"G{i}": (f"P{i}",) for i in range(100)}
    sb = [[[f"G{i}" for i in range(40)] + [f"X{i}" for i in range(10)]]]

    out = effective_pathways(sb, membership=mem)

    assert out["ep_n_picks"] == 50
    assert out["ep_n_annotated"] == 40
    assert out["ep_annotated_frac"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_gives_the_same_numbers():
    mem = _membership(n_genes=600, k=4, n_paths=500)
    sb = _batches(list(mem), n_screens=3, n_batches=4, size=50)

    a = effective_pathways(sb, membership=mem, seed=0)
    b = effective_pathways(sb, membership=mem, seed=0)

    assert a == b


def test_a_different_seed_moves_the_estimate_only_within_noise():
    mem = _membership(n_genes=600, k=4, n_paths=500)
    sb = _batches(list(mem), n_screens=3, n_batches=4, size=50)

    a = effective_pathways(sb, membership=mem, seed=0)["ep_batch"]
    b = effective_pathways(sb, membership=mem, seed=17)["ep_batch"]

    # R_BATCH is set so the seed-to-seed spread sits well under the printed
    # precision of one decimal; anything larger means the draw count regressed.
    assert abs(a - b) < 0.05


# ---------------------------------------------------------------------------
# pathway_weights
# ---------------------------------------------------------------------------

def test_pathway_weights_split_each_gene_across_its_pathways():
    mem = {"A": ("P1", "P2"), "B": ("P2",)}

    weights, n_ann = pathway_weights(["a", "B", "MISSING"], membership=mem)

    assert n_ann == 2
    assert weights == pytest.approx({"P1": 0.5, "P2": 1.5})
    assert sum(weights.values()) == pytest.approx(n_ann)


def test_pathway_weights_ignores_unannotated_genes():
    weights, n_ann = pathway_weights(["NOPE"], membership={"A": ("P1",)})

    assert weights == {}
    assert n_ann == 0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_no_picks_reports_nothing_rather_than_zero():
    out = effective_pathways([], membership={"A": ("P1",)})

    assert out["ep_n_picks"] == 0
    assert out["ep_batch"] is None
    assert out["ep_screen"] is None
    assert out["ep_dataset"] is None
    assert out["ep_annotated_frac"] is None
