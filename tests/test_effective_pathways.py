"""Integration tests for AssayLoop's Reactome EP adapter."""

from __future__ import annotations

import importlib

from assaybench.benchmark.effective_pathways import (
    effective_pathways as assaybench_effective_pathways,
)

ep_adapter = importlib.import_module("assayloop.metrics.effective_pathways")


def _fixture():
    membership = {f"G{i}": (f"P{i % 7}",) for i in range(50)}
    screen_batches = [[[f"G{i}" for i in range(50)]]]
    return membership, screen_batches


def test_explicit_membership_delegates_to_assaybench():
    membership, screen_batches = _fixture()

    expected = assaybench_effective_pathways(
        screen_batches, membership=membership
    )
    observed = ep_adapter.effective_pathways(
        screen_batches, membership=membership
    )

    assert observed == expected


def test_omitted_membership_uses_assayloop_reactome_adapter(monkeypatch):
    membership, screen_batches = _fixture()
    monkeypatch.setattr(ep_adapter, "gmt_membership", lambda: membership)

    expected = assaybench_effective_pathways(
        screen_batches, membership=membership
    )
    observed = ep_adapter.effective_pathways(screen_batches)

    assert observed == expected


def test_pathway_weights_uses_the_same_default_membership(monkeypatch):
    membership, _ = _fixture()
    monkeypatch.setattr(ep_adapter, "gmt_membership", lambda: membership)

    weights, n_annotated = ep_adapter.pathway_weights(["g0", "G1", "missing"])

    assert weights == {"P0": 1.0, "P1": 1.0}
    assert n_annotated == 2
