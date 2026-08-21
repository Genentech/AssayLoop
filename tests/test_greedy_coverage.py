"""Greedy must not fill a batch with candidates the model never scored.

Most models score every candidate, so this is invisible for them. LLMNN does
not: it scores only genes that have an embedding in the PRESAGE source. Under
the old implementation an unscored gene sorted as ``-inf`` and could still be
picked once the scored genes ran out, which is a random fill wearing a greedy
label -- exactly what the inner loop's shortfall accounting exists to avoid.
"""

from __future__ import annotations

import pytest

from assayloop.acquisitions.greedy_from_model import GreedyFromModel
from assaybench.core.types import ModelPrediction


def _pred(scores):
    return ModelPrediction(scores=scores, metadata={"name": "stub"})


CANDIDATES = ["A", "B", "C", "D", "E"]


def test_full_coverage_picks_top_k_by_score():
    """The common case: every candidate scored, greedy is plain top-k."""
    acq = GreedyFromModel(seed=0)
    pred = _pred({"A": 0.1, "B": 0.9, "C": 0.5, "D": 0.7, "E": 0.2})
    assert acq.suggest([], CANDIDATES, 3, pred) == ["B", "D", "C"]


def test_partial_coverage_returns_short_batch():
    """Only 2 of 5 scored, so a batch of 4 comes back with 2 -- not padded."""
    acq = GreedyFromModel(seed=0)
    got = acq.suggest([], CANDIDATES, 4, _pred({"B": 0.9, "D": 0.7}))
    assert got == ["B", "D"]


def test_partial_coverage_never_yields_an_unscored_candidate():
    """Whatever the seed, an unscored gene must not appear in the batch."""
    scored = {"C": 0.4}
    for seed in range(25):
        got = GreedyFromModel(seed=seed).suggest([], CANDIDATES, 5, _pred(scored))
        assert set(got) <= set(scored), f"seed={seed} leaked an unscored candidate: {got}"


def test_epsilon_explore_stays_within_scored_candidates():
    """The ε-greedy explore slots draw from scored candidates only."""
    scored = {"A": 0.1, "B": 0.9, "C": 0.5}
    for seed in range(25):
        got = GreedyFromModel(seed=seed, epsilon=0.9).suggest(
            [], CANDIDATES, 5, _pred(scored)
        )
        assert set(got) <= set(scored), f"seed={seed} explored into unscored: {got}"
        assert len(got) == len(set(got)), f"seed={seed} returned duplicates: {got}"


@pytest.mark.parametrize("pred", [None, _pred({})])
def test_no_scores_at_all_still_falls_back_to_random(pred):
    """Unchanged behaviour: no prediction (or an empty one) samples randomly.

    This path is what makes the change above safe for the existing models --
    the two that return ``scores={}`` only do so when ``candidates`` is empty.
    """
    got = GreedyFromModel(seed=0).suggest([], CANDIDATES, 3, pred)
    assert len(got) == 3
    assert set(got) <= set(CANDIDATES)


def test_empty_candidates_returns_empty():
    assert GreedyFromModel(seed=0).suggest([], [], 3, _pred({})) == []
