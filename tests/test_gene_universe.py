"""The full-genome candidate pool has one definition, not four.

Before this helper existed the f2 pool was rebuilt inline in
``full_genome_table.py``, ``assayloop_no_context.py`` and ``amortized/rl.py``,
and a fourth copy in ``cli.py`` silently omitted the frequency filter -- so
``assayloop run --full-genome`` handed a ranker 22,174 candidates while every
number in the paper was scored against 21,147. These tests pin the semantics
the four sites now share.

The helper itself now lives in ``assaybench`` (with its own copy of these
cases in ``AssayBench/tests/test_screens.py``); what this file additionally
covers is that the re-export through ``assayloop.tasks`` still resolves, which
is the import path every script, doc and model card in this repo uses.
"""

from __future__ import annotations

from dataclasses import dataclass

from assayloop.tasks import gene_universe


@dataclass
class _Screen:
    """Just enough of a ScreenRecord for the helper: it only reads ``.genes``."""

    genes: list[str]


def test_keeps_genes_in_at_least_two_screens_by_default():
    screens = [_Screen(["A", "B", "C"]), _Screen(["B", "C", "D"]), _Screen(["C"])]
    # A and D appear once each; B twice; C three times.
    assert gene_universe(screens) == ["B", "C"]


def test_min_screen_freq_zero_is_the_plain_union():
    screens = [_Screen(["A", "B"]), _Screen(["B", "C"])]
    assert gene_universe(screens, min_screen_freq=0) == ["A", "B", "C"]


def test_result_is_sorted_and_deduplicated():
    screens = [_Screen(["Z", "A"]), _Screen(["A", "Z"])]
    assert gene_universe(screens) == ["A", "Z"]


def test_duplicate_symbol_within_one_library_counts_once():
    """A library that lists a symbol twice must not satisfy freq >= 2 alone.

    Duplicated symbols do occur -- ``AssayBenchGeneBatchTask.__init__`` warns
    about them -- and counting both occurrences would admit a gene that only
    one screen ever measured.
    """
    screens = [_Screen(["A", "A", "B"]), _Screen(["B"])]
    assert gene_universe(screens) == ["B"]


def test_higher_cutoff_is_a_subset_of_a_lower_one():
    screens = [_Screen(["A", "B", "C"]), _Screen(["B", "C"]), _Screen(["C"])]
    f1 = gene_universe(screens, min_screen_freq=1)
    f2 = gene_universe(screens, min_screen_freq=2)
    f3 = gene_universe(screens, min_screen_freq=3)
    assert set(f3) <= set(f2) <= set(f1)
    assert (f1, f2, f3) == (["A", "B", "C"], ["B", "C"], ["C"])


def test_empty_input_is_an_empty_pool():
    assert gene_universe([]) == []


def test_reexports_resolve_to_the_assaybench_definitions():
    """``assayloop.tasks`` must keep re-exporting what moved to assaybench."""
    import assaybench

    from assayloop.tasks import ScreenRecord, gene_universe, screen_from_example

    assert gene_universe is assaybench.gene_universe
    assert ScreenRecord is assaybench.ScreenRecord
    assert screen_from_example is assaybench.screen_from_example
