"""This repository's names for the benchmark's screen sets.

The screens, the manifests that name them, and the loader that resolves one
into :class:`~assaybench.ScreenRecord` objects all live in ``assaybench``:
which 20 screens a reported EF was averaged over is part of the benchmark
definition, not of this repo's plumbing.

What is left here is the part that genuinely belongs to this repo: concise
names for the curated paper sets and the complete train/validation/test folds.
The older ``public*`` names remain aliases because released checkpoints,
results and figure scripts record them. This module resolves either spelling
to arguments for :func:`assaybench.load_screens` and delegates.

Screen-set names that used to resolve against the Genentech-internal corpus
are listed but not implemented: asking for one raises and says so, rather than
quietly handing back public screens under an internal set's name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from assaybench.data.screen_sets import available_manifests, manifest_path
from assaybench.data.screens import ScreenRecord, gene_universe
from assaybench.data.screens import load_screens as _load_screens

log = logging.getLogger("assayloop.tasks.screen_sets")


# ---------------------------------------------------------------------------
# This repo's names
# ---------------------------------------------------------------------------


# The two curated sets reported/used in the paper, mapped to manifests that
# ship with assaybench. ``public*`` spellings are compatibility aliases for
# released artifacts; new commands and documentation use the explicit
# ``paper_*`` names.
_CURATED_ALIASES = {
    "paper_test": "assayloop-test",
    "paper_validation": "assayloop-validation",
    "public": "assayloop-test",
    "public_validation": "assayloop-validation",
    "public_val_curated": "assayloop-validation",
}

# The whole-fold sets, which have no manifest because they are not curated.
#
# ``train``, ``validation`` and ``test`` are the complete AssayBench folds.
# The full test fold is not what the paper reports. The shipped
# screen-description embeddings cover only its curated 20-screen subset, so
# ASSAYFORMER on ``test`` needs an OPENAI_API_KEY (it raises rather than
# substituting). Older ``public_*`` spellings remain compatibility aliases.
_WHOLE_FOLD_SETS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
    "public_train": "train",
    "public_val": "validation",
    "public_test": "test",
}

# Which fold an explicit ``dataset_names`` request resolves against, per
# ``target_set``. Anything not listed falls back to the test fold.
_FOLD_FOR_NAMES = {
    "paper_test": "test",
    "paper_validation": "validation",
    "train": "train",
    "validation": "validation",
    "test": "test",
    "public": "test",
    "public_test": "test",
    "public_train": "train",
    "public_val": "validation",
    "public_validation": "validation",
    "public_val_curated": "validation",
}

_RETIRED_INTERNAL_SCREEN_SETS = {"default", "all"}

# Public screen sets that existed during development and no longer ship. Named
# for the same reason: an old command line should be told what happened rather
# than fall through to the "treat it as a YAML path" branch and report a
# baffling FileNotFoundError.
_RETIRED_PUBLIC_SCREEN_SETS = {
    "public_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's test set is 'paper_test'"
    ),
    "public_validation_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's validation set is 'paper_validation'"
    ),
    "public_val_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's validation set is 'paper_validation'"
    ),
    "public_combined": (
        "the combined test+validation pool was never used in the paper and was "
        "dropped; use 'paper_test' or 'paper_validation', which respect the "
        "temporal split"
    ),
    "public_combined_validation": (
        "the combined test+validation pool was never used in the paper and was "
        "dropped; use 'paper_test' or 'paper_validation', which respect the "
        "temporal split"
    ),
}


def default_paper_test_screen_set_path() -> Path:
    """Path of the paper's 20-screen test manifest, inside the assaybench package."""
    return manifest_path(_CURATED_ALIASES["paper_test"])


def default_paper_validation_screen_set_path() -> Path:
    """Path of the paper's 20-screen validation manifest, inside assaybench."""
    return manifest_path(_CURATED_ALIASES["paper_validation"])


def default_public_screen_set_path() -> Path:
    """Compatibility alias for :func:`default_paper_test_screen_set_path`."""
    return default_paper_test_screen_set_path()


def default_public_validation_screen_set_path() -> Path:
    """Compatibility alias for :func:`default_paper_validation_screen_set_path`."""
    return default_paper_validation_screen_set_path()


def load_screens(
    *,
    dataset_names: Iterable[str] | None = None,
    yaml_path: Path | str | None = None,
    target_set: str = "paper_test",
    strict: bool | None = None,
) -> list[ScreenRecord]:
    """Resolve a screen set to a list of ScreenRecords.

    Every screen set resolves against the public ``Genentech/assaybench``
    dataset on the Hub.

    Resolution order:

    - If ``dataset_names`` is given, return exactly those screens. With
      ``strict=True`` (the default for this branch), missing names raise.
    - Elif ``yaml_path`` is given, read the YAML and load those names.
    - Elif ``target_set == "paper_test"``, use the ``assayloop-test`` manifest
      from :mod:`assaybench.data.screen_sets` — the paper's 20-screen test
      set, a curated subset of the public AssayBench test split.
    - Elif ``target_set == "paper_validation"``, use the
      ``assayloop-validation`` manifest.
    - Elif ``target_set`` is ``"train"``, ``"validation"`` or ``"test"``,
      return every screen on that fold of the public biogrid split. ``test``
      is the whole fold the paper's curated ``paper_test`` set is drawn from;
      use ``"paper_test"`` for the paper's numbers.
    - The legacy names ``public``, ``public_validation``, ``public_train``,
      ``public_val`` and ``public_test`` resolve to the corresponding sets
      above for compatibility with released artifacts.
    - Elif ``target_set`` names a manifest shipped with assaybench (see
      :func:`assaybench.data.screen_sets.available_manifests`, e.g.
      ``"lopo-drug-test"``), use that manifest.
    - Else ``target_set`` is interpreted as a path to a YAML manifest, which
      must name the public dataset in its ``source`` block.

    ``strict`` controls whether missing requested screens cause a hard
    failure. By default we treat explicit ``dataset_names`` requests as
    strict (typos shouldn't silently produce a partial benchmark) and
    YAML-resolved sets as non-strict (stale YAML is allowed to drift).
    """
    if target_set in _RETIRED_INTERNAL_SCREEN_SETS:
        raise ValueError(
            f"target_set={target_set!r} selected the Genentech-internal screen "
            "corpus, which is not part of the public release. Public screen "
            "sets: 'paper_test', 'paper_validation', 'train', 'validation', "
            "'test'."
        )
    if target_set in _RETIRED_PUBLIC_SCREEN_SETS:
        raise ValueError(
            f"target_set={target_set!r} is no longer available: "
            f"{_RETIRED_PUBLIC_SCREEN_SETS[target_set]}."
        )

    if dataset_names is not None:
        # Explicit names win over any set: resolve them on whichever fold the
        # named set lives on, defaulting to test.
        return _load_screens(
            dataset_names=dataset_names,
            split_value=_FOLD_FOR_NAMES.get(target_set, "test"),
            strict=strict,
        )
    if yaml_path is not None:
        return _load_screens(yaml_path=yaml_path, strict=strict)
    if target_set in _CURATED_ALIASES:
        return _load_screens(_CURATED_ALIASES[target_set], strict=strict)
    if target_set in _WHOLE_FOLD_SETS:
        return _load_screens(
            split_value=_WHOLE_FOLD_SETS[target_set], strict=strict
        )
    if target_set in available_manifests() or Path(target_set).is_file():
        return _load_screens(target_set, strict=strict)

    # Neither a known set name nor a readable file. Say both things it
    # could have been; a bare ``FileNotFoundError: 'lopo-nope-test'`` from
    # the path branch reads like a missing file rather than a typo.
    raise ValueError(
        f"target_set={target_set!r} is neither a screen-set name nor a "
        "readable manifest path. Names in this repo: "
        "paper_test, paper_validation, train, validation, test. "
        "Legacy aliases: public, public_validation, public_train, public_val, "
        "public_test, public_val_curated. "
        f"Manifests shipped with assaybench: {', '.join(available_manifests())}."
    )


__all__ = [
    "default_paper_test_screen_set_path",
    "default_paper_validation_screen_set_path",
    "default_public_screen_set_path",
    "default_public_validation_screen_set_path",
    "gene_universe",
    "load_screens",
]
