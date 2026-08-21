"""Resolving a screen-set name to a list of :class:`ScreenRecord`.

Every set resolves against the public ``Genentech/assaybench`` dataset on the
Hub -- the corpus the paper reports on. The screen lists themselves are the
manifests shipped in :mod:`assaybench.data.screen_sets`, which is where they
belong: which 20 screens a reported EF was averaged over is part of the
benchmark definition, not of this repo's plumbing. This module is the adapter
from a manifest to loaded :class:`ScreenRecord` objects, and it keeps the
short ``target_set`` names ("public", "public_train") that this repo's configs
and checkpoints were written against.

Screen-set names that used to resolve against the Genentech-internal corpus
are listed but not implemented: asking for one raises and says so, rather than
quietly handing back public screens under an internal set's name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from assaybench import AssayBenchDataset
from assaybench.data.screen_sets import (
    ScreenSetManifest,
    available_manifests,
    load_manifest,
    load_manifest_file,
    manifest_path,
)

from .gene_batch import ScreenRecord, screen_from_example

log = logging.getLogger("assayloop.tasks.screen_sets")


# ---------------------------------------------------------------------------
# Screen loaders
# ---------------------------------------------------------------------------


# This repo's short names for the two curated sets, mapped to the manifests
# that now ship with assaybench. The short names stay because every config,
# checkpoint and figure script in this repo says ``screen_set: public``.
_CURATED_ALIASES = {
    "public": "assayloop-test",
    "public_validation": "assayloop-validation",
    "public_val_curated": "assayloop-validation",
}


def _source_is_public(source: dict[str, Any]) -> bool:
    """A public manifest's ``source`` block names the public Hub dataset.

    A manifest without one is a leftover internal file, which this release
    cannot resolve.
    """
    return ((source or {}).get("dataset") or "").lower().endswith("/assaybench")


def default_public_screen_set_path() -> Path:
    """Path of the paper's 20-screen test manifest, inside the assaybench package."""
    return manifest_path(_CURATED_ALIASES["public"])


def default_public_validation_screen_set_path() -> Path:
    """Path of the paper's 20-screen validation manifest, inside assaybench."""
    return manifest_path(_CURATED_ALIASES["public_validation"])


class _DefaultFormatDict(dict):
    """dict that returns ``"Not specified"`` for missing format keys.

    Lets ``str.format_map`` render a template even when a row is missing an
    optional placeholder field, instead of raising ``KeyError``.
    """

    def __missing__(self, key: str) -> str:  # noqa: D401
        return "Not specified"


def _render_public_question(item: dict[str, Any]) -> str:
    """Render the AssayBench paper ``biogrid_ranking_prompt`` for a public row.

    Mirrors AssayBench's public ``BiogridDataset`` (``assaybench.dataset.dataset``):
    load the ``biogrid_ranking_prompt`` template and ``.format(**item)`` it
    against the screen's structured fields (``cell_line``, ``library_type``,
    ``experimental_setup``, ``duration``, ``condition_clause``, ``phenotype``,
    ``significance_criteria``, ``ranking_rationale``, ``notes``, ...), which the
    public Genentech/assaybench rows carry but do NOT pre-render into a
    ``question`` field. Falls back to the bare template on any error.
    """
    from assaybench.utils.prompt_loaders import load_objective_prompt

    template = load_objective_prompt("biogrid_ranking_prompt")
    merged = dict(item)
    # AssayBench strips a trailing period from phenotype for cleaner prose
    # (the template appends its own punctuation around it).
    phen = str(merged.get("phenotype") or "")
    if phen.endswith("."):
        merged["phenotype"] = phen[:-1]
    try:
        return template.format_map(_DefaultFormatDict(merged))
    except Exception:
        log.warning(
            "Failed to render biogrid_ranking_prompt for public screen %r; "
            "using bare template.",
            item.get("dataset_name") or item.get("screen_name"),
        )
        return template


def _load_public_screens(
    *,
    wanted: set[str] | None,
    split_field: str = "yearfold0",
    split_value: str | list[str] = "test",
    strict: bool,
) -> list[ScreenRecord]:
    """Load screens from the public ``Genentech/assaybench`` dataset.

    The public dataset packs every benchmark entry into one ``train`` split
    with per-fold split labels in side columns (``yearfold0``,
    ``randomfold0``, ...). We filter on the requested ``split_field ==
    split_value`` and then by ``dataset_name``.

    ``split_value`` may be a list to match multiple folds (e.g.
    ``["test", "validation"]`` for combined splits).
    """
    ds = AssayBenchDataset(dataset_group="Genentech/assaybench", dataset_name="biogrid")
    ds.load()
    # The public dataset has a single 'train' arrow split; the actual fold
    # labels live in side columns.
    rows = ds.dataset["train"]

    accepted_values = {split_value} if isinstance(split_value, str) else set(split_value)

    out: list[ScreenRecord] = []
    seen: set[str] = set()
    for ex in rows:
        if ex.get(split_field) not in accepted_values:
            continue
        name = ex.get("dataset_name") or ex.get("screen_name")
        if not name or name in seen:
            continue
        if wanted is not None and name not in wanted:
            continue
        seen.add(name)
        ex2 = dict(ex)
        # Provenance label: which fold column and value this screen came in
        # on. Set unconditionally -- if the raw row happens to carry its own
        # ``split`` column it is about the arrow split, not the fold we asked
        # for, and would be misleading here.
        ex2["split"] = f"public_{split_field}_{split_value}"
        # Public rows don't carry a pre-rendered prompt; render the official
        # AssayBench ``biogrid_ranking_prompt`` so the LLM acquisition uses the
        # paper prompt rather than the generic fallback block.
        if not ex2.get("question"):
            ex2["question"] = _render_public_question(ex2)
        out.append(screen_from_example(ex2))

    if wanted is not None:
        missing = wanted - seen
        if missing:
            msg = (
                f"{len(missing)} public screens not found "
                f"(first few: {sorted(missing)[:5]})"
            )
            if strict:
                raise ValueError(f"_load_public_screens: {msg}")
            log.warning("_load_public_screens: %s", msg)
    return out


# Screen sets that resolved against the Genentech-internal corpus, which is not
# part of the public release. Named so that a stale config or checkpoint asks for
# them by name and gets told, rather than silently receiving public screens.
_RETIRED_INTERNAL_SCREEN_SETS = {"default", "all"}

# Public screen sets that existed during development and no longer ship. Named
# for the same reason: an old command line should be told what happened rather
# than fall through to the "treat it as a YAML path" branch and report a
# baffling FileNotFoundError.
_RETIRED_PUBLIC_SCREEN_SETS = {
    "public_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's test set is 'public'"
    ),
    "public_validation_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's validation set is 'public_validation'"
    ),
    "public_val_no_gemini_floor": (
        "the no-Gemini-floor robustness variants were dropped from the release; "
        "the paper's validation set is 'public_validation'"
    ),
    "public_combined": (
        "the combined test+validation pool was never used in the paper and was "
        "dropped; use 'public' or 'public_validation', which respect the "
        "temporal split"
    ),
    "public_combined_validation": (
        "the combined test+validation pool was never used in the paper and was "
        "dropped; use 'public' or 'public_validation', which respect the "
        "temporal split"
    ),
}


def load_screens(
    *,
    dataset_names: Iterable[str] | None = None,
    yaml_path: Path | str | None = None,
    target_set: str = "public",
    strict: bool | None = None,
) -> list[ScreenRecord]:
    """Resolve a screen set to a list of ScreenRecords.

    Every screen set resolves against the public ``Genentech/assaybench``
    dataset on the Hub.

    Resolution order:

    - If ``dataset_names`` is given, return exactly those screens. With
      ``strict=True`` (the default for this branch), missing names raise.
    - Elif ``yaml_path`` is given, read the YAML and load those names.
    - Elif ``target_set == "public"``, use the ``assayloop-test`` manifest
      from :mod:`assaybench.data.screen_sets` — the paper's 20-screen test
      set, a curated subset of the public AssayBench test split.
    - Elif ``target_set == "public_validation"`` or
      ``"public_val_curated"``, use the ``assayloop-validation`` manifest.
    - Elif ``target_set == "public_train"``, return every screen on the
      public biogrid TRAIN fold (``yearfold0 == "train"``) — used to
      generate distillation / post-training trace datasets.
    - Elif ``target_set == "public_val"``, return every screen on the
      public biogrid VALIDATION fold (``yearfold0 == "validation"``) —
      used as the prompt-optimization (GEPA) validation set.
    - Elif ``target_set == "public_test"``, return every screen on the
      public biogrid TEST fold (``yearfold0 == "test"``) — the whole fold
      the paper's curated ``public`` set is drawn from. Not what the paper
      reports; use ``"public"`` for that.
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
            "sets: 'public' (paper test set), 'public_validation', "
            "'public_train', 'public_val', 'public_test'."
        )
    if target_set in _RETIRED_PUBLIC_SCREEN_SETS:
        raise ValueError(
            f"target_set={target_set!r} is no longer available: "
            f"{_RETIRED_PUBLIC_SCREEN_SETS[target_set]}."
        )

    # Resolve where the screen names and the source corpus come from. Either a
    # manifest (shipped or on disk), or a ``source`` block synthesized here for
    # the whole-fold sets, which have no manifest because they are not curated.
    manifest: ScreenSetManifest | None = None
    source: dict[str, Any] | None = None
    if dataset_names is not None:
        wanted: set[str] | None = set(dataset_names)
        default_strict = True
        if target_set in (
            "public",
            "public_train",
            "public_val",
            "public_test",
            "public_validation",
            "public_val_curated",
        ):
            # Resolve names against the public dataset even without a YAML.
            # ``public`` and ``public_test`` => test fold; the validation
            # variants and ``public_val`` => validation fold; ``public_train``
            # => train.
            _public_split = {
                "public_train": "train",
                "public_val": "validation",
                "public_validation": "validation",
                "public_val_curated": "validation",
            }.get(target_set, "test")
            source = {
                "dataset": "Genentech/assaybench",
                "config": "biogrid",
                "split_field": "yearfold0",
                "split_value": _public_split,
            }
    elif yaml_path is not None:
        manifest = load_manifest_file(yaml_path)
        default_strict = False
    elif target_set in _CURATED_ALIASES:
        manifest = load_manifest(_CURATED_ALIASES[target_set])
        default_strict = False
    elif target_set == "public_train":
        # All screens on the public biogrid TRAIN fold (no curated subset).
        # Used to generate distillation / post-training trace datasets,
        # held out from the public test set used for evaluation.
        source = {
            "dataset": "Genentech/assaybench",
            "config": "biogrid",
            "split_field": "yearfold0",
            "split_value": "train",
        }
        wanted = None
        default_strict = False
    elif target_set == "public_val":
        # All screens on the public biogrid VALIDATION fold (no curated
        # subset). Used as the GEPA / prompt-optimization validation set,
        # held out from both the train fold and the public test set.
        source = {
            "dataset": "Genentech/assaybench",
            "config": "biogrid",
            "split_field": "yearfold0",
            "split_value": "validation",
        }
        wanted = None
        default_strict = False
    elif target_set == "public_test":
        # All screens on the public biogrid TEST fold (no curated subset).
        # The paper reports on the curated 20 (``public``); this is the whole
        # fold they were drawn from, for anyone who wants the broader
        # evaluation. Note that the shipped screen-description embeddings
        # cover only the curated 20 of this fold, so ASSAYFORMER on the full
        # fold needs an OPENAI_API_KEY -- it raises rather than substituting.
        source = {
            "dataset": "Genentech/assaybench",
            "config": "biogrid",
            "split_field": "yearfold0",
            "split_value": "test",
        }
        wanted = None
        default_strict = False
    elif target_set in available_manifests():
        # A manifest shipped with assaybench by its own name, e.g. the LOPO
        # folds ("lopo-drug-test").
        manifest = load_manifest(target_set)
        default_strict = False
    elif Path(target_set).is_file():
        manifest = load_manifest_file(target_set)
        default_strict = False
    else:
        # Neither a known set name nor a readable file. Say both things it
        # could have been; a bare ``FileNotFoundError: 'lopo-nope-test'`` from
        # the path branch reads like a missing file rather than a typo.
        raise ValueError(
            f"target_set={target_set!r} is neither a screen-set name nor a "
            "readable manifest path. Names in this repo: "
            f"{', '.join(sorted(_CURATED_ALIASES))}, public_train, public_val, "
            "public_test. "
            f"Manifests shipped with assaybench: {', '.join(available_manifests())}."
        )

    if manifest is not None:
        # Log which set is being loaded. Two runs of the same model can differ
        # only in the screen subset behind them, and that is invisible in the
        # resulting numbers.
        log.info("load_screens: %s", manifest.summary())
        source = manifest.source
        wanted = set(manifest.dataset_names)
    strict = default_strict if strict is None else bool(strict)

    # Everything resolves against the public dataset.
    if source is None:
        # Explicit ``dataset_names`` with no screen set named: the public test
        # fold is the default corpus.
        source = {"dataset": "Genentech/assaybench", "config": "biogrid"}
    if not _source_is_public(source):
        raise ValueError(
            f"{yaml_path or target_set} is not a public screen set: its 'source' "
            "block does not name the Genentech/assaybench dataset. Internal "
            "screen manifests cannot be resolved by the public release."
        )
    return _load_public_screens(
        wanted=wanted,
        split_field=source.get("split_field", "yearfold0"),
        split_value=source.get("split_value", "test"),
        strict=strict,
    )


__all__ = [
    "default_public_screen_set_path",
    "default_public_validation_screen_set_path",
    "load_screens",
]
