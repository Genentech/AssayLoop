"""Locating this repo's MSigDB ``.gmt`` files.

The membership map itself -- parsing, pathway lookup, per-batch Jaccard
statistics, the random-draw baseline -- lives in
:class:`assaybench.data.gene_sets.GeneSetMembership`. This module only answers
"which file", because there are two places a ``.gmt`` can come from and this
repo uses both:

* :mod:`assaybench.assets`, which downloads a registered collection by name and
  checks its sha256. The three MSigDB collections the paper uses are registered
  there (``msigdb-hallmark``, ``msigdb-go-bp``, ``msigdb-canonical-pathways``),
  so none of them is committed here.
* ``src/assayloop/data/gene_sets`` (override with ``ASSAYLOOP_GENE_SETS``),
  populated by ``scripts/fetch_gene_sets.sh``. Scripts here also use Reactome,
  which is not in the assaybench registry, so the local directory stays.

The local directory wins when both have the file, so pointing
``ASSAYLOOP_GENE_SETS`` at a pinned copy overrides the shared cache. Neither
lookup ever substitutes a different collection: an unfound or ambiguous source
raises and says what to fetch.

A ``.gmt`` file is one gene set per line::

    <set_name>\\t<description>\\t<gene1>\\t<gene2>\\t...
"""

from __future__ import annotations

from pathlib import Path

from assaybench import assets
from assaybench.data.gene_sets import GeneSetMembership

from .. import config


class MissingGeneSets(FileNotFoundError):
    """Raised when the requested ``.gmt`` membership file is absent."""


def _registered_asset(source: str) -> str | None:
    """The assaybench asset whose name or filename matches ``source``, if any.

    Matches ``"msigdb-hallmark"``, ``"h.all.v2023.2.Hs.symbols.gmt"``, and the
    ``"h.all"`` / ``"c5.go.bp"`` prefixes the configs here are written with.
    Returns ``None`` rather than guessing when more than one collection
    matches.
    """
    token = source.lower()
    hits = [
        a.name
        for a in assets.iter_assets()
        if token in (a.name.lower(), a.filename.lower()) or a.filename.lower().startswith(token)
    ]
    return hits[0] if len(hits) == 1 else None


def _resolve_local(source: str, root: Path) -> Path | None:
    """Find ``source`` under ``root``, or ``None``. Ambiguity raises."""
    if not root.exists():
        return None
    for name in (source, f"{source}.gmt", f"{source}.symbols.gmt"):
        p = root / name
        if p.is_file():
            return p
    # Fuzzy: any .gmt whose stem contains the source token, so "c5.go.bp" finds
    # "c5.go.bp.v2023.2.Hs.symbols.gmt". Ambiguity is an error rather than a
    # coin flip -- silently scoring against a different MSigDB collection would
    # change the pathway numbers without changing anything visible.
    matches = [p for p in sorted(root.glob("*.gmt")) if source.lower() in p.name.lower()]
    if len(matches) > 1:
        raise MissingGeneSets(
            f"Gene-set source {source!r} is ambiguous under {root}: "
            f"{[p.name for p in matches]}. Name the file exactly."
        )
    return matches[0] if matches else None


def _resolve_gmt(source: str) -> Path:
    """Path of the ``.gmt`` for ``source``: local directory first, then the
    assaybench asset cache.

    Raises:
        MissingGeneSets: If neither has it. The message names both ways to get
            it. Nothing here falls back to another collection -- a pathway
            number scored against Hallmark instead of GO BP looks entirely
            normal and is not the number the paper reports.
    """
    root = Path(config.GENE_SETS_PATH)
    local = _resolve_local(source, root)
    if local is not None:
        return local

    asset_name = _registered_asset(source)
    if asset_name is not None:
        try:
            return assets.asset_path(asset_name)
        except assets.MissingAsset as exc:
            raise MissingGeneSets(
                f"Gene-set source {source!r} is the registered assaybench asset "
                f"{asset_name!r}, which is not downloaded, and it is not in "
                f"{root} either.\n"
                f"  Fetch it: assaybench download {asset_name}\n"
                f"  Or:       scripts/fetch_gene_sets.sh"
            ) from exc

    available = [p.name for p in sorted(root.glob("*.gmt"))] if root.exists() else []
    raise MissingGeneSets(
        f"Gene-set source {source!r} not found under {root} "
        f"(available there: {available or 'none'}) and not a registered "
        f"assaybench asset ({', '.join(a.name for a in assets.iter_assets())}). "
        "Run scripts/fetch_gene_sets.sh, or set ASSAYLOOP_GENE_SETS to a "
        "directory containing the file."
    )


_DEFAULT_CACHE: dict[str, GeneSetMembership] = {}


def load_default_gene_sets(source: str | None = None) -> GeneSetMembership:
    """Load (and memoise) the default gene-set membership.

    Raises :class:`MissingGeneSets` if the ``.gmt`` files have not been
    downloaded. It used to return ``None`` instead, which meant a missing
    download silently dropped the pathway-overlap column from the results
    table: the run still finished and the table still rendered, just without
    the metric and without saying so. Callers that want pathway diversity to be
    genuinely optional must catch this and report the omission.
    """
    key = source or config.GENE_SETS_SOURCE
    if key not in _DEFAULT_CACHE:
        _DEFAULT_CACHE[key] = GeneSetMembership(_resolve_gmt(key), source=key)
    return _DEFAULT_CACHE[key]


__all__ = ["GeneSetMembership", "MissingGeneSets", "load_default_gene_sets"]
