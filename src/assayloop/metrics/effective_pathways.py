"""Reactome adapter for AssayBench's Effective Pathways metric.

The data-independent calculation lives in
:mod:`assaybench.benchmark.effective_pathways`. This module preserves
AssayLoop's existing API and supplies the paper's Reactome level-2 membership
from AssayLoop's explicitly fetched pathway files.
"""

from __future__ import annotations

from functools import lru_cache

from assaybench.benchmark.effective_pathways import (
    M_BATCH,
    M_DATASET,
    M_SCREEN,
    RETENTION,
    R_BATCH,
    R_DATASET,
    R_SCREEN,
    SEED,
    _Unit as _Unit,
    effective_n,
    effective_pathways as _effective_pathways,
    pathway_weights as _pathway_weights,
    scope_rng,
)

OTHER_GROUP = "Other (unmapped Reactome leaf)"


@lru_cache(maxsize=1)
def leaf_membership() -> dict[str, tuple[str, ...]]:
    """Map uppercase genes to disease-filtered Reactome leaf sets.

    The sunburst and LLM pathway heatmap use this raw tier for their own
    hierarchy roll-up. Table metrics use :func:`gmt_membership` instead.
    """
    from assayloop.scripts.paper_handoff_timeline import _load_pathway_membership

    return {
        gene: tuple(sorted(pathways))
        for gene, pathways in _load_pathway_membership().items()
    }


@lru_cache(maxsize=1)
def gmt_membership() -> dict[str, tuple[str, ...]]:
    """Map uppercase genes to the paper's Reactome level-2 groups.

    The filtered GMT's leaf sets are too fine to give EP an interpretable
    count. Each leaf is lifted to its direct level-2 group using AssayLoop's
    fetched Reactome hierarchy. Leaves absent from the hierarchy share one
    ``OTHER_GROUP`` bucket.
    """
    from assayloop.scripts.pathway_hierarchy import load as load_hierarchy

    subcategory_of = load_hierarchy()["subcategory_of"]
    return {
        gene: tuple(
            sorted(
                {
                    subcategory_of.get(pathway, OTHER_GROUP)
                    for pathway in pathways
                }
            )
        )
        for gene, pathways in leaf_membership().items()
    }


def pathway_weights(genes, membership=None) -> tuple[dict[str, float], int]:
    """Return fractional pathway weights using Reactome by default."""
    selected_membership = gmt_membership() if membership is None else membership
    return _pathway_weights(genes, membership=selected_membership)


def effective_pathways(
    screen_batches,
    *,
    rarefy: bool = True,
    seed: int = SEED,
    membership=None,
) -> dict:
    """Calculate EP, loading AssayLoop's Reactome membership when omitted."""
    selected_membership = gmt_membership() if membership is None else membership
    return _effective_pathways(
        screen_batches,
        membership=selected_membership,
        rarefy=rarefy,
        seed=seed,
    )


__all__ = [
    "M_BATCH",
    "M_DATASET",
    "M_SCREEN",
    "OTHER_GROUP",
    "RETENTION",
    "R_BATCH",
    "R_DATASET",
    "R_SCREEN",
    "SEED",
    "effective_n",
    "effective_pathways",
    "gmt_membership",
    "leaf_membership",
    "pathway_weights",
    "scope_rng",
]
