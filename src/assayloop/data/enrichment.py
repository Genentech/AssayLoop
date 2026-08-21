"""Enrichment analysis for BioBO (Li et al., ICLR 2026).

Implements the hypergeometric over-representation test and constructs the
πBO prior (Hvarfner et al. 2022) from enrichment results. Used by the
``BioUCBFromModel`` acquisition to bias gene selection toward enriched
pathways while preserving exploration via the decaying prior exponent.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import hypergeom

from .gene_sets import _resolve_gmt


def load_pathway_db(source: str = "h.all") -> dict[str, set[str]]:
    """Load a ``.gmt`` file into ``{pathway_name: set_of_gene_symbols}``."""
    path = _resolve_gmt(source)
    db: dict[str, set[str]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.upper() for g in parts[2:] if g}
            if genes:
                db[name] = genes
    return db


def _enrichment_test(
    top_genes: set[str],
    pathway_db: dict[str, set[str]],
    background_size: int,
) -> list[dict]:
    """Hypergeometric over-representation test with Bonferroni correction.

    Returns a list of dicts with keys: pathway, p_value, p_adj, odds_ratio,
    combined_score -- for pathways with ``p_adj < 0.05``.
    """
    n_tests = len(pathway_db)
    if n_tests == 0 or not top_genes:
        return []

    S = len(top_genes)
    N = background_size
    results = []

    for pw_name, pw_genes in pathway_db.items():
        K = len(pw_genes)
        overlap = len(top_genes & pw_genes)
        if overlap == 0:
            continue

        # P(X >= overlap) under hypergeometric(N, K, S)
        p_val = hypergeom.sf(overlap - 1, N, K, S)
        p_adj = min(p_val * n_tests, 1.0)
        if p_adj >= 0.05:
            continue

        # Odds ratio from the 2x2 contingency table
        a = overlap
        b = S - overlap
        c = K - overlap
        d = N - S - K + overlap
        odds = (a * d) / max(b * c, 1e-12)

        # Combined score (Chen et al. 2013): -odds_ratio * log(p_value)
        combined = -odds * math.log(max(p_val, 1e-300))

        results.append({
            "pathway": pw_name,
            "p_value": p_val,
            "p_adj": p_adj,
            "odds_ratio": odds,
            "combined_score": combined,
            "overlap": overlap,
            "pathway_size": K,
        })

    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results


def enrichment_prior(
    top_genes: list[str],
    unlabeled_genes: list[str],
    pathway_db: dict[str, set[str]],
    background_size: int,
    temperature: float = 0.1,
    agg: str = "mean",
) -> tuple[dict[str, float], list[dict]]:
    """Construct the piBO prior pi(x) from enrichment analysis (Eq. 6).

    Returns ``(prior_dict, enrichment_results)`` where ``prior_dict`` maps
    each unlabeled gene to its prior probability and ``enrichment_results``
    is the list of significant pathways (for tracing/debugging).
    """
    top_set = {g.upper() for g in top_genes}
    unlabeled_upper = [g.upper() for g in unlabeled_genes]
    U_n = len(unlabeled_upper)

    if U_n == 0:
        return {}, []

    results = _enrichment_test(top_set, pathway_db, background_size)

    if not results:
        uniform = 1.0 / U_n
        return {g: uniform for g in unlabeled_genes}, []

    sig_pathways = {r["pathway"]: r["combined_score"] for r in results}

    agg_fn = np.mean if agg == "mean" else np.max
    base_logit = math.log(1.0 / max(U_n - 1, 1))  # logit(1/U_n)

    scores = np.full(U_n, base_logit, dtype=np.float64)
    for i, g in enumerate(unlabeled_upper):
        cs = [
            sig_pathways[pw]
            for pw, pw_genes in pathway_db.items()
            if pw in sig_pathways and g in pw_genes
        ]
        if cs:
            scores[i] += (1.0 / temperature) * float(agg_fn(cs))

    scores -= scores.max()
    exp_s = np.exp(scores)
    prior = exp_s / exp_s.sum()

    return (
        {g: float(prior[i]) for i, g in enumerate(unlabeled_genes)},
        results,
    )


__all__ = ["load_pathway_db", "enrichment_prior"]
