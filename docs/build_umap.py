#!/usr/bin/env python3
"""Build ``assets/data/gene_umap.json`` -- the BPMF gene-embedding explorer.

The paper's Figure 11 shows a 2-D projection of the K=10 BPMF gene embedding
with three colourings (HDBSCAN cluster, CORUM complex family, Reactome
pathway) as three static panels. This writes the same projection out as data
so ``umap.html`` can colour it by any of those on demand, plus hit rate and
DepMap common-essential status, and let the reader search for a gene. The
interactive Reactome view rolls the paper's specific pathways up to top-level
categories so the long tail does not collapse almost every annotated gene
into a single grey "other" group.

The coordinates come from the same helper the figure uses --
``assayloop.scripts._bpmf_embedding.run_umap_cosine``, at the same seed -- so
the scatter on the site is the figure's projection rather than a second,
differently-seeded one that would put the same gene in a different place.

Prerequisites::

    # the cluster labels and per-gene screen counts
    python -m assayloop.scripts.paper_bpmf_k10_export --k 10
    # then this
    python docs/build_umap.py

Nothing here falls back. A missing checkpoint, a missing export, or a gene in
the export that is absent from the embedding raises and names what it wanted.
A scatter silently missing a third of its genes still looks like a scatter.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_umap")

DOCS = Path(__file__).resolve().parent
DATA_OUT = DOCS / "assets" / "data"

# The figure's default seed (``paper_bpmf_k10_organization.py --umap-seed``).
# Changing it re-lays-out the whole scatter, so it is pinned here rather than
# left to whatever the helper's own default happens to be.
UMAP_SEED = 7

# Coordinates are rounded before shipping: 3 decimals on a UMAP axis is far
# below the resolution of a scatter point, and it cuts the JSON roughly in
# half.
COORD_DP = 3


class MissingInput(FileNotFoundError):
    """An input this script needs has not been generated yet."""


def _require(path: Path, what: str, how: str) -> Path:
    if not path.is_file():
        raise MissingInput(f"{what} not found at {path}.\n  Generate it: {how}")
    return path


def load_clusters(path: Path) -> list[dict]:
    """Rows of ``bpmf_k10_gene_clusters.tsv``, typed."""
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise MissingInput(f"{path} has a header but no genes.")
    required = {"gene", "cluster_id", "cluster_label", "train_screens_present",
                "train_hits", "hit_rate", "depmap_common_essential"}
    missing = required - set(rows[0])
    if missing:
        raise MissingInput(
            f"{path} is missing column(s) {sorted(missing)}; it has "
            f"{sorted(rows[0])}. Re-run paper_bpmf_k10_export.py.")
    out = []
    for r in rows:
        out.append({
            "gene": r["gene"],
            "cluster_id": int(r["cluster_id"]),
            "cluster_label": r["cluster_label"],
            "screens": int(r["train_screens_present"]),
            "hits": int(r["train_hits"]),
            "hit_rate": float(r["hit_rate"]),
            "essential": bool(int(r["depmap_common_essential"])),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10,
                    help="BPMF rank; must match the export.")
    ap.add_argument("--seed", type=int, default=UMAP_SEED,
                    help="UMAP seed. The default matches the paper figure.")
    ap.add_argument("--out", type=Path, default=DATA_OUT / "gene_umap.json")
    args = ap.parse_args()

    # Imported here, not at module scope: this pulls in umap-learn and the
    # BPMF pickle, and ``--help`` should not need either.
    from assayloop.scripts._bpmf_embedding import (
        ANALYSIS_DIR, load_full_bpmf, load_pathway_labels, run_umap_cosine,
    )
    from assayloop.scripts.pathway_hierarchy import load as load_pathway_hierarchy
    from assayloop.scripts.paper_handoff_composition import (
        _load_complex_family_membership,
    )

    clusters_tsv = _require(
        ANALYSIS_DIR / f"bpmf_k{args.k}_gene_clusters.tsv",
        "the BPMF gene-cluster export",
        f"python -m assayloop.scripts.paper_bpmf_k10_export --k {args.k}")
    genes_meta = load_clusters(clusters_tsv)
    log.info("%d genes in %s", len(genes_meta), clusters_tsv.name)

    # Raises with the glob it searched if the checkpoint is absent.
    V, gene_names = load_full_bpmf(args.k)
    index = {g: i for i, g in enumerate(gene_names)}

    absent = [r["gene"] for r in genes_meta if r["gene"] not in index]
    if absent:
        raise MissingInput(
            f"{len(absent)} gene(s) in {clusters_tsv.name} are not in the "
            f"K={args.k} embedding, e.g. {', '.join(absent[:5])}. The export "
            f"and the checkpoint are from different fits; re-run "
            f"paper_bpmf_k10_export.py against the same one.")

    rows = [index[r["gene"]] for r in genes_meta]
    log.info("running UMAP (cosine, seed=%d) on %d x %d", args.seed,
             len(rows), V.shape[1])
    coords = run_umap_cosine(V[rows], seed=args.seed)

    # Colourings that need their own lookups. Both raise rather than
    # returning an empty map -- an all-grey scatter reads as "no structure",
    # not as "the annotation file was missing".
    gene_to_specific, _ = load_pathway_labels([r["gene"] for r in genes_meta])
    category_of = load_pathway_hierarchy()["category_of"]
    gene_to_pw = {
        gene: category_of[pathway]
        for gene, pathway in gene_to_specific.items()
        if pathway in category_of
    }
    pathway_counts = {}
    for pathway in gene_to_pw.values():
        pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
    top_pathways = sorted(pathway_counts, key=lambda p: (-pathway_counts[p], p))
    # Keyed by UPPERCASE gene, and the value is a *set* of families: a subunit
    # can sit in complexes the keyword matcher maps to more than one family.
    # The figure colours by one, so the site does too -- sorted()[0] rather
    # than set iteration order, so the choice is at least reproducible.
    complex_families = _load_complex_family_membership()
    gene_to_complex = {
        g: sorted(fams)[0] for g, fams in complex_families.items() if fams
    }
    log.info("%d genes with a Reactome pathway, %d with a CORUM family",
             len(gene_to_pw), sum(1 for r in genes_meta
                                  if r["gene"].upper() in gene_to_complex))

    genes = []
    for r, (x, y) in zip(genes_meta, coords, strict=True):
        genes.append({
            **r,
            "x": round(float(x), COORD_DP),
            "y": round(float(y), COORD_DP),
            # Empty string, not null: the page renders it as the "unassigned"
            # legend entry, which is a real category here (most genes are in
            # no small Reactome set at all) rather than missing data.
            "pathway": gene_to_pw.get(r["gene"], ""),
            "complex": gene_to_complex.get(r["gene"].upper(), ""),
        })

    payload = {
        "schema": 1,
        "k": args.k,
        "seed": args.seed,
        "n_genes": len(genes),
        "projection": "UMAP (cosine metric, n_neighbors=30, min_dist=0.3)",
        "source": clusters_tsv.name,
        "notes": {
            "hit_rate": "train_hits / train_screens_present over the public "
                        "training screens.",
            "essential": "DepMap common-essential flag.",
            "cluster_label": "HDBSCAN cluster in the original K-dimensional "
                             "space, named by its dominant phenotype.",
            "pathway": "Top-level Reactome category containing the gene's "
                       "assigned specific pathway; blank if none does.",
            "complex": "CORUM complex family; blank if the gene is in none.",
        },
        "top_pathways": top_pathways,
        "cluster_labels": sorted({g["cluster_label"] for g in genes}),
        "genes": genes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("wrote %s (%d genes, %.1f KB)", args.out, len(genes),
             args.out.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
