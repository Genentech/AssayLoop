"""Shared helpers for reading and projecting a trained BPMF gene embedding.

Used by the Figure 14 script (``paper_bpmf_k10_organization``) and the gene-
cluster export (``paper_bpmf_k10_export``). Kept in one place so the two agree
on which fit they load and how they project it.
"""
from __future__ import annotations

import glob
import logging
import pickle
import re
from collections import Counter
from pathlib import Path

from assayloop import config
from assayloop.data.gene_sets import reactome_gmt_path

log = logging.getLogger("assayloop.scripts.bpmf_embedding")

BPMF_DIR = config.OUTPUT_PATH / "bpmf"
ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"


def load_full_bpmf(k):
    """Load the FULL-DATA BPMF posterior-mean V for K=k.

    Excludes the data-sweep subset fits ``..._d<N>_s<seed>_...`` that also
    match the K glob.

    Raises :class:`~assayloop.config.MissingConfiguredPath` when no such fit
    is on disk. It used to return ``(None, None)``, and both callers turned
    that into ``log.error(...); return`` -- an exit code of 0 with no figure
    written, which a build pipeline cannot tell apart from success.
    """
    full_re = re.compile(r"^bpmf_public_train_K%d_su1_sv1_\d{8}_\d{6}$" % k)
    pattern = str(BPMF_DIR / ("bpmf_public_train_K%d_su1_sv1_*" % k))
    cands = sorted(
        p
        for p in glob.glob(pattern)
        if full_re.match(Path(p).name) and (Path(p) / "bpmf_result.pkl").exists()
    )
    if not cands:
        raise config.MissingConfiguredPath(
            f"No full-data BPMF fit for K={k}. Looked for directories matching "
            f"{pattern} that hold a bpmf_result.pkl (subset fits "
            f"..._d<N>_s<seed>_... are deliberately excluded). Train one with "
            f"scripts/train_bpmf_gpu.py, or point ASSAYLOOP_OUTPUT at a tree "
            f"that has it."
        )
    with open(Path(cands[-1]) / "bpmf_result.pkl", "rb") as f:
        result = pickle.load(f)
    return result.V_samples.mean(axis=0), result.gene_names


def run_pca(X):
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(X)


def run_umap_cosine(X, seed=42, n_neighbors=30, min_dist=0.3):
    """UMAP with cosine metric (the embedding is compared by cosine).

    There used to be a ``t-SNE`` fallback here for when ``umap-learn`` was
    not importable. It is gone: the caller labels the resulting axes "UMAP"
    and the paper's caption says UMAP, so a silent substitution produced a
    figure whose axes were a different algorithm than its label. ``umap-learn``
    is a hard dependency in ``pyproject.toml``, so an ImportError means a
    broken install, and saying so beats drawing a mislabelled panel.
    """
    from umap import UMAP

    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(X)


def short(lab: str, n: int = 32) -> str:
    """Trim a pathway/complex label to fit in a figure legend."""
    lab = lab.replace("Homo sapiens: ", "")
    return lab if len(lab) <= n else lab[: n - 3] + "..."


def load_pathway_labels(gene_names):
    """Assign genes to Reactome pathways (<=150 genes, most specific).

    Reactome only. This used to fall back to MSigDB's ``c2.cp`` filtered to the
    ``REACTOME_`` prefix when the GMT was absent, and then to an empty map when
    that was absent too. Both are silent substitutions: ``c2.cp``'s Reactome
    subset has different set names and different membership, so the cluster
    labels change without the figure looking any different, and an empty map
    labels every gene "unassigned". :func:`reactome_gmt_path` raises instead.
    """
    gmt_path = reactome_gmt_path()

    pathways = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if 10 <= len(genes) <= 150:
                pathways[name] = genes

    gene_to_pw = {}
    for gene in gene_names:
        gu = gene.upper()
        best, best_size = None, float("inf")
        for pn, pg in pathways.items():
            if gu in pg and len(pg) < best_size:
                best, best_size = pn, len(pg)
        if best:
            gene_to_pw[gene] = best

    pw_counts = Counter(gene_to_pw.values())
    top = [p for p, _ in pw_counts.most_common(10)]
    return gene_to_pw, top


__all__ = [
    "BPMF_DIR",
    "ANALYSIS_DIR",
    "load_full_bpmf",
    "load_pathway_labels",
    "run_pca",
    "run_umap_cosine",
    "short",
]
