"""Export the BPMF K=10 gene clusters + screen metadata for external analysis.

Reproduces the K=10 HDBSCAN clustering used in paper_bpmf_k10_organization and
writes two TSVs:

  bpmf_k10_gene_clusters.tsv   one row per universe gene:
      gene, cluster_id, cluster_label, train_screens_present, train_hits,
      hit_rate, depmap_common_essential
  bpmf_k10_screen_metadata.tsv one row per training screen:
      dataset_name, cleaned_phenotype, phenotype, cell_line, cell_type,
      condition_clause, direction, num_genes, total_hits,
      top_cluster_label, top_cluster_z

    uv run python -m assayloop.scripts.paper_bpmf_k10_export
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np

from assaybench.benchmark.sequential import load_common_essentials
from assayloop import config
from assayloop.scripts._bpmf_embedding import load_full_bpmf
from assayloop.scripts.paper_bpmf_k10_organization import (
    _cluster, _assign_noise, _cluster_labels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_bpmf_k10_export")

ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"


def _load_essentials():
    """DepMap common essentials, from the copy bundled with assaybench.

    Previously read a CSV at the repo root and warned-then-continued when it
    was missing, which exported a ``depmap_common_essential`` column that was
    all-false rather than absent. The packaged loader raises.
    """
    return {g.upper() for g in load_common_essentials()}


def _direction(s):
    ph = (s.phenotype or "").lower()
    if "increase" in ph and "resist" in ph:
        return "resistance"
    if "decrease" in ph or "sensiti" in ph or "reduce" in ph:
        return "sensitizer"
    return "unclear"


def _clean(x):
    return re.sub(r"\s+", " ", str(x or "")).replace("\t", " ").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cluster-method", default="hdbscan")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--min-cluster-size", type=int, default=150)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-screen-freq", type=int, default=2)
    ap.add_argument("--out-dir", default=str(ANALYSIS_DIR))
    args = ap.parse_args()

    from assayloop.tasks import load_screens
    V, genes = load_full_bpmf(args.k)
    if V is None:
        log.error("Full-data BPMF K=%d not found", args.k)
        return
    gi = {g: i for i, g in enumerate(genes)}
    train = load_screens(target_set="public_train")
    pub = load_screens(target_set="public")
    freq = Counter()
    for s in pub:
        for g in set(s.genes):
            freq[g] += 1
    uni = [g for g in genes if freq.get(g, 0) >= args.min_screen_freq]
    ui = {g: i for i, g in enumerate(uni)}
    E = V[[gi[g] for g in uni]]
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)

    # hit / measured matrices over training screens
    ng, ns = len(uni), len(train)
    hits = np.zeros((ng, ns), np.float32)
    meas = np.zeros((ng, ns), np.float32)
    for si, s in enumerate(train):
        for g, h in zip(s.genes, s.hits):
            j = ui.get(g)
            if j is not None:
                meas[j, si] = 1.0
                if h:
                    hits[j, si] = 1.0
    base = hits.sum(0) / np.maximum(meas.sum(0), 1)

    # cluster + label (same pipeline as the figure)
    cl = _cluster(E, args.cluster_method, args.n_clusters, args.seed, args.min_cluster_size)
    if args.cluster_method == "hdbscan":
        cl = _assign_noise(E, cl)
    ids = [c for c in sorted(set(int(x) for x in cl)) if c >= 0]
    cinfo = _cluster_labels(cl, ids, hits, meas, base, train, args.min_hits)
    log.info("%d clusters", len(ids))

    # per-cluster z per screen -> each screen's best-associated cluster
    Z = np.full((len(ids), ns), -np.inf)
    for r, c in enumerate(ids):
        m = cl == c
        k = hits[m].sum(0); n = meas[m].sum(0)
        z = (k - n * base) / np.sqrt(np.maximum(n * base * (1 - base), 1e-9))
        z[k < args.min_hits] = -np.inf
        Z[r] = z
    best_r = Z.argmax(0)
    best_z = Z.max(0)

    ess = _load_essentials()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gene -> cluster ----
    scr_present = meas.sum(1).astype(int)
    scr_hits = hits.sum(1).astype(int)
    gp = out_dir / "bpmf_k10_gene_clusters.tsv"
    with open(gp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene", "cluster_id", "cluster_label", "train_screens_present",
                    "train_hits", "hit_rate", "depmap_common_essential"])
        for i, g in enumerate(uni):
            c = int(cl[i])
            lab = cinfo[c]["label"] if c in cinfo else "unassigned"
            rate = scr_hits[i] / scr_present[i] if scr_present[i] else 0.0
            w.writerow([g, c, lab, scr_present[i], scr_hits[i], "%.4f" % rate,
                        int(g.upper() in ess)])
    log.info("wrote %s (%d genes)", gp, len(uni))

    # ---- screen metadata ----
    sp = out_dir / "bpmf_k10_screen_metadata.tsv"
    with open(sp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset_name", "cleaned_phenotype", "phenotype", "cell_line",
                    "cell_type", "condition_clause", "direction", "num_genes",
                    "total_hits", "top_cluster_label", "top_cluster_z"])
        for si, s in enumerate(train):
            c = ids[best_r[si]]
            lab = cinfo[c]["label"] if np.isfinite(best_z[si]) else "none"
            z = "%.1f" % best_z[si] if np.isfinite(best_z[si]) else ""
            w.writerow([_clean(s.dataset_name), _clean(s.cleaned_phenotype),
                        _clean(s.phenotype), _clean(s.cell_line), _clean(s.cell_type),
                        _clean(s.condition_clause), _direction(s), len(s.genes),
                        int(sum(s.hits)), lab, z])
    log.info("wrote %s (%d screens)", sp, len(train))

    # cluster summary to the log for convenience
    for c in sorted(ids, key=lambda c: -cinfo[c]["size"]):
        log.info("  cluster %d: %-32s n=%d", c, cinfo[c]["label"], cinfo[c]["size"])


if __name__ == "__main__":
    main()
