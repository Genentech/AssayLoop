"""Analyze gene embedding and influence matrices against biological networks.

Evaluates AP/AUROC of recovering STRING, CORUM, and SIGNOR edges from
pairwise gene similarity (embedding cosine or influence score).
Produces a comparison table and paper-ready figures.

Usage::

    uv run python -m assayloop.scripts.analyze_gene_matrices
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from assayloop.scripts._figure_io import save_figure
from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_gene_matrices")

ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"
# STRING / CORUM / SIGNOR parquet tables. Not redistributed with this repo:
# set ASSAYLOOP_GROUND_TRUTH to a directory holding STRING-HUMAN/,
# CORUM-HUMAN/ and SIGNOR-HUMAN/. Every source is required -- a partially
# populated directory would otherwise produce a figure with fewer panels than
# the paper's, without saying so.

# Organize matrices into a training trajectory
INIT_TYPES = [
    ("Random", "random"),
    ("BPMF", "bpmf"),
    ("MF", "mf_raw"),
    ("MF-Sphere", "mf"),
    ("SVD", "svd"),
    ("GenePT", "genept"),
    ("K562", "k562"),
]

# Map init type tag -> (raw_emb, sup_emb, sup_inf, rl_emb, rl_inf) file stems
def _matrix_stems():
    stems = {}
    for label, tag in INIT_TYPES:
        raw_emb = "raw_%s" % tag
        if tag == "bpmf":
            sup_emb = "gf-bpmf-train-hits_model"
            sup_inf = "gf-bpmf-train-hits_model"
            rl_emb = "gf-bpmf-train-hits-rl-fg-s19_model_last"
            rl_inf = "gf-bpmf-train-hits-rl-fg-s19_model_last"
        elif tag == "random":
            raw_emb = "raw_random"
            sup_emb = "gf-random-train-d10_model"
            sup_inf = "gf-random-train-d10_model"
            rl_emb = "gf-random-train-d10-rl_model"
            rl_inf = "gf-random-train-d10-rl_model"
        elif tag == "mf_raw":
            raw_emb = "raw_mf_raw"
            sup_emb = "gf-mf-raw-train-d10_model"
            sup_inf = "gf-mf-raw-train-d10_model"
            rl_emb = "gf-mf-raw-train-d10-rl_model"
            rl_inf = "gf-mf-raw-train-d10-rl_model"
        elif tag == "mf":
            raw_emb = "raw_mf"
            sup_emb = "gf-mf-train-d10_model"
            sup_inf = "gf-mf-train-d10_model"
            rl_emb = "gf-mf-train-d10-rl_model"
            rl_inf = "gf-mf-train-d10-rl_model"
        elif tag == "svd":
            raw_emb = "raw_svd"
            sup_emb = "gf-svd-raw-train-d10_model"
            sup_inf = "gf-svd-raw-train-d10_model"
            rl_emb = "gf-svd-raw-train-d10-rl_model"
            rl_inf = "gf-svd-raw-train-d10-rl_model"
        elif tag == "genept":
            raw_emb = "raw_genept"
            sup_emb = "gf-genept-train-d10_model"
            sup_inf = "gf-genept-train-d10_model"
            rl_emb = "gf-genept-train-d10-rl_model"
            rl_inf = "gf-genept-train-d10-rl_model"
        elif tag == "k562":
            raw_emb = "raw_k562"
            sup_emb = "gf-k562-train-d10_model"
            sup_inf = "gf-k562-train-d10_model"
            rl_emb = "gf-k562-train-d10-rl_model"
            rl_inf = "gf-k562-train-d10-rl_model"
        else:
            continue
        stems[label] = {
            "raw_emb": raw_emb, "sup_emb": sup_emb, "sup_inf": sup_inf,
            "rl_emb": rl_emb, "rl_inf": rl_inf,
        }
    return stems


def _load_matrix(prefix, stem):
    """Load a matrix .npz file. Returns (matrix, gene_names) or (None, None)."""
    path = ANALYSIS_DIR / ("%s_matrix_%s.npz" % (prefix, stem))
    if not path.exists():
        # Try glob for filenames with special characters
        import glob
        matches = sorted(glob.glob(str(ANALYSIS_DIR / ("%s_matrix_%s*.npz" % (prefix, stem)))))
        if matches:
            path = Path(matches[0])
        else:
            return None, None
    d = np.load(path, allow_pickle=True)
    return d["similarity"] if "similarity" in d else d["influence"], d["gene_names"]


def _load_ground_truth():
    """Load STRING, CORUM, SIGNOR and Reactome as {source: {(geneA, geneB)}}.

    Every source is required. A missing one raises and names what to set,
    rather than quietly dropping a panel from the figure.
    """
    gts = {}

    # STRING
    string_dir = config.ground_truth_dir("STRING-HUMAN")
    genes_df = pd.read_parquet(string_dir / "dimension_GeneSymbol.parquet")
    edges_df = pd.read_parquet(string_dir / "dimension_Interactions.parquet")
    id_to_gene = {k: v for k, v in zip(genes_df["string_human_id"],
                                        genes_df["gene_symbol"])
                  if isinstance(v, str)}
    pairs = set()
    for _, row in edges_df.iterrows():
        if row["experiment_score"] < 400:
            continue
        g1 = id_to_gene.get(row["string_human_id_1"])
        g2 = id_to_gene.get(row["string_human_id_2"])
        if g1 and g2 and isinstance(g1, str) and isinstance(g2, str) and g1 != g2:
            pairs.add((min(g1, g2), max(g1, g2)))
    gts["STRING"] = pairs
    log.info("STRING: %d edges", len(pairs))

    # CORUM
    corum_dir = config.ground_truth_dir("CORUM-HUMAN")
    subunits = pd.read_parquet(corum_dir / "dimension_Subunit.parquet")
    complexes = {}
    for _, row in subunits.iterrows():
        gene = row.get("gene_name")
        if gene is None or not isinstance(gene, str) or not gene.strip():
            continue
        complexes.setdefault(row["complex_id"], set()).add(gene.strip().upper())
    pairs = set()
    for members in complexes.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    gts["CORUM"] = pairs
    log.info("CORUM: %d edges from %d complexes", len(pairs), len(complexes))

    # SIGNOR
    signor_dir = config.ground_truth_dir("SIGNOR-HUMAN")
    entities = pd.read_parquet(signor_dir / "Dimension_Entities.parquet")
    facts = pd.read_parquet(signor_dir / "Fact_Protein_Interactions.parquet")
    key_to_gene = {}
    for _, row in entities.iterrows():
        name = row.get("ENTITY_NAME")
        if name and isinstance(name, str) and row.get("TYPE") == "protein":
            key_to_gene[row["ENTITY_KEY"]] = name.upper()
    pairs = set()
    for _, row in facts.iterrows():
        g1 = key_to_gene.get(row["ENTITYA_KEY"])
        g2 = key_to_gene.get(row["ENTITYB_KEY"])
        if g1 and g2 and g1 != g2:
            pairs.add((min(g1, g2), max(g1, g2)))
    gts["SIGNOR"] = pairs
    log.info("SIGNOR: %d edges", len(pairs))

    # Reactome (direct protein-protein interactions, not co-pathway).
    # Downloaded by scripts/fetch_gene_sets.sh; not committed.
    reactome_fi = config.require_path(
        config.GENE_SETS_PATH / "reactome_interactions.txt",
        env_var="ASSAYLOOP_GENE_SETS",
        what="the Reactome interactor table",
        hint="Run scripts/fetch_gene_sets.sh.",
    )
    # UniProt -> gene symbol mapping, built from STRING's tables.
    uniprot_df = pd.read_parquet(string_dir / "dimension_UniprotId.parquet")
    uniprot_to_gene = {}
    for _, row in uniprot_df.iterrows():
        uid = row.get("uniprot_id")
        gene = id_to_gene.get(row.get("string_human_id"))
        if uid and gene:
            uniprot_to_gene["uniprotkb:" + str(uid)] = gene.upper()

    pairs = set()
    with open(reactome_fi) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            g1 = uniprot_to_gene.get(parts[0])
            g2 = uniprot_to_gene.get(parts[3]) if len(parts) > 3 else None
            if g1 and g2 and g1 != g2:
                pairs.add((min(g1, g2), max(g1, g2)))
    if not pairs:
        raise RuntimeError(
            f"No Reactome interactions mapped to gene symbols from {reactome_fi}."
            " The file format may have changed; re-run scripts/fetch_gene_sets.sh."
        )
    gts["REACTOME"] = pairs
    log.info("REACTOME: %d direct interaction edges", len(pairs))

    return gts


def _eval_auroc(matrix, gene_names, positive_pairs):
    """Compute AP and AUROC from a similarity/influence matrix."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    gene_to_idx_upper = {g.upper(): i for i, g in enumerate(gene_names)}
    n = len(gene_names)

    labels = np.zeros((n, n), dtype=np.int32)
    n_pos = 0
    for a, b in positive_pairs:
        ia = gene_to_idx.get(a) or gene_to_idx_upper.get(a)
        ib = gene_to_idx.get(b) or gene_to_idx_upper.get(b)
        if ia is not None and ib is not None:
            labels[ia, ib] = 1
            labels[ib, ia] = 1
            n_pos += 1

    triu = np.triu_indices(n, k=1)
    y_true = labels[triu]
    # Symmetrize the matrix for scoring
    sym = (matrix + matrix.T) / 2
    y_score = sym[triu]

    n_pos_triu = y_true.sum()
    if n_pos_triu < 5:
        return {"ap": 0, "auroc": 0.5, "n_pos": int(n_pos_triu), "n_genes": n}

    return {
        "ap": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "n_pos": int(n_pos_triu),
        "n_genes": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()  # --help / reject stray args; nothing to read back

    gts = _load_ground_truth()
    if not gts:
        log.error("No ground truth loaded")
        return

    stems = _matrix_stems()
    gt_names = sorted(gts.keys())
    out_dir = ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all matrices (with caching)
    cache_file = out_dir / "gene_matrix_analysis_cache.json"
    cache = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
        log.info("Loaded %d cached results", len(cache))

    rows = []
    for init_label, stem_dict in stems.items():
        for stage, (prefix, stem_key) in [
            ("Raw emb.", ("embedding", stem_dict["raw_emb"])),
            ("Sup. emb.", ("embedding", stem_dict["sup_emb"])),
            ("Sup. inf.", ("influence", stem_dict["sup_inf"])),
            ("RL emb.", ("embedding", stem_dict["rl_emb"])),
            ("RL inf.", ("influence", stem_dict["rl_inf"])),
        ]:
            mat_key = "%s|%s" % (prefix, stem_key)

            # Check if all GTs are cached for this matrix
            all_cached = True
            row = {"init": init_label, "stage": stage}
            for gt_name in gt_names:
                per_gt_key = "%s|%s" % (mat_key, gt_name)
                if per_gt_key in cache:
                    row["%s_ap" % gt_name] = cache[per_gt_key]["ap"]
                    row["%s_auroc" % gt_name] = cache[per_gt_key]["auroc"]
                    row["%s_n_pos" % gt_name] = cache[per_gt_key]["n_pos"]
                else:
                    all_cached = False

            if all_cached:
                rows.append(row)
                log.info("%-10s %-12s  %s (cached)", init_label, stage,
                         "  ".join("%s=%.4f" % (g, row.get("%s_auroc" % g, 0))
                                   for g in gt_names))
                continue

            mat, names = _load_matrix(prefix, stem_key)
            if mat is None:
                continue

            row = {"init": init_label, "stage": stage}
            for gt_name in gt_names:
                per_gt_key = "%s|%s" % (mat_key, gt_name)
                if per_gt_key in cache:
                    r = cache[per_gt_key]
                else:
                    r = _eval_auroc(mat, names, gts[gt_name])
                    cache[per_gt_key] = r
                    cache_file.write_text(json.dumps(cache, indent=2))
                row["%s_ap" % gt_name] = r["ap"]
                row["%s_auroc" % gt_name] = r["auroc"]
                row["%s_n_pos" % gt_name] = r["n_pos"]
            rows.append(row)
            log.info("%-10s %-12s  %s", init_label, stage,
                     "  ".join("%s=%.4f" % (g, row.get("%s_auroc" % g, 0))
                               for g in gt_names))

    # Save results
    results_file = out_dir / "gene_matrix_analysis.json"
    results_file.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %s", results_file)

    # Print table
    df = pd.DataFrame(rows)
    print()
    header = "%-10s %-12s" % ("Init", "Stage")
    for gt in gt_names:
        header += "  %s_AP  %s_AUC" % (gt, gt)
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        line = "%-10s %-12s" % (row["init"], row["stage"])
        for gt in gt_names:
            line += "  %6.4f  %6.4f" % (row.get("%s_ap" % gt, 0),
                                          row.get("%s_auroc" % gt, 0))
        print(line)

    # === Figure 1: AUROC comparison across init types and stages ===
    fig, axes = plt.subplots(1, len(gt_names), figsize=(5 * len(gt_names), 6),
                              squeeze=False, sharey=True)

    stage_order = ["Raw emb.", "Sup. emb.", "Sup. inf.", "RL emb.", "RL inf."]
    stage_colors = {"Raw emb.": "#bdc3c7", "Sup. emb.": "#3498db",
                    "Sup. inf.": "#2ecc71", "RL emb.": "#e74c3c",
                    "RL inf.": "#9b59b6"}
    init_labels = [label for label, _ in INIT_TYPES]

    for col, gt in enumerate(gt_names):
        ax = axes[0][col]
        x = np.arange(len(init_labels))
        width = 0.15
        offsets = np.arange(len(stage_order)) - len(stage_order) / 2 + 0.5

        for si, stage in enumerate(stage_order):
            vals = []
            for init_label in init_labels:
                match = [r for r in rows if r["init"] == init_label
                         and r["stage"] == stage]
                vals.append(match[0].get("%s_auroc" % gt, 0.5) if match else 0.5)
            ax.bar(x + offsets[si] * width, vals, width,
                   label=stage if col == 0 else None,
                   color=stage_colors[stage], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(init_labels, fontsize=8, rotation=45, ha="right")
        ax.set_title(gt, fontsize=11, fontweight="bold")
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax.set_ylim(0.4, max(0.7, ax.get_ylim()[1] + 0.02))
        ax.grid(True, axis="y", alpha=0.15)
        if col == 0:
            ax.set_ylabel("AUROC", fontsize=10)

    axes[0][0].legend(fontsize=7, loc="upper left", ncol=1)
    fig.suptitle("Gene Network Recovery: Embedding vs Influence across Training Stages",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / "gene_matrix_auroc_comparison.png"
    save_figure(fig, fname, dpi=150, vector_dpi=300, bbox_inches="tight")
    plt.close(fig)

    # === Figure 2: Training trajectory for BPMF (the main model) ===
    bpmf_rows = [r for r in rows if r["init"] == "BPMF"]
    if bpmf_rows:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        stages = [r["stage"] for r in bpmf_rows]
        x = np.arange(len(stages))
        width = 0.25

        for gi, gt in enumerate(gt_names):
            vals = [r.get("%s_auroc" % gt, 0.5) for r in bpmf_rows]
            ax2.bar(x + (gi - len(gt_names) / 2 + 0.5) * width, vals, width,
                    label=gt)

        ax2.set_xticks(x)
        ax2.set_xticklabels(stages, fontsize=9)
        ax2.set_ylabel("AUROC", fontsize=10)
        ax2.axhline(0.5, color="gray", ls=":", alpha=0.5)
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y", alpha=0.15)
        ax2.set_title("BPMF Init: Network Recovery across Training Stages",
                      fontsize=11, fontweight="bold")
        fig2.tight_layout()
        fname2 = out_dir / "gene_matrix_bpmf_trajectory.png"
        save_figure(fig2, fname2, dpi=150, vector_dpi=300, bbox_inches="tight")
        plt.close(fig2)

    # === Figure 3: Focused network for featured genes ===
    FEATURED = ["MYC", "MDM2", "PIK3CA", "SMAD4", "EGFR", "KRAS",
                "BRAF", "BRCA1", "APC", "ATM", "NRAS", "ERBB2",
                "SMU1", "PRPF4", "EMG1", "PNO1", "VARS1", "PFDN4",
                "RAD18", "NBN", "STIP1",
                "MIS18BP1", "MPHOSPH10", "NDUFS7", "NFU1", "PPDPF",
                "DCP1A", "AMOTL2", "B3GNT7", "PRSS50", "IBA57"]

    # Load the best influence matrix (s19 RL)
    inf_mat, inf_names = _load_matrix("influence",
                                       "gf-bpmf-train-hits-rl-fg-s19_model_last")
    if inf_mat is not None:
        name_to_idx = {g: i for i, g in enumerate(inf_names)}
        featured_idx = [name_to_idx[g] for g in FEATURED if g in name_to_idx]
        featured_names = [inf_names[i] for i in featured_idx]

        sub = inf_mat[np.ix_(featured_idx, featured_idx)].copy()
        np.fill_diagonal(sub, np.nan)

        fig3, ax3 = plt.subplots(figsize=(12, 10))
        vmax = np.nanmax(np.abs(sub))
        im = ax3.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax3.set_xticks(range(len(featured_names)))
        ax3.set_xticklabels(featured_names, fontsize=7, rotation=90)
        ax3.set_yticks(range(len(featured_names)))
        ax3.set_yticklabels(featured_names, fontsize=7)
        ax3.set_xlabel("Target gene (score changes)", fontsize=9)
        ax3.set_ylabel("Probe gene (observed as hit)", fontsize=9)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04,
                    label="Influence (probe → target)")

        # Draw boxes around probe genes (first 12) vs targets (rest)
        n_probes = 12
        ax3.axhline(n_probes - 0.5, color="black", lw=1, alpha=0.3)
        ax3.axvline(n_probes - 0.5, color="black", lw=1, alpha=0.3)

        fig3.tight_layout()
        fname3 = out_dir / "gene_matrix_featured_heatmap.png"
        save_figure(fig3, fname3, dpi=150, vector_dpi=300, bbox_inches="tight")
        plt.close(fig3)


if __name__ == "__main__":
    main()
