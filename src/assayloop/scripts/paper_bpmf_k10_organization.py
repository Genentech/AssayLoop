"""How the BPMF K=10 embedding space is organized (3-row overview).

Same genes, same PCA + cosine-UMAP coordinates, colored three ways:
  row 1: data-driven clusters, labeled by top-associated SCREEN PHENOTYPE
         (binomial z-score of the cluster's genes being hits vs. the screen's
         baseline) -- shows the space is organized by co-essentiality / phenotype.
  row 2: CORUM complexes -- the most cluster-coherent complexes (k-NN enrichment);
         tight molecular machines occupy specific sub-regions.
  row 3: Reactome pathways -- top specific pathways by gene count (the "original"
         annotation coloring), for contrast.

    uv run python -m assayloop.scripts.paper_bpmf_k10_organization
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assayloop import config
from assayloop.scripts._bpmf_embedding import (
    load_full_bpmf, load_pathway_labels, run_pca, run_umap_cosine, short,
)
from assayloop.scripts.paper_handoff_composition import (
    _load_complex_family_membership, COMPLEX_FAMILIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_bpmf_k10_organization")

ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"

# Mechanistic gloss for drug / condition screen tags -> what the screen probes
# (the clusters are defined by screen response, not a shared gene function, so we
# annotate the mechanism rather than invent a pathway).
GLOSS = {
    "JQ1": "BET bromodomain", "Formaldehyde": "aldehyde / crosslink stress",
    "Venetoclax": "BCL-2 / apoptosis", "Apalutamide": "androgen receptor",
    "Enzalutamide": "androgen receptor", "Cisplatin": "Pt / DNA crosslink",
    "Oxaliplatin": "Pt / DNA crosslink", "Gemcitabine": "replication stress",
    "Hydroxyurea": "replication stress", "Doxorubicin": "TOP2 / DNA damage",
    "Etoposide": "TOP2 / DNA damage", "Methotrexate": "antifolate",
    "Thioguanine": "purine antimetabolite", "Cytarabine": "nucleoside analog",
    "Asparaginase": "asparagine depletion", "Aspariginase": "asparagine depletion",
    "Sorafenib": "multikinase inhibitor", "AZD7762": "CHK1 inhibitor",
    "ML-210": "GPX4 / ferroptosis", "Vinorelbine": "microtubule",
    "Pomalidomide": "IMiD / cereblon", "CC-122": "IMiD / cereblon",
    "Avadomide": "IMiD / cereblon", "DNMDP": "PDE3A / apoptosis",
    "blebbistatin": "myosin II", "SARS-CoV-2": "viral host factors",
    "SARS-CoV-1": "viral host factors",
}


def _cluster(E, method, n_clusters, seed, min_cluster_size):
    """Return per-gene cluster labels (-1 = noise/unassigned, HDBSCAN only).

    method: 'kmeans' (fixed n_clusters, convex), or 'hdbscan' (topology-aware on
    the k-NN density -> clusters track the UMAP islands; variable count, leaves
    bridge genes as noise). 'leiden' falls back to kmeans (deps not installed).
    """
    method = method.lower()
    if method == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
            log.info("Clustering with HDBSCAN (min_cluster_size=%d)", min_cluster_size)
            return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10).fit_predict(E)
        except Exception as e:  # noqa: BLE001
            log.warning("HDBSCAN unavailable (%s); using KMeans", e)
    elif method == "leiden":
        log.warning("Leiden (leidenalg/igraph) not installed; using KMeans")
    from sklearn.cluster import KMeans
    log.info("Clustering with KMeans (k=%d)", n_clusters)
    return KMeans(n_clusters=n_clusters, n_init=8, random_state=seed).fit_predict(E)


# Mechanism categories for the (weakly-structured) non-essential clusters. The
# K=10 embedding doesn't resolve single-drug modules, so we label a cluster by
# the dominant drug MECHANISM across its enriched screens, or "Broad drug
# response" when no mechanism clearly dominates.
_CATS = [
    ("DNA-damage response",
     ["cisplatin", "carboplatin", "oxaliplatin", "olaparib", "parp", "rnaseh2",
      "formaldehyde", "gemcitabine", "hydroxyurea", "methanesulfonate", " mms",
      "doxorubicin", "etoposide", "camptothecin", "topotecan", "irinotecan",
      "bleomycin", "mitomycin", "cytarabine", "talazoparib", "niraparib",
      "rucaparib", "radiat", "ionizing", " uv "]),
    ("Apoptosis response",
     ["venetoclax", "navitoclax", "abt-199", "abt-263", "trail", "smac",
      "birinapant", "bcl-2", "bcl2", "mcl-1"]),
    ("Chromatin / transcription",
     ["jq1", "bromodomain", "brd4", "hdac", "vorinostat", "romidepsin", "dnmt",
      "azacit", "decitabine", "ezh2"]),
    ("Immune / viral",
     ["virus", "sars", "influenza", "hiv", "ebola", "enterovirus", "herpes",
      "infection", " il-2", "nk cell", "interferon", "cytokine"]),
]


def _screen_category(s):
    low = ((s.condition_clause or "") + " " + (s.phenotype or "")).lower()
    for name, kws in _CATS:
        if any(w in low for w in kws):
            return name
    if (s.cleaned_phenotype or "").startswith("Host-Pathogen"):
        return "Immune / viral"
    return None


def _screen_tag(s):
    """Short, human-readable theme for a screen (drug, mutant background, or
    phenotype) with dose/ETG codes stripped."""
    cond = s.condition_clause or ""
    m = re.search(r"under (.+?) (?:treatment|selection|infection)", cond, re.I)
    if m:
        return re.sub(r"\s*\(.*", "", m.group(1)).strip()[:26]        # drug/agent
    m = re.search(r"Mutation:\s*([A-Za-z0-9./-]+)", cond)
    if m:
        return "%s-mutant background" % m.group(1)                    # genetic bg
    cat = (s.cleaned_phenotype or s.phenotype or "").strip()
    if cat.startswith("Fitness"):
        return "Fitness"
    if cat.startswith("Host-Pathogen"):
        return "Infection"
    if cat.startswith("Molecular Output"):
        return "Reporter / pathway activity"
    if cat.startswith("Trafficking"):
        return "Trafficking / localization"
    if cat.startswith("Drug"):
        return "Drug response"
    return (cat.split("/")[0].strip() or s.cell_line or "?")[:26]


def _assign_noise(E, cl):
    """Assign HDBSCAN noise (-1) genes to their nearest cluster centroid (cosine),
    keeping the cluster structure but filling the continuum (less grey)."""
    ids = [c for c in sorted(set(int(x) for x in cl)) if c >= 0]
    if not ids:
        return cl
    cent = np.stack([E[cl == c].mean(0) for c in ids])
    cent = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8)
    noise = np.where(cl < 0)[0]
    if len(noise):
        cl = cl.copy()
        cl[noise] = np.array(ids)[(E[noise] @ cent.T).argmax(1)]
    return cl


def _cluster_labels(cl, ids, hits, meas, base, train, min_hits, cat_frac=0.35):
    """Label clusters by their enriched screens (binomial z). Fitness -> tiers;
    otherwise the dominant drug MECHANISM (z-weighted) if one clearly leads, else
    "Broad drug response" (the K=10 embedding doesn't resolve single-drug modules).
    Duplicate labels are merged into one colour downstream."""
    info = {}
    for c in ids:
        m = cl == c
        k = hits[m].sum(0); n = meas[m].sum(0)
        z = (k - n * base) / np.sqrt(np.maximum(n * base * (1 - base), 1e-9))
        z[k < min_hits] = -1e9
        tops = [t for t in np.argsort(-z)[:8] if z[t] > -1e8]
        modal = Counter(_screen_tag(train[t]) for t in tops).most_common(1)
        t0 = tops[0] if tops else int(np.argmax(z))
        rate = float(k[t0] / max(n[t0], 1))
        if modal and modal[0][0] == "Fitness":
            label = "Fitness"
        else:
            # z-weighted mechanism profile over all enriched screens
            sel = np.where(z > 3)[0]
            wt = Counter()
            for t in sel:
                cat = _screen_category(train[t])
                if cat:
                    wt[cat] += float(z[t])
            tot = sum(wt.values())
            if tot > 0 and wt.most_common(1)[0][1] / tot >= cat_frac:
                label = wt.most_common(1)[0][0]
            else:
                label = "Broad drug response"
        info[c] = {"size": int(m.sum()), "label": label, "rate": rate, "z": float(z[t0])}
    # essentiality tiers for fitness clusters (by hit rate)
    fit = sorted([c for c in ids if info[c]["label"] == "Fitness"],
                 key=lambda c: -info[c]["rate"])
    tiers = ["Common essential", "Selective essential", "Context essential"]
    for rank, c in enumerate(fit):
        info[c]["label"] = tiers[min(rank, len(tiers) - 1)]
    # Keep same-mechanism petals as separate clusters: suffix #1, #2, ... by size
    # (largest first) so each cluster stays its own colour instead of merging.
    cnt = Counter(info[c]["label"] for c in ids)
    dup = {l for l, v in cnt.items() if v > 1}
    used = Counter()
    for c in sorted(ids, key=lambda c: -info[c]["size"]):
        if info[c]["label"] in dup:
            used[info[c]["label"]] += 1
            info[c]["label"] = "%s #%d" % (info[c]["label"], used[info[c]["label"]])
    return info


def _panel(ax, xy, up_or_cl, scheme, top, color_of, sizes=None, title=None,
           ylabel=None):
    """Draw one panel. scheme='cluster': up_or_cl is per-gene cluster id array;
    scheme='label': up_or_cl is per-gene upper-name list + `top` label map."""
    if scheme == "cluster":
        clab, order = up_or_cl        # clab = per-gene LABEL string; order = labels
        in_order = np.isin(clab, order)
        ax.scatter(xy[~in_order, 0], xy[~in_order, 1], c="#e5e5e5", s=2,
                   alpha=0.35, rasterized=True)   # uncoloured labels
        for lab in order:
            m = clab == lab
            ax.scatter(xy[m, 0], xy[m, 1], c=[color_of[lab]], s=6, alpha=0.75,
                       rasterized=True, label="%s (%d)" % (lab, top[lab]))
    else:
        up, label_map, top_labels = up_or_cl
        lab_mask = np.array([label_map.get(g) in top_labels for g in up])
        ax.scatter(xy[~lab_mask, 0], xy[~lab_mask, 1], c="#e5e5e5", s=2,
                   alpha=0.35, rasterized=True)
        for lab in top_labels:
            m = np.array([label_map.get(g) == lab for g in up])
            if m.sum() == 0:
                continue
            ax.scatter(xy[m, 0], xy[m, 1], c=[color_of[lab]], s=9, alpha=0.85,
                       rasterized=True, label="%s (%d)" % (short(lab), sizes[lab]))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cluster-method", default="kmeans",
                    choices=["kmeans", "hdbscan", "leiden"],
                    help="kmeans (fixed count) | hdbscan (topology-aware, tracks "
                         "UMAP islands) | leiden (falls back to kmeans).")
    ap.add_argument("--n-clusters", type=int, default=8, help="KMeans cluster count.")
    ap.add_argument("--min-cluster-size", type=int, default=250,
                    help="HDBSCAN min cluster size (smaller = more petals).")
    ap.add_argument("--assign-noise", action="store_true", default=True,
                    help="Assign HDBSCAN noise genes to the nearest cluster "
                         "(less grey in the middle).")
    ap.add_argument("--no-assign-noise", dest="assign_noise", action="store_false")
    ap.add_argument("--top-clusters", type=int, default=12,
                    help="Max clusters to color (largest first); rest -> grey.")
    ap.add_argument("--top-n", type=int, default=10, help="groups colored in CORUM/Reactome rows")
    ap.add_argument("--max-genes", type=int, default=9000)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for clustering + gene subsample (changes clusters).")
    ap.add_argument("--umap-seed", type=int, default=7,
                    help="Separate seed for the UMAP layout only (default 7, the "
                         "paper layout). Change this to reshape UMAP without "
                         "touching the clusters/labels.")
    ap.add_argument("--min-screen-freq", type=int, default=2)
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
    up_full = [g.upper() for g in uni]
    log.info("K=%d: %d universe genes, %d train screens", args.k, len(uni), len(train))

    # hit/measured + baseline (for cluster->phenotype labels)
    ng, ns = len(uni), len(train)
    hits = np.zeros((ng, ns), np.float32); meas = np.zeros((ng, ns), np.float32)
    for si, s in enumerate(train):
        for g, h in zip(s.genes, s.hits):
            j = ui.get(g)
            if j is not None:
                meas[j, si] = 1.0
                if h:
                    hits[j, si] = 1.0
    base = hits.sum(0) / np.maximum(meas.sum(0), 1)

    # row 1: clusters + phenotype labels. Color the largest --top-clusters; the
    # rest (and HDBSCAN noise) are grey.
    cl = _cluster(E, args.cluster_method, args.n_clusters, args.seed, args.min_cluster_size)
    if args.cluster_method == "hdbscan" and args.assign_noise:
        n0 = int((cl < 0).sum())
        cl = _assign_noise(E, cl)
        log.info("assigned %d noise genes to nearest cluster", n0)
    ids = [c for c in sorted(set(int(x) for x in cl)) if c >= 0]
    cinfo = _cluster_labels(cl, ids, hits, meas, base, train, args.min_hits)
    # Merge clusters that share a label into one colour (petals stay visible as
    # same-coloured blobs); grey for any noise (label "").
    lab_of = {c: cinfo[c]["label"] for c in ids}
    clab_full = np.array([lab_of.get(int(c), "") for c in cl])
    size_by_label = Counter(l for l in clab_full.tolist() if l)
    order = [l for l, _ in size_by_label.most_common(args.top_clusters)]
    log.info("%s -> %d clusters, %d merged labels (coloring %d), %d noise",
             args.cluster_method, len(ids), len(size_by_label), len(order),
             int((cl < 0).sum()))
    for l in order:
        log.info("  %-40s n=%d", l, size_by_label[l])

    # row 2: CORUM COARSE complex families (broad coverage, many colored genes)
    cxm = _load_complex_family_membership()             # gene(upper) -> {family}
    famp = {f: i for i, (f, _) in enumerate(COMPLEX_FAMILIES)}
    cx_labels = {g: min(v, key=lambda f: famp.get(f, 999)) for g, v in cxm.items() if v}
    size_cx = Counter(cx_labels[g] for g in up_full if g in cx_labels)
    top_cx = [c for c, _ in size_cx.most_common(args.top_n)]

    # row 3: Reactome specific pathways, top by gene count (matches the original viz)
    pw_raw, top_pw = load_pathway_labels(uni)
    pw_labels = {g.upper(): v for g, v in pw_raw.items()}
    top_pw = top_pw[:args.top_n]
    size_pw = Counter(pw_labels[g] for g in up_full if g in pw_labels)

    # shared projection: KEEP all annotation-labeled genes (so the CORUM/Reactome
    # rows are densely colored), then fill with random genes up to --max-genes.
    rng = np.random.default_rng(args.seed)
    labeled = {i for i, g in enumerate(up_full)
               if cx_labels.get(g) in top_cx or pw_labels.get(g) in top_pw}
    fill_pool = np.array([i for i in range(len(uni)) if i not in labeled])
    n_fill = max(args.max_genes - len(labeled), 0)
    if n_fill and len(fill_pool):
        labeled |= set(rng.choice(fill_pool, min(n_fill, len(fill_pool)),
                                  replace=False).tolist())
    keep = np.array(sorted(labeled))
    Xk = E[keep]
    umap_seed = args.umap_seed if args.umap_seed is not None else args.seed
    coords = {"PCA": run_pca(Xk), "UMAP": run_umap_cosine(Xk, seed=umap_seed)}
    clab_keep = clab_full[keep]
    upk = [up_full[i] for i in keep]
    log.info("Projected %d genes (PCA + cosine UMAP)", len(keep))

    # colors (order = largest merged labels)
    ccmap = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    ccol = {lab: ccmap(i % ccmap.N) for i, lab in enumerate(order)}
    t10 = plt.get_cmap("tab10")
    cxcol = {l: t10(i / max(len(top_cx), 1)) for i, l in enumerate(top_cx)}
    pwcol = {l: t10(i / max(len(top_pw), 1)) for i, l in enumerate(top_pw)}

    fig, axes = plt.subplots(3, 2, figsize=(13, 14.5), dpi=200)
    fig.patch.set_facecolor("white")
    rows = [
        ("Screen phenotype (clusters)", "cluster", (clab_keep, order), size_by_label, None),
        ("CORUM complex", "label", (upk, cx_labels, top_cx), (cx_labels, top_cx), size_cx),
        ("Reactome pathway", "label", (upk, pw_labels, top_pw), (pw_labels, top_pw), size_pw),
    ]
    for ri, (rlabel, scheme, data, top, sizes) in enumerate(rows):
        color_of = ccol if scheme == "cluster" else (cxcol if ri == 1 else pwcol)
        for ci, method in enumerate(["PCA", "UMAP"]):
            _panel(axes[ri][ci], coords[method], data, scheme, top, color_of,
                   sizes=sizes, title=(method if ri == 0 else None),
                   ylabel=(rlabel if ci == 0 else None))
        h, l = axes[ri][0].get_legend_handles_labels()
        if h:
            axes[ri][1].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5),
                               fontsize=11.5, markerscale=2.4, frameon=False,
                               labelspacing=0.6, handletextpad=0.5)

    fig.tight_layout()
    tag = ("kmeans%d" % args.n_clusters if args.cluster_method == "kmeans"
           else args.cluster_method)
    out = ANALYSIS_DIR / ("paper_bpmf_k%d_organization_%s.png" % (args.k, tag))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    log.info("Wrote %s", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
