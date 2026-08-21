"""Visualize how the Gemini->AssayLoop handoff changes acquisition behavior.

Compares three methods on selected test screens:
  1. Gemini-3.1-Pro (blind LLM)
  2. AssayLoop (pure transformer, s19)
  3. Gemini-3.1-Pro -> AssayLoop Handoff (LLM warm-start then transformer)

For each screen, shows a timeline (acquisition rounds) of how many distinct
biological pathways (Reactome) and protein complexes (CORUM) each method's
HITS cover, revealing how the handoff blends LLM breadth with transformer
exploitation. Example genes are annotated at discovery points.

Usage::

    uv run python -m assayloop.scripts.paper_handoff_timeline
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_handoff_timeline")

RUNS_DIR = config.RESULTS_PATH / "runs"
ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Sweep run-id prefixes for the three methods. The handoff prefix must match the
# n_warm used in full_genome_table.py (n encoded in the sweep_id).
METHOD_PREFIXES = {
    "Gemini-3.1-Pro": "sweep-a79fd5ce",
    "AssayLoop": "sweep-fg-f2-fg-assayloop-s19",
    "Gemini → AssayLoop Handoff": "sweep-fg-f2-fg-handoff-gemini-s19-n3",
}

METHOD_COLORS = {
    "Gemini-3.1-Pro": "#4285F4",          # Google blue
    "AssayLoop": "#EA4335",                # red
    "Gemini → AssayLoop Handoff": "#9334E6",  # purple (blend)
}

# Two contrasting, biologically clean screens
DEFAULT_SCREENS = [
    ("U_1733_merged", "NF-κB / TNF Signaling"),
    ("2466", "AAV Transgene Silencing"),
]

# Screen summary metadata (from the ScreenRecords)
SCREEN_META = {
    "U_1733_merged": {
        "cell_line": "HeLa RelA-mNeonGreen",
        "phenotype": "Molecular Output / Reporter",
        "library": "CRISPRn",
        "num_genes": 18385,
        "total_hits": 169,
        "measures": ("Regulators of TNFα-induced NF-κB activity, read out by "
                     "RelA-mNeonGreen nuclear translocation in HeLa cells "
                     "(under IFN-β)."),
    },
    "2466": {
        "cell_line": "K-562",
        "phenotype": "Host-Pathogen / Molecular Output",
        "library": "CRISPRi",
        "num_genes": 18734,
        "total_hits": 88,
        "measures": ("Genes whose CRISPRi knockdown increases rAAV (AAV2-GFP) "
                     "transgene expression in K-562 cells — de-repressors of "
                     "transgene silencing (HUSH complex) plus Fanconi-anemia / "
                     "SMC5-6 genome maintenance."),
    },
}

CACHE_PATH = ANALYSIS_DIR / "handoff_timeline_cache.json"


def _find_run(prefix, screen):
    """Locate the result.json for a method+screen."""
    for base in (RUNS_DIR, config.SHARED_PATH / "runs"):
        matches = glob.glob(str(base / ("%s-*-%s" % (prefix, screen)) / "result.json"))
        if matches:
            return matches[0]
    return None


def _per_round_genes(result_path):
    """Return list of (acquired_genes, hit_flags) per round."""
    r = json.loads(Path(result_path).read_text())
    rounds = []
    for s in r.get("steps", []):
        rounds.append((s.get("acquired_batch", []), s.get("hits", [])))
    return rounds


def load_method_rounds(screens, refresh=False):
    """Load per-round (genes, hits) for all three methods on all screens.

    Caches the extracted lists to CACHE_PATH so subsequent runs skip the slow
    glob over the shared network runs directory. Returns
    ``{screen: {method: [(genes, hits), ...]}}``.
    """
    cache = {}
    if CACHE_PATH.exists() and not refresh:
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}

    result = {}
    dirty = False
    for screen, _ in screens:
        result[screen] = {}
        cached_screen = cache.get(screen, {})
        for method, prefix in METHOD_PREFIXES.items():
            entry = cached_screen.get(method)
            if entry is not None and not refresh:
                # cached as list of [genes, hits] pairs
                result[screen][method] = [(g, h) for g, h in entry]
                continue
            fp = _find_run(prefix, screen)
            if fp is None:
                log.warning("Missing %s for screen %s", method, screen)
                continue
            log.info("Extracting %s / %s from disk...", screen, method)
            rounds = _per_round_genes(fp)
            result[screen][method] = rounds
            cache.setdefault(screen, {})[method] = [[g, h] for g, h in rounds]
            dirty = True

    if dirty:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))
        log.info("Wrote cache -> %s", CACHE_PATH)
    return result


# Reactome pathways that are biologically uninformative as a "shared function"
# label (broad disease/infection catch-alls, or unrelated to these screens).
# Matched case-insensitively as substrings of the pathway name.
ODD_PATHWAY_PATTERNS = (
    "infection", "sars", "cov-", "covid", "influenza", "hiv", "viral",
    "virus", "bacter", "leishmania", "listeria", "salmonella", "tuberculosis",
    "meiosis", "meiotic", "reproduction", "fertilization",
)


def _is_odd_pathway(name):
    low = name.lower()
    return any(p in low for p in ODD_PATHWAY_PATTERNS)


def _load_pathway_membership():
    """gene (upper) -> set of Reactome pathway names (≤200-gene pathways).

    Broad disease/infection/meiosis pathways are dropped (see
    ODD_PATHWAY_PATTERNS) so they never appear as a composition segment or as a
    shared-function label on an influence arc."""
    gmt = DATA_DIR / "gene_sets" / "ReactomePathways.gmt"
    membership = defaultdict(set)
    if not gmt.exists():
        return membership
    n_dropped = 0
    with open(gmt) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            if _is_odd_pathway(name):
                n_dropped += 1
                continue
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if not (5 <= len(genes) <= 200):
                continue
            for g in genes:
                membership[g].add(name)
    if n_dropped:
        log.info("Dropped %d odd/broad Reactome pathways", n_dropped)
    return membership


def _dedup_complexes(by_cid, thresh=0.7):
    """Collapse near-identical CORUM complexes (option 1 dedup).

    CORUM lists many overlapping complexes that share a core (e.g. CUL4A is in
    21 variant complexes), which inflates any "distinct complexes covered"
    count. We union complexes whose subunit sets have Jaccard >= thresh (only
    comparing complexes that share >=1 gene), then keep the largest complex in
    each cluster as its representative. Returns {rep_cid: gene_set}.
    """
    cids = list(by_cid)
    parent = {c: c for c in cids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    gene_to_cids = defaultdict(list)
    for c, gs in by_cid.items():
        for g in gs:
            gene_to_cids[g].append(c)

    checked = set()
    for cl in gene_to_cids.values():
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                key = (a, b) if a < b else (b, a)
                if key in checked:
                    continue
                checked.add(key)
                ga, gb = by_cid[a], by_cid[b]
                union_sz = len(ga | gb)
                if union_sz and len(ga & gb) / union_sz >= thresh:
                    union(a, b)

    clusters = defaultdict(list)
    for c in cids:
        clusters[find(c)].append(c)

    reps = {}
    for members in clusters.values():
        rep = max(members, key=lambda c: len(by_cid[c]))
        reps[rep] = by_cid[rep]
    return reps


def _load_complex_membership(dedup=False):
    """gene (upper) -> set of CORUM complex names.

    dedup collapses near-identical CORUM complexes (Jaccard>=0.7); off by
    default since it only trims ~11% (CORUM's overlapping complexes are mostly
    genuinely distinct)."""
    import pandas as pd
    # Raises if ASSAYLOOP_GROUND_TRUTH is unset/absent rather than returning
    # an empty membership, which would read as "no complexes recovered".
    corum_dir = config.ground_truth_dir("CORUM-HUMAN")
    membership = defaultdict(set)
    subunits = pd.read_parquet(corum_dir / "dimension_Subunit.parquet")
    try:
        cx = pd.read_parquet(corum_dir / "fact_Complex.parquet")
        cid_to_name = dict(zip(cx.index, cx["complex_name"]))
    except Exception:
        cid_to_name = {}
    by_cid = defaultdict(set)
    for _, row in subunits.iterrows():
        g = row.get("gene_name")
        if isinstance(g, str) and g.strip():
            by_cid[row["complex_id"]].add(g.strip().upper())

    if dedup:
        n_before = len(by_cid)
        reps = _dedup_complexes(by_cid)
        log.info("CORUM dedup: %d -> %d complexes (Jaccard>=0.7)",
                 n_before, len(reps))
        by_cid = reps

    for cid, genes in by_cid.items():
        name = cid_to_name.get(cid, "Complex_%s" % cid)
        for g in genes:
            membership[g].add(name)
    return membership


def _detect_handoff_round(gemini_rounds, handoff_rounds):
    """Round index (1-based) where handoff diverges from Gemini."""
    for i, (gr, hr) in enumerate(zip(gemini_rounds, handoff_rounds)):
        g_genes = gr[0]
        h_genes = hr[0]
        # Overlap fraction; handoff==gemini during warm-start
        if not g_genes or not h_genes:
            continue
        overlap = len(set(g_genes) & set(h_genes)) / max(len(h_genes), 1)
        if overlap < 0.5:
            return i  # first divergent round (0-based -> handoff happens here)
    return len(handoff_rounds)


def _cumulative_coverage(rounds, membership):
    """Cumulative count of distinct groups covered by HIT genes, per round.

    Returns (coverage_per_round, new_by_round) where new_by_round[i] is a list
    of (gene, group) pairs that FIRST covered a new group in round i.
    """
    covered = set()
    coverage = []
    new_by_round = [[] for _ in rounds]
    for i, (genes, hits) in enumerate(rounds):
        for g, h in zip(genes, hits):
            if not h:
                continue
            for grp in membership.get(g.upper(), ()):
                if grp not in covered:
                    covered.add(grp)
                    new_by_round[i].append((g, grp))
        coverage.append(len(covered))
    return coverage, new_by_round


def _shorten_pathway(name):
    """Trim Reactome/CORUM group name for a compact callout."""
    name = name.replace("Homo sapiens: ", "")
    if len(name) > 34:
        name = name[:31] + "…"
    return name


def _pick_callout_genes(new_by_round, round_idx, k=3):
    """Pick up to k representative (gene, group) pairs newly covered at a round."""
    seen_genes = set()
    picks = []
    for g, grp in new_by_round[round_idx] if 0 <= round_idx < len(new_by_round) else []:
        if g in seen_genes:
            continue
        seen_genes.add(g)
        picks.append((g, grp))
        if len(picks) >= k:
            break
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screens", default=None,
                    help="Comma-sep screen names (default: two curated screens)")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore cache and re-extract per-round data from disk")
    args = ap.parse_args()

    if args.screens:
        screens = [(s.strip(), s.strip()) for s in args.screens.split(",")]
    else:
        screens = DEFAULT_SCREENS

    log.info("Loading pathway/complex membership...")
    pw_mem = _load_pathway_membership()
    cx_mem = _load_complex_membership()
    log.info("Pathways: %d genes, Complexes: %d genes", len(pw_mem), len(cx_mem))

    all_rounds = load_method_rounds(screens, refresh=args.refresh)

    out_dir = ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_screens = len(screens)
    fig, axes = plt.subplots(2, n_screens, figsize=(7 * n_screens, 10.5),
                             squeeze=False)

    for col, (screen, screen_label) in enumerate(screens):
        method_rounds = all_rounds.get(screen, {})

        # Detect handoff round
        handoff_round = None
        if "Gemini-3.1-Pro" in method_rounds and "Gemini → AssayLoop Handoff" in method_rounds:
            handoff_round = _detect_handoff_round(
                method_rounds["Gemini-3.1-Pro"],
                method_rounds["Gemini → AssayLoop Handoff"])
            log.info("Screen %s: handoff at round %d", screen, handoff_round + 1)

        for row, (mem, metric_name) in enumerate([
            (pw_mem, "Reactome Pathways"),
            (cx_mem, "CORUM Complexes"),
        ]):
            ax = axes[row][col]

            handoff_new = None
            for method, rounds in method_rounds.items():
                coverage, new_by_round = _cumulative_coverage(rounds, mem)
                # Prepend the (0, 0) origin so curves start from the corner
                x = np.arange(0, len(coverage) + 1)
                y = np.array([0] + list(coverage))
                ax.plot(x, y, marker="o", markersize=4,
                       color=METHOD_COLORS[method], label=method, lw=2)
                if method == "Gemini → AssayLoop Handoff":
                    handoff_new = (coverage, new_by_round)

            # Mark handoff point
            if handoff_round is not None and handoff_round < 10:
                ax.axvline(handoff_round + 0.5, color="#9334E6", ls="--",
                          lw=1.2, alpha=0.6)
                ymax = ax.get_ylim()[1]
                ax.text(handoff_round + 0.5, ymax * 0.03,
                       " handoff (n=%d)" % handoff_round, color="#9334E6",
                       fontsize=7, rotation=90, va="bottom", ha="left")

            # Gene callouts on the handoff curve: what it picks up right after
            # handoff (first transformer round) and at the end.
            if handoff_new is not None and handoff_round is not None:
                coverage, new_by_round = handoff_new
                callout_rounds = sorted({handoff_round, len(coverage) - 1})
                for ridx in callout_rounds:
                    if not (0 <= ridx < len(coverage)):
                        continue
                    picks = _pick_callout_genes(new_by_round, ridx, k=3)
                    if not picks:
                        continue
                    lines = ["+%s  (%s)" % (g, _shorten_pathway(grp))
                             for g, grp in picks]
                    txt = "\n".join(lines)
                    xpos = ridx + 1
                    ypos = coverage[ridx]
                    # offset the box so it doesn't sit on the marker
                    ax.annotate(
                        txt, xy=(xpos, ypos),
                        xytext=(xpos - 2.5 if ridx == len(coverage) - 1 else xpos + 0.3,
                                ypos + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.14),
                        fontsize=6, color="#4A148C",
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#F3E5F5",
                                  ec="#9334E6", lw=0.6, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#9334E6",
                                        lw=0.8, alpha=0.7))

            ax.set_xlabel("Acquisition round (100 genes each)", fontsize=9)
            ax.set_ylabel("Cumulative distinct\n%s covered by hits" % metric_name,
                         fontsize=9)
            ax.set_xticks(range(0, 11))
            ax.set_xlim(0, 10.3)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            if row == 0:
                import textwrap
                meta = SCREEN_META.get(screen, {})
                ax.set_title("%s  (%s)" % (screen_label, screen),
                            fontsize=11, fontweight="bold")
                # Screen summary box: what it measures + key metadata
                if meta:
                    measures = "\n".join(textwrap.wrap(
                        "Measures: " + meta.get("measures", ""), width=48))
                    summary = (
                        "%s\n"
                        "─────────────────────────────\n"
                        "Cell line: %s  •  %s\n"
                        "Library: %s  •  %s genes  •  %d total hits" % (
                            measures, meta["cell_line"], meta["phenotype"],
                            meta["library"], format(meta["num_genes"], ","),
                            meta["total_hits"]))
                    ax.text(0.97, 0.03, summary, transform=ax.transAxes,
                           fontsize=6.3, ha="right", va="bottom",
                           bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA",
                                     ec="#999999", lw=0.6, alpha=0.95))
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("How the LLM→Transformer Handoff Shapes Biological Coverage\n"
                 "Distinct pathways & complexes discovered over the acquisition timeline",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fname = out_dir / "paper_handoff_timeline_annotated.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "paper_handoff_timeline_annotated.pdf", dpi=300,
                bbox_inches="tight")
    log.info("Wrote %s", fname)
    plt.close(fig)


if __name__ == "__main__":
    main()
