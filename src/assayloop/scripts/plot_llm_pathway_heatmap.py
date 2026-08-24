"""Do different LLMs go after different biology? -- enrichment heatmap.

The sunburst contrasts method *classes*, where the differences are large. This
figure asks the finer question the sunburst can't show: given that every LLM
covers a lot of pathway space, do they cover *different* space? Rows are
Reactome top-level categories, columns are methods, and the cell color is
log2(method's share / random's share) -- red = over-represented relative to a
uniform batch, blue = under. The number in each cell is the raw share, so no
cell relies on color alone and the capped scale loses nothing.

Three learned/heuristic rankers are kept on the right as non-LLM anchors; their
enrichments run several times larger than any LLM's, which is the point. The
strip underneath carries the three axes where LLMs *do* differ: how broadly
they spread (effective pathways), how many distinct genes they ever touch, and
how well-annotated -- i.e. how canonical -- their picks are.

Attribution is identical to the sunburst: weight 1 per acquired gene split
evenly across its pathways in the 5-200 gene disease-filtered Reactome GMT,
pooled over the 20 held-out full-genome screens. "Random" is the annotated gene
universe, i.e. the expected composition of a uniformly drawn batch.

Aggregated once into ``output/analysis/llm_pathway_heatmap_data.json``; pass
--refresh to rebuild.

Usage::

    uv run python -m assayloop.scripts.plot_llm_pathway_heatmap [--refresh]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from assayloop.scripts._figure_io import save_figure
from assayloop import config
from assayloop.scripts.pathway_hierarchy import load as load_hierarchy
from assayloop.scripts.plot_pathway_sunburst import (
    _effective_n, _gmt_membership, _rarefied_eff, _weights,
)

# Runs are looked up locally first, then in the optional shared directory
# (ASSAYLOOP_RESULTS / ASSAYLOOP_SHARED_PATH).
RUN_DIRS = [
    config.RESULTS_PATH / "runs",
    config.SHARED_PATH / "runs",
]
ANALYSIS = config.OUTPUT_PATH / "analysis"
CACHE = ANALYSIS / "llm_pathway_heatmap_data.json"
OUT = ANALYSIS / "llm_pathway_heatmap.png"

# (column label, run-dir prefix, group). Sweep ids verified against the EF/nAUC
# in tab:baselines_results before use. "Random" is synthesised, not run.
METHODS = [
    ("Random",          None,                           "ref"),
    ("GPT-5.6 Sol",     "sweep-07e318e9",               "llm"),
    ("Gemini-3.1-Pro",  "sweep-a79fd5ce",               "llm"),
    ("GLM-5.1",         "sweep-e0dd3203",               "llm"),
    ("Claude Opus-4.8", "sweep-5e3d4dbc",               "llm"),
    ("Kimi-K2.6",       "sweep-3c93a0fa",               "llm"),
    ("ICBR-EF",         "sweep-210a554b",               "meta"),
    ("Haystack",        "sweep-f9ae7231",               "search"),
    ("kNN",             "sweep-fg-f2-fg-knn",           "ranker"),
    ("BPMF",            "sweep-fg-f2-fg-bpmf",          "ranker"),
    ("AssayFormer",     "sweep-fg-f2-fg-assayloop-s19", "ranker"),
]
# section names follow tab:baselines_results
GROUPS = ("llm", "meta", "search", "ranker")
GROUP_LABEL = {"ref": "", "llm": "Base LLMs", "meta": "Meta", "search": "Search",
               "ranker": "Learned rankers"}

N_ROWS = 12      # categories shown, ranked by pooled weight
CAP = 1.0        # |log2 ratio| the color scale saturates at (0.5x .. 2x)

SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
RULE = "#c3c2b7"
# diverging blue <-> red about the documented neutral midpoint. Blue arm is the
# palette's sequential ramp (steps 650/400/100); the red arm mirrors its
# lightness off categorical slot 8.
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [
    (0.00, "#104281"), (0.17, "#3987e5"), (0.38, "#cde2fb"),
    (0.50, "#f0efec"),
    (0.62, "#f7c4c3"), (0.83, "#e34948"), (1.00, "#8f2321"),
])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _run_genes(prefix: str) -> list[str]:
    """All acquired genes for a sweep, from whichever run root holds it."""
    for root in RUN_DIRS:
        dirs = sorted(root.glob(f"{prefix}-[0-9][0-9]-*"))
        if not dirs:
            continue
        genes = []
        for rd in dirs:
            fp = rd / "result.json"
            if not fp.is_file():
                continue
            r = json.loads(fp.read_text())
            for step in r.get("steps", []):
                genes.extend(step.get("acquired_batch", []))
        if genes:
            return genes
    return []


def build_cache() -> dict:
    hier = load_hierarchy()
    cat_of, sub_of = hier["category_of"], hier["subcategory_of"]
    membership = _gmt_membership()          # leaf sets -- the category shares
    # Level-2 vocabulary for the EP number, so it matches the sunburst and
    # tab:baselines_results. See plot_pathway_sunburst for why the two differ.
    from assayloop.metrics.effective_pathways import gmt_membership
    ep_membership = gmt_membership()

    out = {}
    for label, prefix, _grp in METHODS:
        if prefix is None:
            genes = sorted(membership)
        else:
            genes = _run_genes(prefix)
            if not genes:
                raise SystemExit(f"no cached runs for {label!r} ({prefix}-NN-*)")
        by_cat, _by_sub, _sc, by_path, n_ann, n_tot = _weights(
            genes, membership, cat_of, sub_of)
        total = sum(by_cat.values())
        out[label] = {
            "share": {c: w / total for c, w in by_cat.items()},
            # rarefied, matching the sunburst and tab:baselines_results
            "eff_pathways": _rarefied_eff(genes, ep_membership),
            "eff_pathways_raw": _effective_n(by_path.values()),
            "n_picks": n_tot, "n_annotated": n_ann,
            "n_unique": len({g.upper() for g in genes}),
        }
        print(f"{label:16s} {n_tot:6d} picks  {n_ann / max(n_tot, 1):5.1%} annotated  "
              f"{out[label]['n_unique']:6d} unique  eff={out[label]['eff_pathways']:6.0f}")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(data: dict) -> None:
    cols = [m[0] for m in METHODS]
    ref = data["Random"]["share"]

    pooled: dict[str, float] = defaultdict(float)
    for d in data.values():
        for c, w in d["share"].items():
            pooled[c] += w
    rows = [c for c, _ in sorted(pooled.items(), key=lambda kv: -kv[1])][:N_ROWS]

    share = np.array([[data[m]["share"].get(c, 0.0) for m in cols] for c in rows])
    with np.errstate(divide="ignore"):
        lg = np.log2(share / np.array([ref.get(c, np.nan) for c in rows])[:, None])
    lg[:, 0] = np.nan                                   # the reference column

    fig = plt.figure(figsize=(7.6, 5.5), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(rows), 3.4], hspace=0.06,
                          left=0.175, right=0.995, top=0.80, bottom=0.075)
    ax, axs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    ax.imshow(np.clip(lg, -CAP, CAP), cmap=DIVERGING,
              norm=TwoSlopeNorm(vmin=-CAP, vcenter=0.0, vmax=CAP),
              aspect="auto", interpolation="nearest")

    # every cell carries its own share, so identity is never color-alone and
    # the capped scale hides nothing; ink or white by the fill's luminance
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = lg[i, j]
            if np.isnan(v):
                fc, txt = INK2, f"{share[i, j] * 100:.1f}"
            else:
                r, g, b, _ = DIVERGING((np.clip(v, -CAP, CAP) + CAP) / (2 * CAP))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                fc = INK if lum > 0.55 else "#ffffff"
                txt = f"{share[i, j] * 100:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4, color=fc,
                    fontweight="bold" if j == 0 else "normal")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7.6, color=INK2, rotation=38,
                       ha="left", rotation_mode="anchor")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.4, color=INK2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # hairline gaps between cells, and rules between method groups
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.4)
    ax.tick_params(which="minor", length=0)

    groups = [m[2] for m in METHODS]
    edges = [j for j in range(1, len(cols)) if groups[j] != groups[j - 1]]
    for j in edges:
        ax.axvline(j - 0.5, color=RULE, lw=1.0, zorder=5)
    # one group caption per span, above the rotated column labels; x in data
    # coords, y in axes fraction so the captions clear the tallest label
    tr = ax.get_xaxis_transform()
    for g in GROUPS:
        idx = [j for j, gg in enumerate(groups) if gg == g]
        ax.text(float(np.mean(idx)), 1.235, GROUP_LABEL[g], ha="center",
                va="bottom", fontsize=7.6, color=MUTED, transform=tr,
                clip_on=False)
        ax.plot([idx[0] - 0.4, idx[-1] + 0.4], [1.225, 1.225], transform=tr,
                color=RULE, lw=0.8, clip_on=False)

    # ---- summary strip: the axes where the LLMs actually differ ----
    axs.set_xlim(-0.5, len(cols) - 0.5)
    axs.set_ylim(3, 0)
    axs.axis("off")
    strip = [
        ("Effective pathways", lambda d: f"{d['eff_pathways']:.1f}"),
        ("Unique genes", lambda d: f"{d['n_unique']:,}"),
        ("Annotated picks", lambda d: f"{d['n_annotated'] / max(d['n_picks'], 1):.0%}"),
    ]
    for i, (name, fmt) in enumerate(strip):
        y = i + 0.5
        axs.text(-0.62, y, name, ha="right", va="center", fontsize=7.4, color=INK2)
        for j, m in enumerate(cols):
            axs.text(j, y, fmt(data[m]), ha="center", va="center", fontsize=6.8,
                     color=INK2 if j else INK,
                     fontweight="bold" if j == 0 else "normal")
        axs.axhline(i, color="#e6e5de", lw=0.8, xmin=0.0, xmax=1.0)
    for j in edges:
        axs.axvline(j - 0.5, color=RULE, lw=1.0)

    # ---- colorbar ----
    cax = fig.add_axes((0.175, 0.012, 0.30, 0.016))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=TwoSlopeNorm(vmin=-CAP, vcenter=0.0, vmax=CAP), cmap=DIVERGING),
        cax=cax, orientation="horizontal")
    cb.set_ticks([-CAP, 0, CAP])
    cb.set_ticklabels(["0.5x", "random", "2x"])
    cb.ax.tick_params(labelsize=6.6, colors=MUTED, length=0)
    cb.outline.set_visible(False)
    fig.text(0.49, 0.020, "share of picks relative to a uniform batch "
             "(capped; cell values are the raw %)", fontsize=6.6, color=MUTED,
             ha="left", va="center")

    save_figure(fig, OUT, vector_dpi=300, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-aggregate from run results")
    a = ap.parse_args()
    data = build_cache() if (a.refresh or not CACHE.is_file()) \
        else json.loads(CACHE.read_text())
    draw(data)


if __name__ == "__main__":
    main()
