"""What biology does each method actually go after? -- two-level pathway sunbursts.

One sunburst per method. Inner ring = Reactome top-level category (Signal
Transduction, Metabolism, Immune System, ...); outer ring = Reactome's level-2
groups inside that category (Immune System -> Innate / Adaptive / Cytokine
Signaling, ...), drawn as alternating tints of the parent hue. A method that
concentrates on a few programs shows a couple of fat wedges; a method that
spreads shows a finely divided ring. The number in the hole is the effective
number of pathways, exp(Shannon entropy) over the pathway distribution -- the
same spread the rings show, as one number.

Two Reactome granularities are in play and they are deliberately different. The
*rings* are drawn from the GMT's leaf sets, rolled up here into the two drawn
tiers, because that is what gives the outer ring its resolution. The *numbers*
are the EP statistic of :mod:`assayloop.metrics.effective_pathways`, which
counts over the 186 level-2 groups -- exactly the tier the outer ring shows.
So the hole and footer are a faithful summary of the outer ring, and they match
tab:baselines_results digit for digit.

Attribution: every acquired gene carries weight 1, split evenly across the
Reactome pathways it belongs to (the 5-200 gene, disease-filtered GMT of
`paper_handoff_timeline`), and each pathway's weight lands in its top-level
category. Genes with no annotation are dropped; the annotated fraction is
reported per method. The "Random" panel is the gene-universe composition, i.e.
the expected composition of a uniformly drawn batch.

Each panel reports the same metric at all three scopes of
:mod:`assayloop.metrics.effective_pathways`, matching tab:baselines_results:
EP-D in the hole (all picks pooled), EP-B and EP-S in the footer (per batch and
per screen, averaged). All three are rarefied to fixed annotated-gene counts, so
the scopes are *not* comparable to one another -- only down a column.

Matching the table means reading each run the way the table reads it, which is
not the same for every panel; :func:`_run_screen_batches` is the single place
that knows the difference, and the LLM pathway heatmap shares it. The Random
panel is the one deliberate exception: the table's Random row is one realised
uniform-draw sweep, while a reference panel should not inherit that draw's
luck, so here it is the expectation over draws (see
:func:`_random_ep_expectation`).

Data comes from the cached full-genome run results (steps[].acquired_batch),
20 test screens x 1000 picks per method. Aggregated once into
``output/analysis/pathway_sunburst_data.json``; pass --refresh to rebuild.

Usage::

    uv run python -m assayloop.scripts.plot_pathway_sunburst [--refresh]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from assayloop import config
from assayloop.llm.replay import load_logged_response_texts, replay_llm_steps
from assayloop.scripts._figure_io import save_figure
from assayloop.scripts.pathway_hierarchy import load as load_hierarchy

RUN_DIRS = [
    config.RESULTS_PATH / "runs",
    config.SHARED_PATH / "runs",
    config.PUBLISHED_PATH / "runs",
]
ANALYSIS = config.OUTPUT_PATH / "analysis"
CACHE = ANALYSIS / "pathway_sunburst_data.json"
OUT = ANALYSIS / "pathway_sunburst.png"

BUDGET, BATCH_SIZE, MIN_SCREEN_FREQ = 1000, 100, 2
N_RANDOM_DRAWS = 24
RANDOM_DRAW_SEED = 10_000

# (panel title, run-dir glob prefix, how the results table reads the run)
# "Random" is synthesised from the gene universe, not from runs.
METHODS = [
    ("Random",                    None,                                    None),
    ("Screen-kNN",                "sweep-fg-f2-fg-knn",                   "greedy"),
    ("BPMF",                      "sweep-fg-f2-fg-bpmf",                  "greedy"),
    ("Gemini-3.1-Pro",            "sweep-a79fd5ce",                       "replay"),
    ("AssayFormer",               "sweep-fg-f2-fg-assayloop-s19",         "greedy"),
    ("AssayLoop (Gemini handoff)", "sweep-fg-f2-fg-handoff-gemini-s19-n3", "greedy"),
]

N_CATS = 8          # inner-ring categories kept; the tail folds into "Other"
N_SUB = 6           # outer-ring level-2 groups drawn individually per category
MIN_SUB = 0.010     # ...and only if the group holds this share of the panel
LABEL_MIN = 0.06    # direct-label an inner wedge at >= this share

SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
# Validated categorical slot order (dataviz palette, light surface)
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
        "#4a3aa7", "#e34948"]
OTHER = "#a9a79f"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _gmt_membership() -> dict[str, list[str]]:
    """gene -> *leaf* pathways, the 5-200 gene filtered GMT the PO metric uses.

    Deliberately the leaf tier, not the level-2 vocabulary the EP columns score
    over: this module rolls leaves up to the level-2 and top-level rings itself
    via ``cat_of``/``sub_of``, so it needs leaf names to map from.
    """
    from assayloop.metrics.effective_pathways import leaf_membership
    return {g: list(ps) for g, ps in leaf_membership().items()}


def _weights(genes, membership, cat_of, sub_of):
    """Fractional weights for a bag of gene picks, at all three levels."""
    by_path: dict[str, float] = defaultdict(float)
    n_ann = 0
    for g in genes:
        ps = membership.get(g.upper())
        if not ps:
            continue
        n_ann += 1
        w = 1.0 / len(ps)
        for p in ps:
            by_path[p] += w
    by_cat: dict[str, float] = defaultdict(float)
    by_sub: dict[str, float] = defaultdict(float)
    sub_cat: dict[str, str] = {}
    for p, w in by_path.items():
        c = cat_of.get(p, "Other")
        s = sub_of.get(p, p)
        by_cat[c] += w
        by_sub[s] += w
        sub_cat[s] = c
    return dict(by_cat), dict(by_sub), sub_cat, dict(by_path), n_ann, len(genes)


_UNIVERSE: list[str] | None = None


def _f2_universe() -> list[str]:
    """Build the same multi-screen acquisition universe as the table."""
    global _UNIVERSE
    if _UNIVERSE is None:
        from assayloop.tasks import load_screens

        screens = load_screens(target_set="public")
        freq: Counter = Counter()
        for screen in screens:
            for gene in set(screen.genes):
                freq[gene] += 1
        _UNIVERSE = sorted(
            gene for gene, count in freq.items() if count >= MIN_SCREEN_FREQ
        )
    return _UNIVERSE


def _random_ep_expectation(ep_membership) -> dict:
    """Return mean EP and its spread across uniform full-genome sweeps."""
    from assayloop.metrics.effective_pathways import effective_pathways
    from assayloop.tasks import load_screens

    n_screens = len(load_screens(target_set="public"))
    universe = np.array(_f2_universe())
    rows = []
    for rep in range(N_RANDOM_DRAWS):
        rng = np.random.default_rng(RANDOM_DRAW_SEED + rep)
        screens = []
        for _ in range(n_screens):
            pick = universe[rng.choice(len(universe), size=BUDGET, replace=False)]
            screens.append([
                list(pick[i:i + BATCH_SIZE])
                for i in range(0, BUDGET, BATCH_SIZE)
            ])
        ep = effective_pathways(screens, membership=ep_membership)
        rows.append([ep["ep_batch"], ep["ep_screen"], ep["ep_dataset"]])

    values = np.array(rows, dtype=float)
    out = {}
    for index, scope in enumerate(("ep_batch", "ep_screen", "ep_dataset")):
        out[scope] = float(values[:, index].mean())
        out[f"{scope}_sd"] = float(values[:, index].std(ddof=1))
    return out


def _run_dirs(prefix: str) -> list[Path]:
    """Find all completed cached runs for one sweep."""
    for root in RUN_DIRS:
        dirs = [
            path
            for path in sorted(root.glob(f"{prefix}-[0-9][0-9]-*"))
            if (path / "result.json").is_file()
        ]
        if dirs:
            return dirs
    return []


def _run_screen_batches(
    prefix: str, *, replay: bool = False, min_batch: int = 0
) -> list[list[list[str]]]:
    """Return ``screen -> batch -> genes`` as the table scores each run.

    Direct gene-list LLMs are replayed from their unabridged logged response
    against the shared f2 universe. Greedy scorers use their stored batches.
    """
    out = []
    for run_dir in _run_dirs(prefix):
        steps = json.loads((run_dir / "result.json").read_text()).get("steps", [])
        if replay:
            batches = replay_llm_steps(
                steps,
                _f2_universe(),
                batch_size=BATCH_SIZE,
                response_texts=load_logged_response_texts(
                    run_dir, expected_steps=len(steps)
                ),
            )
        else:
            batches = [step.get("acquired_batch") or [] for step in steps]
        out.append([batch for batch in batches if len(batch) >= min_batch])
    return out


def build_cache() -> dict:
    hier = load_hierarchy()
    cat_of, sub_of = hier["category_of"], hier["subcategory_of"]
    membership = _gmt_membership()          # leaf sets -- the rings
    # ...and the level-2 vocabulary the table scores over -- the numbers.
    from assayloop.metrics.effective_pathways import (
        effective_pathways, gmt_membership)
    ep_membership = gmt_membership()

    out = {}
    for title, prefix, read in METHODS:
        if prefix is None:
            genes = sorted(membership)
            ep = _random_ep_expectation(ep_membership)
        else:
            batches = _run_screen_batches(
                prefix,
                replay=read == "replay",
                min_batch=0 if read == "replay" else 2,
            )
            genes = [gene for screen in batches for batch in screen for gene in batch]
            if not genes:
                raise SystemExit(f"no cached runs for {title!r} ({prefix}-NN-*)")
            ep = effective_pathways(batches, membership=ep_membership)
        ep_b, ep_s, ep_d = ep["ep_batch"], ep["ep_screen"], ep["ep_dataset"]
        by_cat, by_sub, sub_cat, by_path, n_ann, n_tot = _weights(
            genes, membership, cat_of, sub_of)
        out[title] = {
            "by_cat": by_cat, "by_sub": by_sub, "sub_cat": sub_cat,
            "eff_pathways": ep_d,
            "eff_pathways_raw": _effective_n(by_path.values()),
            "ep_batch": ep_b, "ep_screen": ep_s,
            "ep_batch_sd": ep.get("ep_batch_sd"),
            "ep_screen_sd": ep.get("ep_screen_sd"),
            "ep_dataset_sd": ep.get("ep_dataset_sd"),
            "n_picks": n_tot, "n_annotated": n_ann,
        }
        print(f"{title:32s} {n_tot:7d} picks  {n_ann / max(n_tot,1):5.1%} annotated  "
              f"{len(by_path):5d} pathways  {len(by_sub):4d} level-2  "
              f"{len(by_cat):3d} categories")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _tint(hex_color: str, amount: float) -> tuple:
    """Blend toward white; amount 0 = the hue itself, 1 = white."""
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _effective_n(weights) -> float:
    """exp(Shannon entropy); see :mod:`assayloop.metrics.effective_pathways`."""
    from assayloop.metrics.effective_pathways import effective_n
    return effective_n(weights)


def draw(data: dict) -> None:
    # one shared category order across every panel, so a hue means the same
    # thing everywhere; ranked by pooled weight over all methods
    pooled: dict[str, float] = defaultdict(float)
    for d in data.values():
        tot = sum(d["by_cat"].values())
        for c, w in d["by_cat"].items():
            pooled[c] += w / tot
    order = [c for c, _ in sorted(pooled.items(), key=lambda kv: -kv[1])][:N_CATS]
    color_of = dict(zip(order, HUES))
    cats = order + ["Other"]
    color_of["Other"] = OTHER

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15), dpi=300,
                             subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor(SURFACE)

    for ax, (title, _prefix, _read) in zip(axes.ravel(), METHODS):
        d = data[title]
        ax.set_facecolor(SURFACE)
        ax.axis("off")

        total = sum(d["by_cat"].values())
        cat_w = {c: d["by_cat"].get(c, 0.0) / total for c in cats[:-1]}
        cat_w["Other"] = 1.0 - sum(cat_w.values())

        # inner ring, categories in the shared order (largest-first within panel
        # would repaint on filtering; the fixed order keeps hue == entity)
        inner = [cat_w[c] for c in cats]
        ax.pie(inner, radius=0.72, startangle=90, counterclock=False,
               colors=[color_of[c] for c in cats],
               wedgeprops=dict(width=0.30, edgecolor=SURFACE, linewidth=1.4))

        # outer ring: Reactome's level-2 groups inside each category (e.g.
        # Immune System -> Innate / Adaptive / Cytokine Signaling), as
        # alternating tints of the parent hue. Leaf pathways are far too fine
        # for a ring -- ~1900 of them turns it into a hairline comb -- so the
        # second tier of the tree is the readable level. A group is only drawn
        # on its own if it clears MIN_SUB of the panel (and is one of the N_SUB
        # largest in its category); everything under that pools into one palest
        # wedge, which keeps the ring free of unreadable slivers.
        kept = set(cats[:-1])
        outer_w, outer_c = [], []
        for c in cats:
            if c == "Other":
                sel = [w for s, w in d["by_sub"].items()
                       if d["sub_cat"].get(s, "Other") not in kept]
            else:
                sel = [w for s, w in d["by_sub"].items() if d["sub_cat"].get(s) == c]
            sel = sorted((w / total for w in sel), reverse=True)
            n_head = min(N_SUB, sum(w >= MIN_SUB for w in sel))
            head, tail = sel[:n_head], sel[n_head:]
            for j, w in enumerate(head):
                outer_w.append(w)
                outer_c.append(_tint(color_of[c], 0.24 if j % 2 else 0.44))
            if tail:
                outer_w.append(sum(tail))
                outer_c.append(_tint(color_of[c], 0.62))
        ax.pie(outer_w, radius=1.0, startangle=90, counterclock=False,
               colors=outer_c,
               wedgeprops=dict(width=0.26, edgecolor=SURFACE, linewidth=0.8))

        # hole: effective number of pathways (exp Shannon) -- the spread, as a number
        eff = d["eff_pathways"]
        ax.text(0, 0.10, f"{eff:.1f}", ha="center", va="center", fontsize=15,
                color=INK, fontweight="semibold")
        ax.text(0, -0.14, "EP-D", ha="center", va="center", fontsize=6.2,
                color=MUTED)

        # direct labels on the inner wedges big enough to hold one; ink or white
        # picked by the fill's luminance so the label always clears contrast
        ang = 90.0
        for c in cats:
            span = cat_w[c] * 360.0
            if cat_w[c] >= LABEL_MIN:
                mid = math.radians(ang - span / 2.0)
                r, g, b = matplotlib.colors.to_rgb(color_of[c])
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(0.57 * math.cos(mid), 0.57 * math.sin(mid),
                        f"{cat_w[c]:.0%}", ha="center", va="center",
                        fontsize=5.8, color=(INK if lum > 0.55 else "#ffffff"),
                        fontweight="bold")
            ang -= span

        # square data window with headroom, so title/stat sit at a fixed
        # distance from the ring in every panel
        stat = f"EP-B {d['ep_batch']:.1f}   EP-S {d['ep_screen']:.1f}"
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-1.30, 1.30)
        ax.text(0, 1.06, title, ha="center", va="bottom", fontsize=8.6, color=INK,
                linespacing=1.15)
        ax.text(0, -1.14, stat, ha="center", va="center", fontsize=6.6, color=MUTED)

    handles = [Patch(facecolor=color_of[c], edgecolor=SURFACE, label=c) for c in cats]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.2,
                     frameon=False, bbox_to_anchor=(0.5, -0.005),
                     handlelength=1.0, handleheight=1.0, columnspacing=1.4,
                     labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.125,
                        wspace=0.02, hspace=0.02)
    save_figure(fig, OUT, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-aggregate from run results")
    a = ap.parse_args()
    data = build_cache() if (a.refresh or not CACHE.is_file()) else json.loads(CACHE.read_text())
    draw(data)


if __name__ == "__main__":
    main()
