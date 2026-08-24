"""Does the acquirer actually use the hit feedback? — with vs. without.

The main-text table only reports the hit-informed rows; this figure carries the
ablation. One dumbbell per method: dark dot = the run as reported, light dot =
the same method run blind. LLM rows are ordered by EF; the two rows below the
rule run component-then-system, AssayFormer before the AssayLoop built on it.

The two groups are blinded differently, which is why a rule separates them:

* **LLMs** — the same 10 x 100 loop with the per-round hit labels stripped from
  the prompt. The model still sees which genes it already picked, just not which
  of them hit.
* **AssayFormer**, and the **AssayLoop** system built on it — no readout is ever
  seen, by any component. AssayFormer's blind arm is its opening ranking, one
  batch of 1000 against an empty observation history
  (`assayformer_no_context.py`); EF at a fixed 1000-gene budget is
  order-invariant, so 1 x 1000 and 10 x 100 are the same measurement. AssayLoop's
  is both halves blinded at once (`assayloop_no_context.py`): the blind-prompt
  Gemini for rounds 1-3, then AssayFormer's opening ranking, minus what Gemini
  already took, for the remaining 700 genes. Nothing is handed over between them
  because there is nothing to hand over -- which is the point of the row.

nAUC moves the same way for every LLM and is kept in DATA below; add its entry
back to PANELS to draw it as a second panel. It is undefined for AssayFormer's
single-batch arm (nAUC is not order-invariant), which is the other reason that
row is set apart; AssayLoop's blind arm keeps its 10 rounds, so its nAUC is
defined.

LLM numbers are the cached per-screen means from the same sweeps the baselines
table draws on (`results_index.py`, `n_hits_vs_random` and `hits_auc_normalized`).
Update if those sweeps are re-run.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from assayloop.scripts._figure_io import save_figure
from assayloop import config

OUT = config.OUTPUT_PATH / "analysis" / "llm_label_ablation.png"

# model -> {metric: (with labels, without labels, paired stderr of the gap)}
# The stderr is over the 20 test screens of the per-screen (with - without)
# difference -- the screens are shared, so the paired spread is the right ruler
# (the unpaired across-screen std is ~10x larger and says nothing about the gap).
# It is NOT drawn: at 1-2 SE per model the whiskers overlap the dots and read as
# clutter. Kept here for the caption, which is where the uncertainty belongs --
# the claim is "positive for every model", not per-model significance.
DATA = {
    "Gemini-3.1-Pro":  {"EF": (4.69, 4.35, 0.21), "nAUC": (21.0, 18.7, 1.0)},
    "GLM-5.1":         {"EF": (3.98, 3.63, 0.15), "nAUC": (17.0, 15.4, 0.5)},
    "Claude Opus-4.8": {"EF": (3.84, 3.53, 0.28), "nAUC": (17.5, 15.9, 1.0)},
    "Kimi-K2.6":       {"EF": (2.99, 2.71, 0.18), "nAUC": (14.7, 13.0, 0.8)},
    "Qwen3.6-27B":     {"EF": (2.50, 2.31, 0.12), "nAUC": (11.2, 10.6, 0.5)},
    # The transformer's analogue of "no labels" is its opening ranking: one
    # batch of 1000 with an empty history, no readout ever seen. Different
    # mechanism from the blind LLM prompt, so it sits in its own group -- but
    # the same question, and EF at a fixed budget is order-invariant so the two
    # numbers are the same measurement. Both arms are the table's
    # domain-adjusted EF -- assayformer_no_context.py reports the *raw*
    # n_hits_vs_random (4.04 blind), which is a different scale from every other
    # number in this figure; on the table's scale the blind arm is 3.93 and the
    # with-feedback arm reproduces the table row exactly.
    "AssayFormer":     {"EF": (4.83, 3.93, 0.40), "nAUC": None},
    # AssayLoop = Gemini-3.1-Pro -> AssayFormer, from assayloop_no_context.py.
    # Both arms are scored with the table's domain-adjusted EF/nAUC, so the
    # with-feedback side reproduces the LaTeX row exactly (5.66 / 21.7) and the
    # gap is like-for-like. 14/20 screens improve. Drawn below AssayFormer, the
    # component it is built on, rather than in EF order.
    "AssayLoop":       {"EF": (5.66, 4.99, 0.29), "nAUC": (21.7, 19.5, 1.5)},
}
# rows drawn as a separate group under a rule (blinded by a different mechanism:
# no readout at all, rather than a prompt with the labels stripped)
SEPARATE = ["AssayFormer", "AssayLoop"]
PANELS = [("EF", "Enrichment factor (EF)")]

SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, AXIS = "#e6e5de", "#c3c2b7"
WITHOUT, WITH = "#86b6ef", "#1c5cab"   # blue ramp steps 250 / 550

models = list(DATA)                     # LLMs by EF; then AssayFormer, AssayLoop
# the separated group sits at the bottom, under a gap and a hairline
GAP = 0.75
_main = [m for m in models if m not in SEPARATE]
ypos, _y = {}, float(len(_main) + len(SEPARATE) - 1 + GAP)
for m in _main:
    ypos[m], _y = _y, _y - 1
_y -= GAP
for m in SEPARATE:
    ypos[m], _y = _y, _y - 1
RULE_Y = (ypos[_main[-1]] + ypos[SEPARATE[0]]) / 2 if SEPARATE else None
ys = [ypos[m] for m in models]

fig, axes = plt.subplots(1, len(PANELS), figsize=(3.9 * len(PANELS), 3.4),
                         dpi=300, squeeze=False)
axes = list(axes.ravel())
fig.patch.set_facecolor(SURFACE)

for ax, (key, xlabel) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)

    lo = min(v[key][1] for v in DATA.values() if v[key])
    hi = max(v[key][0] for v in DATA.values() if v[key])
    pad = 0.14 * (hi - lo)

    if RULE_Y is not None:
        ax.axhline(RULE_Y, color=AXIS, lw=0.7, zorder=1)

    for y, m in zip(ys, models):
        if not DATA[m][key]:
            continue
        w, wo = DATA[m][key][:2]
        ax.plot([wo, w], [y, y], color=WITHOUT, lw=2.0, solid_capstyle="round",
                zorder=2)
        ax.scatter([wo], [y], s=70, facecolor=WITHOUT, edgecolor=SURFACE,
                   linewidth=1.6, zorder=3)
        ax.scatter([w], [y], s=70, facecolor=WITH, edgecolor=SURFACE,
                   linewidth=1.6, zorder=4)
        # label only the labelled-run endpoint; the axis carries the rest
        ax.annotate(f"{w:.2f}" if key == "EF" else f"{w:.1f}", (w, y),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=7.6, color=INK)

    ax.set_xlim(lo - pad, hi + 2.2 * pad)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(models if ax is axes[0] else [""] * len(models),
                       fontsize=8.6, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8.8, color=INK2, labelpad=4)

leg = fig.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", markerfacecolor=WITH,
           markeredgecolor=SURFACE, markeredgewidth=1.2, markersize=7,
           label="With hit feedback"),
    Line2D([0], [0], marker="o", ls="none", markerfacecolor=WITHOUT,
           markeredgecolor=SURFACE, markeredgewidth=1.2, markersize=7,
           label="Without hit feedback"),
], loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2, fontsize=8.4,
   frameon=False, handletextpad=0.35, columnspacing=1.6)
for t in leg.get_texts():
    t.set_color(INK2)

fig.tight_layout(rect=(0, 0, 1, 0.95))
save_figure(fig, OUT, facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)
