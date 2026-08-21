"""Why BPMF is a better embedding INIT: textbook-biology recovery is the wrong proxy.

Scatter: x = how well the RAW embedding recovers curated PPI networks
(STRING/CORUM/SIGNOR/REACTOME, mean AUROC), y = downstream task performance
(AssayLoop Transformer NVR by init). Two points per init — supervised
(Transformer) and +GRPO (RL) — connected vertically (x is a property of the
init, fixed across stages). The best PPI-recovery embedding (GenePT, a
literature/LLM embedding) is near-worst on task at BOTH stages; BPMF (built from
the screen hit matrix) wins at both while looking ~random on PPI recovery. A
faint dotted least-squares fit per stage shows the (GenePT-leveraged) downward
trend; RL lifts everything but keeps the negative slope.

x is read from output/analysis/gene_matrix_analysis.json (Raw emb. stage).
y is from the full_genome ablation table (--min-screen-freq 2): supervised =
"Transformer (<init>)" rows, RL = "<init> + GRPO" rows. Update if sweeps change.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from assayloop import config

ANALYSIS = config.OUTPUT_PATH / "analysis"
OUT = ANALYSIS / "gene_embedding_init_story.png"
GTS = ["CORUM", "REACTOME", "SIGNOR", "STRING"]

# Downstream NVR by init (full_genome ablations, --min-screen-freq 2)
NVR_SUP = {"Random": 2.47, "BPMF": 3.83, "MF": 3.01, "MF-Sphere": 3.72,
           "SVD": 3.72, "GenePT": 1.06, "K562": 2.95}
# Random +GRPO uses the genuine RL model (gf-random-train-d10-rl/model_last.pt;
# its model.pt was a copy of the supervised init) — sweep-fg-f2-fg-ablation-random-rl-last.
NVR_RL = {"Random": 2.71, "BPMF": 4.83, "MF": 4.05, "MF-Sphere": 4.31,
          "SVD": 4.40, "GenePT": 2.60, "K562": 3.15}

SURFACE="#ffffff"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8880"
GRID="#e6e5de"; AXIS="#c3c2b7"; BLUE="#2a78d6"; ORANGE="#e8792b"; GRAY="#a9a79f"

rows = json.loads((ANALYSIS / "gene_matrix_analysis.json").read_text())
def raw_auc(it):
    r = [x for x in rows if x["init"] == it and x["stage"] == "Raw emb."][0]
    return float(np.mean([r.get(f"{g}_auroc", 0.5) for g in GTS]))

inits = list(NVR_SUP)
color = {i: (BLUE if i == "BPMF" else ORANGE if i == "GenePT" else GRAY) for i in inits}
x = {i: raw_auc(i) for i in inits}

fig, ax = plt.subplots(figsize=(7.8, 6.1), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=9)

# faint per-stage least-squares fits (drawn first, recessive)
xline = np.linspace(0.47, 0.82, 50)
for nvr, lab in [(NVR_SUP, "supervised"), (NVR_RL, "+GRPO")]:
    pts_fit = [(x[i], nvr[i]) for i in inits if nvr[i] is not None]
    fx, fy = np.array([p[0] for p in pts_fit]), np.array([p[1] for p in pts_fit])
    m, b = np.polyfit(fx, fy, 1)
    ax.plot(xline, m * xline + b, ls=":", lw=1.2, color=MUTED, alpha=0.55, zorder=1)
    ax.text(0.815, m * 0.815 + b, f" {lab}", fontsize=7.4, color=MUTED,
            va="center", ha="left", alpha=0.9)

# paired dots: supervised (filled) -> RL (open). MF/MF-Sphere share the exact
# same x (sphering doesn't change cosine), so curve their connectors apart.
CURVE = {"MF": -0.45, "MF-Sphere": 0.45}
for i in inits:
    hi = i in ("BPMF", "GenePT")
    c = color[i]; xi = x[i]
    ax.scatter([xi], [NVR_SUP[i]], s=140 if hi else 78, c=c,
               edgecolor=SURFACE, linewidth=1.3, zorder=4)                 # supervised
    if NVR_RL[i] is None:
        continue
    ax.scatter([xi], [NVR_RL[i]], s=140 if hi else 78, facecolor=SURFACE,
               edgecolor=c, linewidth=2.0, zorder=4)                       # +GRPO
    conn = FancyArrowPatch((xi, NVR_SUP[i]), (xi, NVR_RL[i]),
                           connectionstyle=f"arc3,rad={CURVE.get(i, 0.0)}",
                           arrowstyle="-", color=c, lw=1.3, alpha=0.45, zorder=2)
    ax.add_patch(conn)

# label each init once, beside its pair
LOFF = {"Random": (10, -12), "BPMF": (12, 8), "MF": (-6, -14), "MF-Sphere": (9, 6),
        "SVD": (9, 6), "GenePT": (-10, -2), "K562": (10, -10)}
for i in inits:
    dx, dy = LOFF[i]
    yl = NVR_RL[i] if (dy > 0 and NVR_RL[i] is not None) else NVR_SUP[i]
    ax.annotate(i, (x[i], yl), textcoords="offset points", xytext=(dx, dy),
                fontsize=10, fontweight=("bold" if i in ("BPMF", "GenePT") else "normal"),
                color=(INK if i in ("BPMF", "GenePT") else INK2),
                ha=("right" if dx < 0 else "left"), zorder=6)

ax.set_xlabel("Average Recovery AUROC (STRING / CORUM / SIGNOR / REACTOME)",
              fontsize=10.5, color=INK2)
ax.set_ylabel("Downstream task performance  (AssayFormer EF)",
              fontsize=10.5, color=INK2)

ax.annotate("built from the screen hit matrix →\nbest on task at both stages",
            xy=(x["BPMF"], NVR_RL["BPMF"]), xytext=(0.32, 0.95),
            textcoords="axes fraction", fontsize=8.6, color=BLUE, va="top",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1, alpha=0.5))
ax.annotate("literature/LLM embedding → best PPI\nrecovery, near-worst on task",
            xy=(x["GenePT"], NVR_SUP["GenePT"]), xytext=(0.52, 0.20),
            textcoords="axes fraction", fontsize=8.6, color=ORANGE, va="top",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1, alpha=0.5))

ax.axhline(NVR_SUP["Random"], color=AXIS, ls=":", lw=1, alpha=0.7, zorder=1)

leg = ax.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", markerfacecolor=INK2,
           markeredgecolor=SURFACE, markersize=9, label="Supervised"),
    Line2D([0], [0], marker="o", ls="none", markerfacecolor=SURFACE,
           markeredgecolor=INK2, markeredgewidth=2, markersize=9, label="+ GRPO (RL)"),
], loc="lower left", fontsize=8.5, frameon=False, handletextpad=0.4,
   bbox_to_anchor=(0.005, 0.005))
for t in leg.get_texts(): t.set_color(INK2)

ax.set_xlim(0.47, 0.82); ax.set_ylim(0.7, 5.15)
fig.tight_layout()
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
fig.savefig(str(OUT).replace(".png", ".pdf"), facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)
