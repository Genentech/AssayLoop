"""Embedding drift during training, for EVERY init source (supplementary).

Repeats the two `plot_embedding_drift` panels — (a) pairwise-cosine geometry
preservation init->+GRPO, and (b) top-100 neighbor RBO across each training
transition — once per embedding init in the init-story figure (Random, BPMF, MF,
MF-Sphere, SVD, GenePT, K562). Uses the per-stage cosine-similarity matrices in
output/analysis/embedding_matrix_*.npz (gene-aligned within each source), the
same source->matrix mapping the AUROC analysis uses.

Contrast to expect: structured inits (BPMF and the MF/SVD family) barely move,
while a random init is reorganized by training (low RBO, scattered hexbin).

    uv run python -m assayloop.scripts.plot_embedding_drift_by_source
"""
from __future__ import annotations
import gc
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assayloop import config
from assayloop.scripts.analyze_gene_matrices import INIT_TYPES, _matrix_stems
from assayloop.scripts.plot_embedding_drift import (
    _topk_neighbors, _rbo_rows, K_NEIGH, RBO_P,
    WHITE, INK, INK2, MUTED, GRID, AXIS, BLUE, ORANGE, GRAY,
)

ANALYSIS = config.OUTPUT_PATH / "analysis"
OUT = ANALYSIS / "gene_embedding_drift_by_source.png"
N_PAIRS = 150_000     # sampled gene pairs for the hexbin
N_GENES_RBO = 6_000   # sampled genes for the RBO distribution (per transition)


def _load(stem):
    f = sorted(glob.glob(str(ANALYSIS / ("embedding_matrix_%s*.npz" % stem))))
    if not f:
        return None, None
    d = np.load(f[0], allow_pickle=True)
    return d["similarity"].astype(np.float32), list(d["gene_names"])


def _source_metrics(stems, rng):
    """Return (x_init, y_rl, r, {transition: rbo_array}) for one source, or None."""
    raw, gn_r = _load(stems["raw_emb"])
    sup, gn_s = _load(stems["sup_emb"])
    rl, gn_l = _load(stems["rl_emb"])
    if raw is None or sup is None or rl is None:
        return None
    if not (gn_r == gn_s == gn_l):
        return None                      # this script assumes aligned matrices
    n = raw.shape[0]
    # Genes the source actually covers have a unit self-cosine; uncovered genes
    # get a zero init vector (cosine 0 with everything) and would otherwise form
    # a spurious vertical stripe at cos_init=0. Restrict to covered genes.
    valid = np.flatnonzero(np.diag(raw) > 0.5)
    n_excluded = n - len(valid)

    # (a) pairwise cosine, init vs +GRPO (pairs among covered genes)
    vi = valid[rng.integers(0, len(valid), size=N_PAIRS * 2)]
    vj = valid[rng.integers(0, len(valid), size=N_PAIRS * 2)]
    m = vi < vj
    ii, jj = vi[m][:N_PAIRS], vj[m][:N_PAIRS]
    x_init, y_rl = raw[ii, jj], rl[ii, jj]
    r = float(np.corrcoef(x_init, y_rl)[0, 1])

    # (b) top-k neighbor RBO across transitions, on a covered-gene sample
    tk = {"init": _topk_neighbors(raw, K_NEIGH),
          "sup": _topk_neighbors(sup, K_NEIGH),
          "rl": _topk_neighbors(rl, K_NEIGH)}
    samp = rng.choice(valid, size=min(N_GENES_RBO, len(valid)), replace=False)
    rbo = {
        "init→sup": _rbo_rows(tk["init"][samp], tk["sup"][samp], RBO_P),
        "sup→+GRPO": _rbo_rows(tk["sup"][samp], tk["rl"][samp], RBO_P),
        "init→+GRPO": _rbo_rows(tk["init"][samp], tk["rl"][samp], RBO_P),
    }
    del raw, sup, rl, tk
    gc.collect()
    return x_init, y_rl, r, rbo, n_excluded


def main():
    stems = _matrix_stems()
    sources = [lab for lab, _ in INIT_TYPES if lab in stems]
    rng = np.random.default_rng(0)

    nrows = len(sources)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.0 * nrows), dpi=160,
                             squeeze=False)
    fig.patch.set_facecolor(WHITE)
    rbo_keys = ["init→sup", "sup→+GRPO", "init→+GRPO"]
    rbo_display = ["init → SFT", "SFT → RL", "init → RL"]
    colors = [GRAY, BLUE, ORANGE]

    for row, src in enumerate(sources):
        axA, axB = axes[row]
        res = _source_metrics(stems[src], rng)
        for ax in (axA, axB):
            ax.set_facecolor(WHITE)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
        if res is None:
            axA.text(0.5, 0.5, "%s: matrices missing" % src, ha="center",
                     va="center", transform=axA.transAxes, color=MUTED)
            continue
        x_init, y_rl, r, rbo, n_excluded = res

        # Panel A: pairwise geometry
        axA.hexbin(x_init, y_rl, gridsize=45, bins="log", cmap="Blues", mincnt=1)
        axA.plot([-1, 1], [-1, 1], color=ORANGE, lw=1.1, ls="--")
        axA.set_xlim(-1, 1); axA.set_ylim(-1, 1)
        axA.text(0.04, 0.93, "r = %.3f" % r, transform=axA.transAxes,
                 fontsize=9.5, color=INK, va="top")
        axA.set_ylabel("%s\n\ncos (after RL)" % src, fontsize=9.5, color=INK,
                       fontweight="bold")
        if row == nrows - 1:
            axA.set_xlabel("pairwise cos — init", fontsize=9, color=INK2)

        # Panel B: neighbor RBO
        data = [rbo[k] for k in rbo_keys]
        parts = axB.violinplot(data, showextrema=False, widths=0.82)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i]); body.set_alpha(0.35)
            body.set_edgecolor(colors[i]); body.set_linewidth(1.1)
        for i, d in enumerate(data, 1):
            med = float(np.median(d))
            axB.scatter([i], [med], color=INK, s=14, zorder=5)
            axB.text(i, med, "  %.2f" % med, va="center", ha="left",
                     fontsize=8, color=INK)
        axB.set_ylim(-0.02, 1.03)
        axB.set_xticks(range(1, len(rbo_display) + 1))
        axB.set_xticklabels(rbo_display if row == nrows - 1 else [""] * len(rbo_display),
                            fontsize=9, color=INK2)
        axB.set_ylabel("Top-%d neighbor RBO" % K_NEIGH, fontsize=9, color=INK2)
        axB.grid(True, axis="y", color=GRID, lw=0.7)
        axB.set_axisbelow(True)
        # set off the end-to-end init->RL violin from the two per-stage steps
        axB.axvline(2.5, color=MUTED, ls=":", lw=1.0, alpha=0.8)
        if row == 0:
            axB.text(1.5, 1.05, "per-stage", ha="center", va="bottom",
                     fontsize=8.5, color=MUTED, style="italic")
            axB.text(3.0, 1.05, "end-to-end", ha="center", va="bottom",
                     fontsize=8.5, color=MUTED, style="italic")
        print("  %-10s  pairwise r=%.3f  RBO init→+GRPO median=%.3f  "
              "(excluded %d uncovered genes)"
              % (src, r, np.median(rbo["init→+GRPO"]), n_excluded))

    fig.tight_layout()
    fig.savefig(OUT, facecolor=WHITE, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".png", ".pdf"), facecolor=WHITE, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
