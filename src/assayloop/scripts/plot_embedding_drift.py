"""Gene embeddings barely move during training (given a BPMF init).

Views of the BPMF-initialized gene-embedding table across the training pipeline
(BPMF init -> supervised -> +GRPO). The supervised/RL geometries come from the
per-stage cosine-similarity matrices in output/analysis/embedding_matrix_*.npz;
the init geometry is rebuilt from the model's ACTUAL init pkl (the on-disk
raw_bpmf matrix may be a different BPMF fit). All are gene-aligned.

  (a) Pairwise gene geometry is preserved end-to-end: a random sample of gene
      pairs has almost the same cosine at BPMF init as in the fully-trained
      (+GRPO) model (points hug the diagonal; Pearson r near 1).
  (b) Nearest-neighbor MEMBERSHIP is preserved: Jaccard overlap of each gene's
      top-100 neighbors across each training transition.
  (c) Nearest-neighbor RANK is preserved: Rank-Biased Overlap (RBO, top-weighted,
      handles non-conjoint lists) of each gene's ranked top-100 neighbors.

Jaccard/RBO on the actual neighbor lists are more discriminative than the cosine
of the full similarity profile, which saturates when embeddings are dense.

Run after the embedding matrices exist (scripts/compute_embedding_matrix.py):
    uv run python -m assayloop.scripts.plot_embedding_drift
"""
from __future__ import annotations
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assayloop import config

ANALYSIS = config.OUTPUT_PATH / "analysis"
OUT = ANALYSIS / "gene_embedding_drift.png"

WHITE = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8880"
GRID = "#e6e5de"; AXIS = "#c3c2b7"; BLUE = "#2a78d6"; ORANGE = "#e8792b"; GRAY = "#a9a79f"

# The supervised model's *actual* BPMF init (from its config.json
# init_gene_factors). Set ASSAYLOOP_BPMF_CHECKPOINT to the matching
# bpmf_result.pkl; there is no bundled default.
INIT_PKL = config.BPMF_CHECKPOINT
SUP = "embedding_matrix_gf-bpmf-train-hits_model.npz"
RL = "embedding_matrix_gf-bpmf-train-hits-rl-fg-s19_model_last.npz"

K_NEIGH = 100      # nearest-neighbor horizon
RBO_P = 0.98       # RBO persistence: ~1/(1-p)=50 -> weight concentrated in top ~50


def _load_sim(pattern):
    matches = sorted(glob.glob(str(ANALYSIS / pattern)))
    if not matches:
        raise FileNotFoundError("No embedding matrix for %s" % pattern)
    d = np.load(matches[0], allow_pickle=True)
    return d["similarity"].astype(np.float32), list(d["gene_names"])


def _init_sim_for_genes(gene_names, pkl_path):
    """BPMF-init cosine-similarity matrix aligned to ``gene_names`` (the model
    initializes each gene to its BPMF posterior-mean factor). Returns (sim, keep)
    where keep marks genes actually present in the BPMF fit."""
    res = pickle.load(open(pkl_path, "rb"))
    V = res.posterior_mean_V()
    idx = {g: i for i, g in enumerate(res.gene_names)}
    rows = np.zeros((len(gene_names), V.shape[1]), dtype=np.float32)
    keep = np.zeros(len(gene_names), dtype=bool)
    for i, g in enumerate(gene_names):
        j = idx.get(g)
        if j is not None:
            rows[i] = V[j]
            keep[i] = True
    Vn = rows[keep]
    Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9)
    return (Vn @ Vn.T).astype(np.float32), keep


def _topk_neighbors(S, k):
    """Per-gene ranked top-k neighbor indices (descending similarity, self
    excluded). Returns (n, k) int array."""
    n = S.shape[0]
    np.fill_diagonal(S, -np.inf)                      # never pick self
    part = np.argpartition(S, -k, axis=1)[:, -k:]     # (n, k) unordered top-k
    rows = np.arange(n)[:, None]
    order = np.argsort(-S[rows, part], axis=1)        # sort each row desc
    topk = part[rows, order]
    np.fill_diagonal(S, 0.0)
    return topk


def _jaccard_at_k(A, B):
    """Per-gene Jaccard overlap of two (n, k) neighbor-index arrays."""
    n, k = A.shape
    out = np.empty(n)
    for i in range(n):
        inter = len(set(A[i].tolist()) & set(B[i].tolist()))
        out[i] = inter / (2 * k - inter)
    return out


def _rbo_ext(a, b, p):
    """Rank-Biased Overlap (extrapolated) of two equal-length ranked id lists."""
    k = len(a)
    sa, sb = set(), set()
    overlap = 0
    agg = 0.0
    for d in range(k):
        x, y = a[d], b[d]
        if x == y:
            overlap += 1
        else:
            if x in sb:
                overlap += 1
            if y in sa:
                overlap += 1
        sa.add(x); sb.add(y)
        agg += (overlap / (d + 1)) * (p ** (d + 1))
    return (overlap / k) * (p ** k) + ((1.0 - p) / p) * agg


def _rbo_rows(A, B, p):
    """Per-gene RBO of two (n, k) ranked neighbor-index arrays."""
    n = A.shape[0]
    out = np.empty(n)
    Al, Bl = A.tolist(), B.tolist()
    for i in range(n):
        out[i] = _rbo_ext(Al[i], Bl[i], p)
    return out


def main():
    sup, names = _load_sim(SUP)
    rl, n_rl = _load_sim(RL)
    assert names == n_rl, "stage matrices are not gene-aligned"
    raw, keep = _init_sim_for_genes(names, INIT_PKL)
    sup = sup[np.ix_(keep, keep)]
    rl = rl[np.ix_(keep, keep)]
    n = raw.shape[0]
    print("genes with BPMF init: %d / %d" % (n, len(names)))

    # (a) sampled pairwise cosines: init vs fully-trained
    rng = np.random.default_rng(0)
    n_sample = 300_000
    ii = rng.integers(0, n, size=n_sample * 2)
    jj = rng.integers(0, n, size=n_sample * 2)
    m = ii < jj
    ii, jj = ii[m][:n_sample], jj[m][:n_sample]
    x_init, y_rl = raw[ii, jj], rl[ii, jj]
    r_pair = float(np.corrcoef(x_init, y_rl)[0, 1])

    # (b, c) top-k neighbor membership (Jaccard) and rank (RBO) per transition
    tk = {"init": _topk_neighbors(raw, K_NEIGH),
          "sup": _topk_neighbors(sup, K_NEIGH),
          "rl": _topk_neighbors(rl, K_NEIGH)}
    trans = [("BPMF → SFT", "init", "sup"),
             ("SFT → RL", "sup", "rl"),
             ("BPMF → RL", "init", "rl")]
    jac = {lab: _jaccard_at_k(tk[a], tk[b]) for lab, a, b in trans}
    rbo = {lab: _rbo_rows(tk[a], tk[b], RBO_P) for lab, a, b in trans}

    labels = [t[0] for t in trans]
    colors = [GRAY, BLUE, ORANGE]

    def _violins(ax, data_by_label, ylabel):
        data = [data_by_label[k] for k in labels]
        parts = ax.violinplot(data, showextrema=False, widths=0.82)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i]); body.set_alpha(0.35)
            body.set_edgecolor(colors[i]); body.set_linewidth(1.2)
        for i, d in enumerate(data, 1):
            med = float(np.median(d))
            ax.scatter([i], [med], color=INK, s=16, zorder=5)
            ax.text(i, med, "  %.2f" % med, va="center", ha="left",
                    fontsize=9, color=INK)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10, color=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        lo = min(float(np.percentile(d, 1)) for d in data)
        ax.set_ylim(max(0.0, lo - 0.05), 1.02)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=200)
    fig.patch.set_facecolor(WHITE)

    # --- Panel A: pairwise geometry preservation ---
    axA.set_facecolor(WHITE)
    hb = axA.hexbin(x_init, y_rl, gridsize=55, bins="log", cmap="Blues", mincnt=1)
    axA.plot([-1, 1], [-1, 1], color=ORANGE, lw=1.3, ls="--")
    axA.set_xlim(-1, 1); axA.set_ylim(-1, 1)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axA.spines[sp].set_color(AXIS)
    axA.tick_params(colors=MUTED, labelsize=9)
    axA.set_xlabel("Pairwise cosine (BPMF init)", fontsize=10.5, color=INK2)
    axA.set_ylabel("Pairwise cosine (after RL)", fontsize=10.5, color=INK2)
    axA.text(0.04, 0.93, "Pearson r = %.3f" % r_pair,
             transform=axA.transAxes, fontsize=10.5, color=INK, va="top")
    cb = fig.colorbar(hb, ax=axA, fraction=0.046, pad=0.02)
    cb.set_label("pairs (log)", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)

    # --- Panel B: nearest-neighbor preservation (RBO, rank-weighted) ---
    # RBO subsumes membership + order; Jaccard@100 is computed/printed for the
    # caption but not plotted (near-identical to RBO here).
    _violins(axB, rbo, "Top-%d neighbor RBO" % K_NEIGH)

    # Set off the end-to-end BPMF->RL violin from the two per-stage transitions
    # that compose it (BPMF->SFT then SFT->RL).
    axB.axvline(2.5, color=MUTED, ls=":", lw=1.2, alpha=0.8)
    ylo, yhi = axB.get_ylim()
    axB.text(1.5, yhi, "per-stage", ha="center", va="top", fontsize=9.5,
             color=MUTED, style="italic")
    axB.text(3.0, yhi, "end-to-end", ha="center", va="top", fontsize=9.5,
             color=MUTED, style="italic")

    fig.tight_layout()
    fig.savefig(OUT, facecolor=WHITE, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".png", ".pdf"), facecolor=WHITE, bbox_inches="tight")
    print("wrote", OUT)
    print("pairwise r(init, +GRPO) = %.4f" % r_pair)
    for lab in labels:
        print("  %-18s  Jaccard@%d median=%.3f  RBO median=%.3f"
              % (lab, K_NEIGH, np.median(jac[lab]), np.median(rbo[lab])))


if __name__ == "__main__":
    main()
