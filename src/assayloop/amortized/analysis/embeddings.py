"""Part A: learned gene embeddings vs PRESAGE knowledge sources.

Quantifies how much of the learned geometry reflects known biology (kNN
overlap, linear CKA, RSA, ridge decodability) and surfaces candidate novel,
screen-driven associations (gene pairs that are close in the learned space but
not in any PRESAGE source, cross-checked against empirical co-hit lift).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .loaders import (
    CoHitIndex,
    PathwayIndex,
    RankerArtifacts,
    SourceMatrix,
    cosine_topk,
    l2norm,
    load_presage_source,
)

log = logging.getLogger("assayloop.amortized.analysis.embeddings")


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representations of the same
    n rows. Scale/rotation-invariant; in [0, 1]."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denom = (np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    return float(cross / denom) if denom > 0 else 0.0


def rsa_spearman(Xn: np.ndarray, Yn: np.ndarray, *, sample: int = 1500, seed: int = 0) -> float:
    """Spearman correlation of pairwise cosine similarities (RSA). Operates on
    a random sample of rows for tractability. Inputs assumed L2-normalized."""
    from scipy.stats import spearmanr

    n = Xn.shape[0]
    if n < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n > sample:
        sel = rng.choice(n, size=sample, replace=False)
        Xn, Yn = Xn[sel], Yn[sel]
    sx = Xn @ Xn.T
    sy = Yn @ Yn.T
    iu = np.triu_indices(sx.shape[0], k=1)
    rho, _ = spearmanr(sx[iu], sy[iu])
    return float(rho)


def knn_overlap(Xn: np.ndarray, Yn: np.ndarray, k: int) -> dict[str, float]:
    """Mean Jaccard overlap of top-k neighbor sets in two spaces over the same
    rows, plus the random-chance baseline ``k/(N-1)``."""
    n = Xn.shape[0]
    k = min(k, max(1, n - 1))
    idx_x, _ = cosine_topk(Xn, k)
    idx_y, _ = cosine_topk(Yn, k)
    jac = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = set(idx_x[i].tolist())
        b = set(idx_y[i].tolist())
        inter = len(a & b)
        jac[i] = inter / (len(a | b) or 1)
    chance = k / max(1, n - 1)
    return {"k": k, "mean_jaccard": float(jac.mean()),
            "median_jaccard": float(np.median(jac)),
            "chance_jaccard": float(chance),
            "lift_over_chance": float(jac.mean() / chance) if chance > 0 else float("nan")}


def ridge_r2(X: np.ndarray, Y: np.ndarray, *, sample: int = 4000, seed: int = 0,
             alpha: float = 10.0, folds: int = 3) -> float:
    """Cross-validated variance-weighted R^2 of predicting source ``Y`` from the
    learned ``X`` (how linearly decodable the known signal is)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    n = X.shape[0]
    if n < folds * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n > sample:
        sel = rng.choice(n, size=sample, replace=False)
        X, Y = X[sel], Y[sel]
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha)
        m.fit(X[tr], Y[tr])
        pred = m.predict(X[te])
        scores.append(r2_score(Y[te], pred, multioutput="variance_weighted"))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Per-source comparison
# ---------------------------------------------------------------------------


def compare_source(
    learned_by_gene: dict[str, np.ndarray],
    sm: SourceMatrix,
    *,
    knn_ks: Sequence[int] = (10, 20, 50),
    seed: int = 0,
) -> dict[str, Any]:
    """Compare learned embeddings to one PRESAGE source over their common genes."""
    L = np.stack([learned_by_gene[g] for g in sm.genes], axis=0).astype(np.float32)
    S = sm.X.astype(np.float32)
    Ln = l2norm(L)
    Sn = l2norm(S)
    out: dict[str, Any] = {
        "source": sm.source,
        "n_common_genes": len(sm.genes),
        "source_dim": sm.dim,
        "cka": linear_cka(L, S),
        "rsa_spearman": rsa_spearman(Ln, Sn, seed=seed),
        "ridge_r2": ridge_r2(L, S, seed=seed),
        "knn": {str(k): knn_overlap(Ln, Sn, k) for k in knn_ks},
    }
    return out


# ---------------------------------------------------------------------------
# Novelty: learned-near but PRESAGE-far pairs
# ---------------------------------------------------------------------------


def novel_pairs(
    learned_by_gene: dict[str, np.ndarray],
    genes: list[str],
    sources: dict[str, SourceMatrix],
    cohit: CoHitIndex,
    pathways: PathwayIndex | None = None,
    *,
    neighbors_per_gene: int = 10,
    top_n: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rank gene pairs by ``learned_cos - max_source_cos`` (over sources where
    BOTH genes are known), then attach empirical co-hit lift to flag pairs the
    model links from screen data beyond prior knowledge."""
    Ln = l2norm(np.stack([learned_by_gene[g] for g in genes], axis=0))
    idx, sim = cosine_topk(Ln, neighbors_per_gene)

    # Per-source normalized lookups for arbitrary-pair cosine.
    src_norm: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    for name, sm in sources.items():
        src_norm[name] = ({g: i for i, g in enumerate(sm.genes)}, l2norm(sm.X))

    seen: set[tuple[int, int]] = set()
    cands: list[dict[str, Any]] = []
    for i, g in enumerate(genes):
        for jcol in range(idx.shape[1]):
            j = int(idx[i, jcol])
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            a, b = genes[key[0]], genes[key[1]]
            learned_cos = float(sim[i, jcol])
            # Max source cosine across sources where both known.
            max_src = -1.0
            n_src_cov = 0
            for name, (g2r, Sn) in src_norm.items():
                ra, rb = g2r.get(a), g2r.get(b)
                if ra is not None and rb is not None:
                    n_src_cov += 1
                    c = float(Sn[ra] @ Sn[rb])
                    if c > max_src:
                        max_src = c
            if n_src_cov == 0:
                continue  # can't claim novelty vs knowledge without coverage
            cands.append({
                "gene_a": a, "gene_b": b,
                "learned_cos": learned_cos,
                "max_source_cos": max_src,
                "novelty": learned_cos - max_src,
                "n_sources_cov": n_src_cov,
            })
    cands.sort(key=lambda d: d["novelty"], reverse=True)
    top = cands[:top_n]
    for d in top:
        lift = cohit.lift(d["gene_a"], d["gene_b"])
        d["cohit_lift"] = lift.get("lift")
        d["cohit_n_screens"] = lift.get("n_screens")
        d["cohit_n_both_hit"] = lift.get("n_both_hit")
        if pathways is not None:
            d.update(pathways.cooccurrence(d["gene_a"], d["gene_b"]))
    return top


# ---------------------------------------------------------------------------
# Clustering + per-source coherence + 2D projection
# ---------------------------------------------------------------------------


def cluster_and_project(
    learned_by_gene: dict[str, np.ndarray],
    genes: list[str],
    sources: dict[str, SourceMatrix],
    *,
    n_clusters: int = 25,
    seed: int = 0,
) -> dict[str, Any]:
    """KMeans on the learned embeddings; per-cluster coherence in each PRESAGE
    space (mean intra-cluster cosine minus the global background); plus 2D
    coordinates for plotting. UMAP is used when ``umap-learn`` is installed;
    otherwise the projection falls back to PCA and labels itself as such."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    L = np.stack([learned_by_gene[g] for g in genes], axis=0).astype(np.float32)
    Ln = l2norm(L)
    n_clusters = max(2, min(n_clusters, L.shape[0] // 5 or 2))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    labels = km.fit_predict(Ln)

    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(L)
    coords = pca_coords
    projection_method = "PCA"
    projection_note = "umap-learn is not installed; using PCA coordinates as a fallback."
    umap_coords = None
    umap_params = {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}
    try:
        from umap import UMAP  # type: ignore  # noqa: PLC0415

        model = UMAP(
            n_neighbors=min(15, max(2, L.shape[0] - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
            n_components=2,
        )
        umap_coords = model.fit_transform(Ln)
        coords = umap_coords
        projection_method = "UMAP"
        projection_note = "UMAP fit on L2-normalized learned gene embeddings."
    except Exception as e:  # noqa: BLE001
        log.warning("UMAP projection unavailable; falling back to PCA: %s", e)

    # Per-source background mean cosine + per-cluster intra coherence.
    src_norm = {name: ({g: i for i, g in enumerate(sm.genes)}, l2norm(sm.X))
                for name, sm in sources.items()}

    def intra_cos(rows: np.ndarray) -> float:
        if rows.shape[0] < 2:
            return float("nan")
        sims = rows @ rows.T
        iu = np.triu_indices(rows.shape[0], k=1)
        return float(sims[iu].mean())

    clusters = []
    for c in range(n_clusters):
        members = [genes[i] for i in np.nonzero(labels == c)[0]]
        learned_coh = intra_cos(Ln[labels == c])
        src_coh = {}
        for name, (g2r, Sn) in src_norm.items():
            ridx = [g2r[g] for g in members if g in g2r]
            if len(ridx) >= 2:
                rows = Sn[ridx]
                # background: mean cosine over a random sample of all source rows
                rng = np.random.default_rng(seed + c)
                bk = Sn[rng.choice(Sn.shape[0], size=min(500, Sn.shape[0]), replace=False)]
                bg = float((bk @ bk.T)[np.triu_indices(bk.shape[0], k=1)].mean())
                src_coh[name] = {"intra_cos": intra_cos(rows),
                                 "background_cos": bg,
                                 "coherence": intra_cos(rows) - bg,
                                 "n_in_source": len(ridx)}
        clusters.append({
            "cluster": c, "size": len(members),
            "learned_intra_cos": learned_coh,
            "example_genes": members[:25],
            "source_coherence": src_coh,
        })
    clusters.sort(key=lambda d: d["size"], reverse=True)
    return {
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
        "coords": coords.tolist(),
        "pca_coords": pca_coords.tolist(),
        "umap_coords": umap_coords.tolist() if umap_coords is not None else None,
        "projection_method": projection_method,
        "projection_note": projection_note,
        "umap_params": umap_params,
        "genes": genes,
        "clusters": clusters,
    }


def neighbor_examples(
    learned_by_gene: dict[str, np.ndarray],
    genes: list[str],
    sources: dict[str, SourceMatrix],
    example_genes: Sequence[str],
    *,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Side-by-side top-k neighbors in learned vs each source for example genes."""
    Ln = l2norm(np.stack([learned_by_gene[g] for g in genes], axis=0))
    gpos = {g: i for i, g in enumerate(genes)}
    src_norm = {name: ({g: i for i, g in enumerate(sm.genes)}, l2norm(sm.X), sm.genes)
                for name, sm in sources.items()}
    out = []
    for g in example_genes:
        if g not in gpos:
            continue
        row = {"gene": g, "learned": [], "sources": {}}
        sims = Ln @ Ln[gpos[g]]
        order = np.argsort(-sims)
        row["learned"] = [{"gene": genes[j], "cos": float(sims[j])}
                          for j in order[1:k + 1]]
        for name, (g2r, Sn, sgenes) in src_norm.items():
            if g not in g2r:
                continue
            ssims = Sn @ Sn[g2r[g]]
            sorder = np.argsort(-ssims)
            row["sources"][name] = [{"gene": sgenes[j], "cos": float(ssims[j])}
                                    for j in sorder[1:k + 1]]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_embeddings_analysis(
    art: RankerArtifacts,
    train_screens: Sequence,
    cohit: CoHitIndex,
    pathways: PathwayIndex | None = None,
    *,
    sources: Sequence[str],
    knn_ks: Sequence[int] = (10, 20, 50),
    n_clusters: int = 25,
    top_pairs: int = 200,
    example_genes: Sequence[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    from .loaders import trained_gene_set

    trained = trained_gene_set(train_screens)
    # Analysis gene universe: trained genes that exist in the vocab (all do),
    # excluding the <unk> slot. Sorted for determinism.
    genes = sorted(g for g in trained if g in art.stoi and art.stoi[g] != 0)
    learned_by_gene = {g: art.E[art.idx(g)] for g in genes}
    log.info("Embedding analysis over %d trained genes (vocab=%d).",
             len(genes), art.vocab_size)

    loaded: dict[str, SourceMatrix] = {}
    per_source = []
    for src in sources:
        sm = load_presage_source(src, genes)
        if sm is None:
            per_source.append({"source": src, "available": False})
            continue
        loaded[src] = sm
        res = compare_source(learned_by_gene, sm, knn_ks=knn_ks, seed=seed)
        res["available"] = True
        res["coverage_frac"] = len(sm.genes) / max(1, len(genes))
        per_source.append(res)

    novelty = novel_pairs(
        learned_by_gene, genes, loaded, cohit,
        pathways=pathways,
        top_n=top_pairs, seed=seed,
    ) if loaded else []

    clustering = cluster_and_project(
        learned_by_gene, genes, loaded, n_clusters=n_clusters, seed=seed,
    )

    # Default example genes: a few of the most frequent hits across train.
    if example_genes is None:
        from collections import Counter
        hit_counter: Counter = Counter()
        for s in train_screens:
            n = min(len(s.genes), len(s.hits))
            for i in range(n):
                if s.hits[i]:
                    hit_counter[s.genes[i]] += 1
        example_genes = [g for g, _ in hit_counter.most_common(15) if g in genes]

    examples = neighbor_examples(learned_by_gene, genes, loaded, example_genes, k=10)

    return {
        "n_trained_genes": len(genes),
        "vocab_size": art.vocab_size,
        "sources_requested": list(sources),
        "per_source": per_source,
        "novel_pairs": novelty,
        "clustering": clustering,
        "neighbor_examples": examples,
        "example_genes": list(example_genes),
    }


__all__ = [
    "linear_cka",
    "rsa_spearman",
    "knn_overlap",
    "ridge_r2",
    "compare_source",
    "novel_pairs",
    "cluster_and_project",
    "neighbor_examples",
    "run_embeddings_analysis",
]
