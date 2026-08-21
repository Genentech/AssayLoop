"""Gene-embedding initialisations for AssayFormer (paper §4.1.1, ablated §6.7.1).

Stage 1 of AssayFormer's training is *gene token initialisation*: the gene
factors ``v_g`` are seeded from a geometry learned on historical screens rather
than at random, and the paper finds this the single most important factor in
downstream performance (EF 1.06 for GenePT+PCA vs 3.83 for BPMF after SFT).

This module supplies the matrix-factorisation family of those initialisations,
all aligned to the ranker's :class:`GeneVocab`:

- :class:`BPMFFactors` -- posterior-mean ``V`` from a trained BPMF checkpoint
  (the default, and the one AssayLoop ships).
- :func:`svd_vocab_factors` -- truncated SVD of the marginal-centered hit
  matrix (paper's "SVD" ablation).
- :func:`mf_vocab_factors` -- masked ALS matrix completion (the "MF" and, with
  ``normalize=True``, "MF-Sphere" ablations).

The remaining initialisations live elsewhere: ``random`` is inline in
:mod:`assayloop.amortized.train`, and GenePT / K562 Perturb-seq PCA are in
:mod:`assayloop.data.gene_embeddings.presage`.

Genes with no measured screen (and the pad/``<unk>`` id 0) get a zero factor
and a ``logit(marginal)`` bias instead, so a gene the geometry cannot speak to
still carries a sensible context-free prior.
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("assayloop.amortized.gene_factors")


def marginal_hit_freq(examples, vocab, *, alpha: float = 20.0) -> np.ndarray:
    """Per-vocab-id smoothed marginal hit frequency from training examples.

    ``freq[id] = (hits + alpha*global) / (measured + alpha)``; ids never measured
    (and ``<unk>``) get the global mean. Returns float32 array of len(vocab).
    """
    measured: dict[int, int] = defaultdict(int)
    hit: dict[int, int] = defaultdict(int)
    for ex in examples:
        for vid, h in zip(ex.gene_idx.tolist(), ex.hit.tolist()):
            measured[vid] += 1
            if h:
                hit[vid] += 1
    tot_m = sum(measured.values())
    tot_h = sum(hit.values())
    glob = (tot_h / tot_m) if tot_m else 0.5
    out = np.full(len(vocab), glob, dtype=np.float32)
    for vid, m in measured.items():
        out[vid] = (hit[vid] + alpha * glob) / (m + alpha)
    return out


def svd_vocab_factors(
    examples,
    vocab,
    d_gene: int,
    marginal: "np.ndarray | None" = None,
    *,
    seed: int = 0,
    target_norm: float = 1.0,
    normalize: bool = True,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Pickle-free stand-in for BPMF's gene geometry via a truncated SVD of the
    marginal-centered gene x screen hit matrix.

    Builds ``M[g, s] = hit(g, s) - marginal[g]`` for *measured* (gene, screen)
    pairs (0 elsewhere), then takes the top-``d_gene`` left singular vectors
    scaled by ``sqrt(singular value)``. The per-gene ``bias`` is 0 for covered
    genes (pure bilinear û·V_g, matching BPMF) and ``logit(marginal)`` for
    uncovered genes (marginal already removed from ``M``, so ``V`` and the bias
    don't double-count it).

    ``normalize`` (default True) per-gene L2-normalizes the covered factors onto
    a **constant-norm shell** before scaling -- this reproduces BPMF's signature
    geometry (median/mean row-norm ~= 1), where ranking is decided by *direction*
    (context-controllable) rather than by fixed per-gene magnitude. Raw SVD
    instead leaves heavy-tailed norms: a few high-norm genes dominate the ranking
    regardless of context and the low-norm bulk is inert, which starves RL of
    leverage. Set ``normalize=False`` for the legacy heterogeneous-norm geometry.

    ``target_norm`` sets the shell radius (``normalize=True``) or rescales the
    *mean* nonzero row-norm (``normalize=False``).

    Returns ``(V_vocab[len(vocab), d_gene], bias_vocab[len(vocab)])``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    n_genes = len(vocab)
    if marginal is None:
        marginal = marginal_hit_freq(examples, vocab)
    marginal = np.asarray(marginal, dtype=np.float32)

    rows, cols, vals = [], [], []
    for j, ex in enumerate(examples):
        gid = np.asarray(ex.gene_idx, dtype=np.int64)
        h = np.asarray(ex.hit, dtype=np.float32)
        rows.append(gid)
        cols.append(np.full(gid.shape, j, dtype=np.int64))
        vals.append(h - marginal[gid])
    if not rows:
        raise ValueError("svd_vocab_factors: no training examples to factorize.")
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals).astype(np.float32)
    n_screens = len(examples)
    M = coo_matrix((vals, (rows, cols)), shape=(n_genes, n_screens)).tocsr()

    k = int(d_gene)
    if k >= min(M.shape):
        raise ValueError(
            f"--d-gene {k} must be < min(n_genes={n_genes}, n_screens={n_screens}) "
            "for the SVD init."
        )
    try:
        u, s, _ = svds(M, k=k, random_state=seed)
    except TypeError:  # older scipy lacks random_state
        u, s, _ = svds(M, k=k)
    order = np.argsort(s)[::-1]  # svds returns ascending singular values
    u, s = u[:, order], s[order]
    V_vocab = (u * np.sqrt(np.maximum(s, 0.0))[None, :]).astype(np.float32)
    V_vocab[0] = 0.0  # keep padding/<unk> id at 0

    rn = np.linalg.norm(V_vocab, axis=1)
    covered = rn > 1e-8
    tgt = float(target_norm) if target_norm and target_norm > 0 else 1.0
    if normalize:
        # Project covered genes onto a constant-norm shell (BPMF-like): keep the
        # learned direction, equalize magnitude so context controls the ranking.
        V_vocab[covered] *= (tgt / rn[covered])[:, None]
    elif target_norm and target_norm > 0:
        mean_rn = float(rn[covered].mean()) if covered.any() else 0.0
        if mean_rn > 0:
            V_vocab *= (tgt / mean_rn)

    # Bias = 0 for covered genes (pure bilinear û·V_g, no static shortcut -> the
    # encoder must rank via context; matches BPMF and preserves RL-ability);
    # logit(marginal) cold-start prior for uncovered genes only.
    m = np.clip(marginal, 1e-4, 1.0 - 1e-4)
    bias_vocab = np.log(m / (1.0 - m)).astype(np.float32)
    bias_vocab[covered] = 0.0
    bias_vocab[0] = 0.0
    return V_vocab, bias_vocab


def mf_vocab_factors(
    examples,
    vocab,
    d_gene: int,
    marginal: "np.ndarray | None" = None,
    *,
    reg: float = 0.1,
    n_iters: int = 15,
    seed: int = 0,
    target_norm: float = 1.0,
    normalize: bool = True,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Pickle-free BPMF surrogate via **masked** low-rank matrix *completion*
    (regularized ALS) of the gene x screen hit matrix.

    Unlike :func:`svd_vocab_factors` (which zero-fills unobserved entries and so
    treats "not measured" as "not a hit"), this fits ``M[g, s] = hit - marginal``
    over the **observed entries only** -- i.e. true matrix completion, like
    Gibbs BPMF's probit likelihood. Alternating least squares:

        V_g <- (Σ_{s∈obs(g)} u_s u_sᵀ + reg·I)⁻¹ Σ_{s∈obs(g)} M[g,s] u_s
        u_s <- (Σ_{g∈obs(s)} V_g V_gᵀ + reg·I)⁻¹ Σ_{g∈obs(s)} M[g,s] V_g

    Genes never measured shrink toward 0 (handled by ``reg``); the screen
    factors ``u`` are discarded.

    ``normalize`` (default True) per-gene L2-normalizes the measured factors onto
    a **constant-norm shell** (BPMF-like: median/mean row-norm ~= 1) so ranking
    is decided by *direction* (context-controllable) rather than by a few
    high-norm genes. ALS leaves the bulk of weakly-observed genes near the
    origin (heavy-tailed norms) which starves RL of leverage; normalization keeps
    the learned directions but equalizes magnitude. ``target_norm`` is the shell
    radius (``normalize=True``) or the target mean row-norm (``normalize=False``).
    Per-gene ``bias`` is 0 for measured genes and ``logit(marginal)`` otherwise.

    Returns ``(V_vocab[len(vocab), d_gene], bias_vocab[len(vocab)])``.
    """
    n_genes = len(vocab)
    if marginal is None:
        marginal = marginal_hit_freq(examples, vocab)
    marginal = np.asarray(marginal, dtype=np.float32)

    g_list, s_list, v_list = [], [], []
    for j, ex in enumerate(examples):
        gid = np.asarray(ex.gene_idx, dtype=np.int64)
        h = np.asarray(ex.hit, dtype=np.float32)
        g_list.append(gid)
        s_list.append(np.full(gid.shape, j, dtype=np.int64))
        v_list.append(h - marginal[gid])
    if not g_list:
        raise ValueError("mf_vocab_factors: no training examples to factorize.")
    g = np.concatenate(g_list)
    s = np.concatenate(s_list)
    val = np.concatenate(v_list).astype(np.float64)
    n_screens = len(examples)
    K = int(d_gene)

    def _groups(key):
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uniq, starts = np.unique(ks, return_index=True)
        ends = np.append(starts[1:], len(ks))
        return order, uniq, starts, ends

    g_order, g_uniq, g_st, g_en = _groups(g)
    s_order, s_uniq, s_st, s_en = _groups(s)

    rng = np.random.default_rng(seed)
    V = (rng.standard_normal((n_genes, K)) * 0.1)
    U = (rng.standard_normal((n_screens, K)) * 0.1)
    V[0] = 0.0
    eyeK = reg * np.eye(K)

    for _ in range(int(n_iters)):
        # V-step: solve each gene's factor from its observed screens
        for gid, a, b in zip(g_uniq, g_st, g_en):
            idx = g_order[a:b]
            us = U[s[idx]]
            A = us.T @ us + eyeK
            V[gid] = np.linalg.solve(A, us.T @ val[idx])
        V[0] = 0.0
        # U-step: solve each screen's latent from its observed genes
        for sid, a, b in zip(s_uniq, s_st, s_en):
            idx = s_order[a:b]
            vg = V[g[idx]]
            A = vg.T @ vg + eyeK
            U[sid] = np.linalg.solve(A, vg.T @ val[idx])

    # Genes with no observations keep no factor (ALS never touched them).
    measured = np.zeros(n_genes, dtype=bool)
    measured[g_uniq] = True
    V[~measured] = 0.0
    V_vocab = V.astype(np.float32)
    rn = np.linalg.norm(V_vocab, axis=1)
    nz = measured & (rn > 1e-8)
    tgt = float(target_norm) if target_norm and target_norm > 0 else 1.0
    if normalize:
        # Project measured genes onto a constant-norm shell (BPMF-like): keep the
        # learned direction, equalize magnitude so context controls the ranking.
        V_vocab[nz] *= (tgt / rn[nz])[:, None]
    elif target_norm and target_norm > 0:
        mean_rn = float(rn[measured].mean()) if measured.any() else 0.0
        if mean_rn > 0:
            V_vocab *= (tgt / mean_rn)
    V_vocab[0] = 0.0

    # Bias = 0 for covered genes so their score is the *pure* bilinear û·V_g
    # (no static shortcut -> the encoder must rank via context; matches BPMF's
    # vocab_factors and keeps the base a good RL substrate). Uncovered genes get
    # logit(marginal) as a cold-start prior.
    m = np.clip(marginal, 1e-4, 1.0 - 1e-4)
    bias_vocab = np.log(m / (1.0 - m)).astype(np.float32)
    bias_vocab[measured] = 0.0
    bias_vocab[0] = 0.0
    return V_vocab, bias_vocab


class BPMFFactors:
    """Posterior-mean gene factors from a trained BPMF checkpoint.

    This is the initialisation AssayLoop uses (paper §4.1.1 / Appendix B.1): the
    probit matrix factorisation is fit to the historical hit matrix by Gibbs
    sampling (:mod:`assayloop.models.bpmf_model`), and the posterior mean of
    ``V`` seeds AssayFormer's gene embedding table.
    """

    def __init__(
        self,
        result_path: str | Path,
        vocab,
        *,
        marginal: np.ndarray | None = None,
    ):
        with open(result_path, "rb") as f:
            res = pickle.load(f)
        self.K = int(res.K)
        # Posterior mean over the retained Gibbs samples (n_genes x K). The
        # per-sample tensor can be ~1GB, so keep only the mean.
        self.V_mean = np.asarray(res.V_samples).mean(axis=0).astype(np.float32)

        gene_to_row = {g: i for i, g in enumerate(res.gene_names)}
        self.id_to_row = np.full(len(vocab), -1, dtype=np.int64)
        for sym, vid in vocab.stoi.items():
            r = gene_to_row.get(sym)
            if r is not None:
                self.id_to_row[vid] = r
        n_mapped = int((self.id_to_row >= 0).sum())

        self.marginal = (marginal.astype(np.float64) if marginal is not None
                         else np.full(len(vocab), 0.5))
        log.info("BPMFFactors: K=%d, %d/%d vocab genes mapped to BPMF",
                 self.K, n_mapped, len(vocab))

    def vocab_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Scoring-head parameters anchored to BPMF, aligned to the vocab.

        Returns ``(V_vocab[len(vocab), K], bias_vocab[len(vocab)])`` where:

        - BPMF-known genes get ``V = posterior-mean V`` and ``bias = 0`` so their
          score is the *pure* bilinear ``û · V_g`` -- there is no static per-gene
          term, which forces the encoder's ``û`` (the screen latent) to carry the
          ranking signal, i.e. to do the in-context inference.
        - Out-of-BPMF genes (and ``<unk>``/pad id 0) get ``V = 0`` and
          ``bias = logit(marginal)``, a sensible static prior for genes the
          factorisation never saw.
        """
        n = len(self.id_to_row)
        V_vocab = np.zeros((n, self.K), dtype=np.float32)
        m = np.clip(self.marginal, 1e-4, 1.0 - 1e-4)
        bias = np.log(m / (1.0 - m)).astype(np.float32)
        mask = self.id_to_row >= 0
        V_vocab[mask] = self.V_mean[self.id_to_row[mask]]
        bias[mask] = 0.0
        V_vocab[0] = 0.0
        bias[0] = 0.0
        return V_vocab, bias


__all__ = [
    "BPMFFactors",
    "marginal_hit_freq",
    "mf_vocab_factors",
    "svd_vocab_factors",
]
