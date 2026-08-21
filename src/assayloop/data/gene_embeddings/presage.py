"""PRESAGE gene-embedding loader.

`PRESAGE <https://github.com/Genentech/PRESAGE>`_ ships a 3.4 GB cache
(Zenodo `15587986 <https://zenodo.org/records/15587986>`_) of
pre-computed gene embeddings from many knowledge sources.

Run ``scripts/fetch_presage_cache.sh`` once to download / unpack.

Cache layout (after unpack):

::

    <PRESAGE_CACHE>/
        data/
        splits/
        configs/
        pathway_embeddings/<source>.embeddings.pkl
        other_embeddings/<source>.embeddings.pkl

Each ``.embeddings.pkl`` is a pickled pandas DataFrame whose index is
HGNC gene symbols and whose columns are the per-gene embedding
features. Dimension varies per source.

This loader is permissive: if a source file is absent we raise a
``MissingPresageSource`` with a clear hint. If the whole cache is
absent we raise ``MissingPresageCache``.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)
from typing import Sequence

import numpy as np

from ... import config
from .base import GeneEmbeddingProvider


# A best-effort catalog of source short-names we expect to find. New
# sources can be added by passing a custom filename. Keys are
# lowercase canonical short names; values are search hints used when
# resolving on disk.
PRESAGE_SOURCES: dict[str, list[str]] = {
    "genept": [
        "genept_ada", "genept", "gene_pt", "ncbi_genept", "gpt_ncbi",
    ],
    "biogpt": ["biogpt", "bio_gpt"],
    "esm2": ["esm2", "esm"],
    "depmap": ["crisprgeneeffectdepmap", "depmap", "crispr_gene_effect", "crispr"],
    "msigdb": ["h.all", "hallmark", "msigdb"],
    "stringdb": ["stringdb", "string"],
    "periscope": ["periscope", "ops"],
    "funk_ops": ["funk_ops", "funk", "funk_ops_2022"],
    "reactome": ["reactome"],
    "go_bp": ["go_bp", "go.bp", "biological_process"],
    "go_mf": ["go_mf", "go.mf", "molecular_function"],
    "go_cc": ["go_cc", "go.cc", "cellular_component"],
}


class MissingPresageCache(FileNotFoundError):
    pass


class MissingPresageSource(FileNotFoundError):
    pass


def _cache_root() -> Path:
    root = Path(config.PRESAGE_CACHE_PATH)
    return root


def _find_source_file(source: str) -> Path:
    """Resolve ``source`` short-name to a file under the cache."""
    root = _cache_root()
    if not root.exists():
        raise MissingPresageCache(
            f"PRESAGE cache not found at {root}. "
            "Run scripts/fetch_presage_cache.sh to download (3.4 GB), "
            "or set ASSAYLOOP_PRESAGE_CACHE to an existing cache root."
        )
    hints = PRESAGE_SOURCES.get(source.lower(), [source.lower()])
    candidate_dirs = [
        root / "pathway_embeddings",
        root / "other_embeddings",
        root,
    ]
    candidate_dirs = [d for d in candidate_dirs if d.exists()]
    # 1. Direct hits.
    for d in candidate_dirs:
        for hint in hints + [source]:
            for ext in (".embeddings.pkl", ".pkl", ".pickle"):
                p = d / f"{hint}{ext}"
                if p.is_file():
                    return p
    # 2. Fuzzy, CASE-INSENSITIVE substring match over the directory listing
    #    (PRESAGE filenames are mixed-case, e.g. GenePT_ada / CRISPRGeneEffectDepMap).
    for d in candidate_dirs:
        entries = sorted(p for p in d.iterdir()
                         if p.is_file() and p.suffix in (".pkl", ".pickle"))
        for hint in hints + [source]:
            h = hint.lower()
            for p in entries:
                if h in p.name.lower():
                    return p
    raise MissingPresageSource(
        f"PRESAGE source {source!r} not found under {root}. "
        f"Tried hints={hints!r} in {[str(d) for d in candidate_dirs]}."
    )


def _load_source_df(source: str):
    """Load one source's pickle as a pandas DataFrame indexed by gene."""
    path = _find_source_file(source)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # PRESAGE pickles can be:
    # - pd.DataFrame indexed by gene
    # - dict[str, np.ndarray]
    # - torch.Tensor with a sibling gene-list file
    if hasattr(obj, "index") and hasattr(obj, "values"):  # DataFrame
        return obj
    if isinstance(obj, dict):
        import pandas as pd

        genes = list(obj.keys())
        vals = np.stack([np.asarray(obj[g], dtype=np.float32) for g in genes])
        return pd.DataFrame(vals, index=genes)
    if hasattr(obj, "numpy") and callable(obj.numpy):  # torch tensor
        import pandas as pd

        arr = obj.numpy()
        genes_file = path.with_name(path.stem + ".genes.json")
        if not genes_file.is_file():
            raise MissingPresageSource(
                f"Loaded a tensor from {path} but no sibling "
                f"{genes_file.name}; cannot key by gene."
            )
        import json

        with open(genes_file) as f:
            genes = json.load(f)
        return pd.DataFrame(arr, index=genes)
    raise TypeError(
        f"Unsupported PRESAGE pickle payload type {type(obj)} at {path}."
    )


class PresageGeneEmbedding(GeneEmbeddingProvider):
    """A GeneEmbeddingProvider backed by one PRESAGE source.

    Args:
        source: short name of the embedding source (default ``"genept"``).
        cache_unknown: keep a record of unknown gene queries so the runner
            can report coverage. Default ``True``.
        gene_normalizer: optional callable ``str -> str`` applied to
            the query gene before lookup (e.g. uppercase, MGI->HGNC).
    """

    def __init__(
        self,
        source: str = "genept",
        cache_unknown: bool = True,
        gene_normalizer=None,
    ):
        self.source = source
        self._df = _load_source_df(source)
        self._values = np.asarray(self._df.values, dtype=np.float32)
        # Build a fast index (case-insensitive).
        idx = list(self._df.index)
        self._gene_to_row = {g: i for i, g in enumerate(idx)}
        self._gene_to_row_upper = {g.upper(): i for i, g in enumerate(idx)}
        self._dim = self._values.shape[1]
        self._unknown: set[str] = set() if cache_unknown else set()
        self._gene_normalizer = gene_normalizer

    @property
    def dim(self) -> int:
        return self._dim

    def mean_embedding(self) -> np.ndarray | None:
        mean = getattr(self, "_mean_vec", None)
        if mean is None:
            mean = self._values.mean(axis=0)
            self._mean_vec = mean
        return mean

    def sample_matrix(self, n: int, *, seed: int = 0) -> np.ndarray | None:
        total = self._values.shape[0]
        if total == 0:
            return None
        rng = np.random.default_rng(seed)
        k = min(n, total)
        idx = rng.choice(total, size=k, replace=False)
        return self._values[idx]

    def name(self) -> str:
        return f"presage:{self.source}"

    def embed(self, gene: str) -> np.ndarray | None:
        if not gene:
            return None
        if self._gene_normalizer is not None:
            gene = self._gene_normalizer(gene)
        idx = self._gene_to_row.get(gene)
        if idx is None:
            idx = self._gene_to_row_upper.get(gene.upper())
        if idx is None:
            self._unknown.add(gene)
            return None
        return self._values[idx]

    def compute_coverage(self,genes_list: list[str]) -> dict[str, int]:
        """Compute coverage stats on a list of genes."""
        known = 0
        unknown = 0
        for g in genes_list:
            if self.embed(g) is not None:
                known += 1
            else:
                unknown += 1
        return {"known": known, "unknown": unknown}
        
    def coverage_stats(self) -> dict[str, int]:
        return {
            "n_known_in_source": len(self._gene_to_row),
            "n_unknown_queries": len(self._unknown),
        }


def genept_vocab_factors(
    vocab,
    d_gene: int,
    marginal: "np.ndarray | None" = None,
    *,
    source: str = "genept",
    gene_normalizer=None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Vocab-aligned gene factors from a PRESAGE source, PCA-reduced to ``d_gene``.

    Mirrors :meth:`BPMFFactors.vocab_factors` so it can drive the same frozen
    bilinear head: covered genes get a (whitened) PCA projection of their PRESAGE
    embedding and ``bias = 0`` (so the score is the pure bilinear ``û · V_g``);
    uncovered genes (and ``<unk>``/pad id 0) get ``V = 0`` and
    ``bias = logit(marginal)`` -- a sensible static prior for genes the source
    cannot represent.

    Args:
        vocab: a ``GeneVocab`` (needs ``.stoi`` and ``len``).
        d_gene: target factor dimension (PCA components fit on covered genes).
        marginal: optional per-vocab-id marginal hit frequency for the bias of
            uncovered genes; defaults to 0.5 (-> bias 0).
        source: PRESAGE source short-name (default ``"genept"``).

    Returns ``(V_vocab[len(vocab), d_gene], bias_vocab[len(vocab)])`` (float32).
    """
    prov = PresageGeneEmbedding(source=source, gene_normalizer=gene_normalizer)
    n = len(vocab)

    # Gather raw embeddings for covered vocab genes.
    rows: list[int] = []
    raw: list[np.ndarray] = []
    for sym, vid in vocab.stoi.items():
        if vid == 0:
            continue  # reserved <unk>/pad
        v = prov.embed(sym)
        if v is not None:
            rows.append(int(vid))
            raw.append(np.asarray(v, dtype=np.float64))
    if not raw:
        raise ValueError(
            f"PRESAGE source {source!r} covered 0 of {n} vocab genes; "
            "cannot build gene factors."
        )

    X = np.stack(raw, axis=0)                       # (n_cov, D_raw)
    if d_gene > X.shape[0]:
        raise ValueError(
            f"d_gene={d_gene} exceeds the {X.shape[0]} covered genes available "
            f"from PRESAGE source {source!r}; lower --d-gene."
        )
    mean = X.mean(axis=0)
    Xc = X - mean
    # Whitened PCA: unit-variance components so the bilinear head starts at a
    # sane scale regardless of the raw source magnitude.
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(d_gene)
    proj = U[:, :k] * np.sqrt(max(X.shape[0] - 1, 1))   # (n_cov, k), unit-var cols

    V_vocab = np.zeros((n, k), dtype=np.float32)
    rows_arr = np.asarray(rows, dtype=np.int64)
    V_vocab[rows_arr] = proj.astype(np.float32)

    if marginal is not None:
        m = np.clip(np.asarray(marginal, dtype=np.float64), 1e-4, 1.0 - 1e-4)
        bias = np.log(m / (1.0 - m)).astype(np.float32)
    else:
        bias = np.zeros(n, dtype=np.float32)
    bias[rows_arr] = 0.0
    V_vocab[0] = 0.0
    bias[0] = 0.0
    return V_vocab, bias


def presage_concat(
    sources: Sequence[str],
    *,
    pca_dim: int | None = None,
    gene_normalizer=None,
) -> "ConcatPresageEmbedding":
    """Return a GeneEmbeddingProvider that concatenates embeddings from
    multiple PRESAGE sources. Optionally PCA-reduce the concat vector
    to ``pca_dim`` dimensions (fit on the genes common to all sources).
    """
    providers = [PresageGeneEmbedding(source=s, gene_normalizer=gene_normalizer)
                 for s in sources]
    return ConcatPresageEmbedding(providers, pca_dim=pca_dim)


class ConcatPresageEmbedding(GeneEmbeddingProvider):
    def __init__(
        self,
        providers: list[PresageGeneEmbedding],
        pca_dim: int | None = None,
    ):
        if not providers:
            raise ValueError("Need at least one source")
        self.providers = providers
        self._sub_dims = [p.dim for p in providers]
        self._dim = sum(self._sub_dims) if pca_dim is None else pca_dim
        self._pca_dim = pca_dim
        self._pca = None
        self._mean = None
        if pca_dim is not None:
            self._fit_pca()

    @property
    def dim(self) -> int:
        return self._dim

    def name(self) -> str:
        return "presage_concat:" + "+".join(p.source for p in self.providers)

    def _raw_embed(self, gene: str) -> tuple[np.ndarray, bool]:
        parts = []
        any_known = False
        for p, d in zip(self.providers, self._sub_dims):
            v = p.embed(gene)
            if v is None:
                parts.append(np.zeros(d, dtype=np.float32))
            else:
                parts.append(v.astype(np.float32))
                any_known = True
        return np.concatenate(parts), any_known

    def embed(self, gene: str) -> np.ndarray | None:
        raw, known = self._raw_embed(gene)
        if not known:
            return None
        if self._pca is None:
            return raw
        return self._pca @ (raw - self._mean)

    def _fit_pca(self) -> None:
        """Fit PCA on the genes common to all source DataFrames."""
        common_genes = set(self.providers[0]._gene_to_row.keys())
        for p in self.providers[1:]:
            common_genes &= set(p._gene_to_row.keys())
        if len(common_genes) < (self._pca_dim or 0) + 1:
            self._pca = None
            self._mean = None
            self._dim = sum(self._sub_dims)
            return
        gene_list = sorted(common_genes)
        # Build NxD matrix lazily.
        rows = []
        for g in gene_list:
            raw, _ = self._raw_embed(g)
            rows.append(raw)
        X = np.stack(rows, axis=0)
        mean = X.mean(axis=0)
        Xc = X - mean
        # Use truncated SVD for stability.
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self._pca = Vt[: self._pca_dim]
        self._mean = mean
        self._dim = self._pca_dim


def k562_perturbseq_vocab_factors(
    vocab,
    d_gene: int,
    marginal: "np.ndarray | None" = None,
    dataset: str = "replogle_k562_gw",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Gene factors from K562 Perturb-seq DEG rankings, PCA-reduced to d_gene.

    Builds a reciprocal-rank matrix from the PRESAGE DEG lists: for each
    perturbation i and gene j, score = 1/rank(j in perturbation i's DEGs).
    PCA on this matrix gives a gene embedding capturing perturbation-response
    similarity — genes with correlated perturbation profiles cluster together.
    """
    import time
    t0 = time.time()
    cache = _cache_root()
    deg_path = cache / "data" / dataset / "degs" / "merged.degs.json"
    if not deg_path.is_file():
        raise FileNotFoundError(
            "K562 DEG file not found at %s. Need PRESAGE cache with %s data."
            % (deg_path, dataset))

    with open(deg_path) as f:
        deg_data = json.load(f)

    perturbations = sorted(deg_data.keys())
    deg_genes = sorted(set(g for v in deg_data.values() for g in v))
    gene_to_col = {g: i for i, g in enumerate(deg_genes)}
    n_pert = len(perturbations)
    n_genes = len(deg_genes)

    # Build reciprocal-rank matrix: (n_genes, n_pert)
    # Each column = one perturbation's reciprocal-rank profile over genes
    rank_matrix = np.zeros((n_genes, n_pert), dtype=np.float32)
    for pi, pname in enumerate(perturbations):
        for rank, g in enumerate(deg_data[pname]):
            col = gene_to_col.get(g)
            if col is not None:
                rank_matrix[col, pi] = 1.0 / (rank + 1)

    # PCA to d_gene
    mean = rank_matrix.mean(axis=0)
    Xc = rank_matrix - mean
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(d_gene)
    proj = U[:, :k] * np.sqrt(max(n_genes - 1, 1))

    # Map to vocab — covered genes get K562 factors, uncovered get
    # nearest-neighbor imputation from GenePT embeddings
    n = len(vocab)
    V_vocab = np.zeros((n, k), dtype=np.float32)
    covered_vids = []
    uncovered_vids = []
    uncovered_syms = []
    for sym, vid in vocab.stoi.items():
        if vid == 0:
            continue
        col = gene_to_col.get(sym) or gene_to_col.get(sym.upper())
        if col is not None:
            V_vocab[vid] = proj[col].astype(np.float32)
            covered_vids.append(vid)
        else:
            uncovered_vids.append(vid)
            uncovered_syms.append(sym)

    # Impute uncovered genes via GenePT nearest neighbor
    n_imputed = 0
    if uncovered_vids:
        try:
            prov = PresageGeneEmbedding(source="genept")
            # Build GenePT embeddings for covered and uncovered genes
            cov_genept = []
            cov_k562 = []
            for sym, vid in vocab.stoi.items():
                if vid == 0:
                    continue
                col = gene_to_col.get(sym) or gene_to_col.get(sym.upper())
                if col is None:
                    continue
                gpt = prov.embed(sym)
                if gpt is not None:
                    cov_genept.append(np.asarray(gpt, dtype=np.float32))
                    cov_k562.append(V_vocab[vid])

            if cov_genept:
                cov_genept_arr = np.stack(cov_genept)  # (n_cov, D_genept)
                cov_k562_arr = np.stack(cov_k562)      # (n_cov, k)
                # Normalize for cosine similarity
                norms = np.linalg.norm(cov_genept_arr, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                cov_genept_normed = cov_genept_arr / norms

                for uid, usym in zip(uncovered_vids, uncovered_syms):
                    gpt = prov.embed(usym)
                    if gpt is None:
                        continue
                    gpt_arr = np.asarray(gpt, dtype=np.float32)
                    gpt_norm = gpt_arr / max(np.linalg.norm(gpt_arr), 1e-8)
                    sims = cov_genept_normed @ gpt_norm
                    nn_idx = int(np.argmax(sims))
                    V_vocab[uid] = cov_k562_arr[nn_idx]
                    n_imputed += 1
        except Exception as e:
            log.warning("GenePT imputation failed (%s); uncovered genes get zero factors.", e)

    bias = np.zeros(n, dtype=np.float32)
    V_vocab[0] = 0.0

    log.info("k562_perturbseq_vocab_factors: %d/%d covered, %d imputed via GenePT, "
             "%d zero, d_gene=%d, %d perturbations (%.1fs).",
             len(covered_vids), n, n_imputed,
             len(uncovered_vids) - n_imputed, k, n_pert, time.time() - t0)
    return V_vocab, bias


__all__ = [
    "PRESAGE_SOURCES",
    "MissingPresageCache",
    "MissingPresageSource",
    "PresageGeneEmbedding",
    "ConcatPresageEmbedding",
    "presage_concat",
    "k562_perturbseq_vocab_factors",
    "genept_vocab_factors",
]
