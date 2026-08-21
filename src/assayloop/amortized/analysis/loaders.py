"""Loading + shared helpers for ranker analysis.

Pulls together the three inputs the analysis needs:

1. the trained ranker artifacts (``gene_embeddings.npy`` + ``vocab.json`` +
   ``config.json``) and, on demand, the live model for probing;
2. the set of genes that actually received gradients (union of genes across the
   ``public_train`` screens) -- only these have meaningful learned vectors;
3. PRESAGE knowledge-source embeddings restricted to a common gene list.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("assayloop.amortized.analysis.loaders")


# ---------------------------------------------------------------------------
# Ranker artifacts
# ---------------------------------------------------------------------------


@dataclass
class RankerArtifacts:
    out_dir: Path
    itos: list[str]                 # vocab id -> gene symbol (index 0 = <unk>)
    stoi: dict[str, int]
    E: np.ndarray                   # (vocab, d_gene) learned gene embeddings
    arch: dict[str, Any]
    cfg: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def idx(self, gene: str) -> int:
        i = self.stoi.get(gene)
        if i is None:
            i = self.stoi.get(str(gene).upper(), 0)
        return i


def load_ranker(checkpoint: str | Path) -> RankerArtifacts:
    ckpt = Path(checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    vocab = json.loads((ckpt / "vocab.json").read_text())
    itos = vocab["itos"]
    stoi = {g: i for i, g in enumerate(itos)}
    E = np.load(ckpt / "gene_embeddings.npy").astype(np.float32)
    if E.shape[0] != len(itos):
        log.warning("Embedding rows (%d) != vocab size (%d).", E.shape[0], len(itos))
    return RankerArtifacts(
        out_dir=ckpt, itos=itos, stoi=stoi, E=E,
        arch=cfg.get("arch", {}), cfg=cfg,
    )


def build_model(checkpoint: str | Path, *, device: str = "auto"):
    """Build the live inference model (for probing). Lazy torch import."""
    from ...models.amortized_ranker import AmortizedRankerModel

    return AmortizedRankerModel(checkpoint=str(checkpoint), device=device)


# ---------------------------------------------------------------------------
# Screens + trained-gene set
# ---------------------------------------------------------------------------


def load_train_screens() -> list:
    from ...tasks import load_screens

    return load_screens(target_set="public_train")


def trained_gene_set(screens: Sequence) -> set[str]:
    """Genes that appear in >=1 training screen (so they received gradients)."""
    genes: set[str] = set()
    for s in screens:
        genes.update(s.genes)
    return genes


# ---------------------------------------------------------------------------
# Empirical co-hit statistics (shared by embeddings novelty + probe validation)
# ---------------------------------------------------------------------------


@dataclass
class CoHitIndex:
    """Per-screen in-library and hit gene sets, optionally tagged by a group
    key (e.g. ``cleaned_phenotype``), supporting empirical co-hit lift:

    ``lift(a, b) = P(a,b both hit | both in-library) / (rate(a) * rate(b))``
    estimated over the screens where both are in-library.
    """

    in_lib: list[set[str]] = field(default_factory=list)
    hit: list[set[str]] = field(default_factory=list)
    group: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, screens: Sequence, *, group_field: str = "cleaned_phenotype") -> "CoHitIndex":
        in_lib, hit, group = [], [], []
        for s in screens:
            genes = list(s.genes)
            hits = [bool(h) for h in s.hits]
            n = min(len(genes), len(hits))
            in_lib.append(set(genes[:n]))
            hit.append({genes[i] for i in range(n) if hits[i]})
            gv = getattr(s, group_field, "") or ""
            group.append(str(gv))
        return cls(in_lib=in_lib, hit=hit, group=group)

    def _screen_ids(self, group: str | None) -> list[int]:
        if group is None or group == "" or group.lower() == "all":
            return list(range(len(self.in_lib)))
        return [i for i, g in enumerate(self.group) if g == group]

    def lift(self, a: str, b: str, *, group: str | None = None) -> dict[str, float]:
        ids = self._screen_ids(group)
        n_both_lib = n_a = n_b = n_both_hit = 0
        for i in ids:
            lib = self.in_lib[i]
            if a in lib and b in lib:
                n_both_lib += 1
                ah = a in self.hit[i]
                bh = b in self.hit[i]
                n_a += ah
                n_b += bh
                n_both_hit += ah and bh
        if n_both_lib == 0:
            return {"n_screens": 0, "p_both": 0.0, "lift": float("nan")}
        ra = n_a / n_both_lib
        rb = n_b / n_both_lib
        p_both = n_both_hit / n_both_lib
        denom = ra * rb
        lift = (p_both / denom) if denom > 0 else float("nan")
        return {
            "n_screens": n_both_lib, "n_both_hit": n_both_hit,
            "rate_a": ra, "rate_b": rb, "p_both": p_both, "lift": lift,
        }

    def groups(self) -> list[str]:
        from collections import Counter

        c = Counter(g for g in self.group if g)
        return [g for g, _ in c.most_common()]


# ---------------------------------------------------------------------------
# Pathway co-occurrence statistics
# ---------------------------------------------------------------------------


@dataclass
class PathwayIndex:
    """Gene-set membership across the bundled GMT pathway collections.

    The index is intentionally small and literal: it answers whether two genes
    are listed in any of the same pathways, and returns a few pathway examples
    for dashboard tables. Missing GMT files are treated as an empty index so
    ranker analysis remains optional with respect to pathway annotations.
    """

    source_to_gene_sets: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    n_pathways_by_source: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, sources: Sequence[str] | None = None) -> "PathwayIndex":
        from ... import config

        root = Path(config.GENE_SETS_PATH)
        if not root.exists():
            log.warning("Gene-set directory not found at %s; pathway co-occurrence disabled.", root)
            return cls()

        if sources:
            paths: list[Path] = []
            for src in sources:
                src_l = src.lower()
                matches = [
                    p for p in sorted(root.glob("*.gmt"))
                    if src_l in p.name.lower() or src_l in p.stem.lower()
                ]
                paths.extend(matches[:1])
        else:
            paths = sorted(root.glob("*.gmt"))

        source_to_gene_sets: dict[str, dict[str, frozenset[str]]] = {}
        n_pathways_by_source: dict[str, int] = {}
        for path in paths:
            gene_to_sets: dict[str, set[str]] = {}
            n_sets = 0
            try:
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        set_name = parts[0]
                        n_sets += 1
                        for gene in parts[2:]:
                            if gene:
                                gene_to_sets.setdefault(gene.upper(), set()).add(set_name)
            except OSError as e:
                log.warning("Could not load gene-set file %s: %s", path, e)
                continue
            source = path.stem
            source_to_gene_sets[source] = {
                g: frozenset(names) for g, names in gene_to_sets.items()
            }
            n_pathways_by_source[source] = n_sets

        return cls(
            source_to_gene_sets=source_to_gene_sets,
            n_pathways_by_source=n_pathways_by_source,
        )

    @property
    def available(self) -> bool:
        return bool(self.source_to_gene_sets)

    @property
    def sources(self) -> list[str]:
        return sorted(self.source_to_gene_sets)

    def gene_summary(self, gene: str, *, max_examples: int = 5) -> dict[str, Any]:
        gene_u = gene.upper()
        pathways: set[str] = set()
        sources: list[str] = []
        for source, gene_to_sets in self.source_to_gene_sets.items():
            sets = gene_to_sets.get(gene_u)
            if not sets:
                continue
            sources.append(source)
            pathways.update(sets)
        examples = sorted(pathways)[:max_examples]
        return {
            "pathway_n": len(pathways),
            "pathway_examples": examples,
            "pathway_sources": sources,
        }

    def cooccurrence(self, a: str, b: str, *, max_examples: int = 5) -> dict[str, Any]:
        a_u, b_u = a.upper(), b.upper()
        shared: set[str] = set()
        shared_sources: list[str] = []
        both_annotated_sources = 0
        any_annotated_sources = 0
        for source, gene_to_sets in self.source_to_gene_sets.items():
            sa = gene_to_sets.get(a_u)
            sb = gene_to_sets.get(b_u)
            if sa or sb:
                any_annotated_sources += 1
            if sa and sb:
                both_annotated_sources += 1
                inter = sa & sb
                if inter:
                    shared_sources.append(source)
                    shared.update(inter)

        if both_annotated_sources:
            status = "both_annotated"
        elif any_annotated_sources:
            status = "partial"
        elif self.available:
            status = "unannotated"
        else:
            status = "unavailable"

        examples = sorted(shared)[:max_examples]
        return {
            "pathway_cooccurs": bool(shared),
            "pathway_n": len(shared),
            "pathway_examples": examples,
            "pathway_sources": shared_sources,
            "pathway_status": status,
        }


# ---------------------------------------------------------------------------
# PRESAGE source loading restricted to a gene list
# ---------------------------------------------------------------------------


@dataclass
class SourceMatrix:
    source: str
    genes: list[str]                # genes present in BOTH this source and the query
    X: np.ndarray                   # (n_common, D) source embeddings (aligned to genes)
    dim: int


def load_presage_source(source: str, genes: Sequence[str]) -> SourceMatrix | None:
    """Load one PRESAGE source restricted to ``genes`` (order preserved for the
    subset that is known). Returns ``None`` if the source is missing."""
    from ...data.gene_embeddings.presage import (
        PresageGeneEmbedding,
        MissingPresageCache,
        MissingPresageSource,
    )

    try:
        prov = PresageGeneEmbedding(source=source)
    except (MissingPresageCache, MissingPresageSource) as e:
        log.warning("PRESAGE source %r unavailable: %s", source, e)
        return None
    X, mask = prov.embed_batch(list(genes))
    common = [g for g, m in zip(genes, mask) if m]
    if not common:
        log.warning("PRESAGE source %r shares no genes with the query set.", source)
        return None
    Xc = X[mask].astype(np.float32)
    return SourceMatrix(source=source, genes=common, X=Xc, dim=Xc.shape[1])


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def l2norm(X: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)


def cosine_topk(Xn: np.ndarray, k: int, *, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbors (excluding self) for L2-normalized rows ``Xn``.

    Returns ``(idx, sim)`` each shape ``(N, k)``. Blocked to bound memory.
    """
    N = Xn.shape[0]
    k = min(k, max(1, N - 1))
    idx_out = np.empty((N, k), dtype=np.int64)
    sim_out = np.empty((N, k), dtype=np.float32)
    for start in range(0, N, block):
        end = min(start + block, N)
        sims = Xn[start:end] @ Xn.T              # (b, N)
        for r in range(end - start):
            sims[r, start + r] = -np.inf         # exclude self
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        part_sims = sims[rows, part]
        order = np.argsort(-part_sims, axis=1)
        idx_out[start:end] = part[rows, order]
        sim_out[start:end] = part_sims[rows, order]
    return idx_out, sim_out


__all__ = [
    "RankerArtifacts",
    "load_ranker",
    "build_model",
    "load_train_screens",
    "trained_gene_set",
    "CoHitIndex",
    "PathwayIndex",
    "SourceMatrix",
    "load_presage_source",
    "l2norm",
    "cosine_topk",
]
