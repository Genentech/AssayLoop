"""Placeholder gene-embedding providers for tests and smoke runs.

These let you exercise models / acquisitions end-to-end without the
(3.4 GB) PRESAGE cache. They are **not** meaningful biological
embeddings, and nothing selects them automatically when the cache is
missing -- that case raises ``MissingPresageCache``. Pass one of these
explicitly if you want it.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import GeneEmbeddingProvider


class OnehotGeneEmbedding(GeneEmbeddingProvider):
    """Stable per-gene hashed one-hot in a fixed-D space (uses the modulo
    of the gene-name hash as the column index)."""

    def __init__(self, dim: int = 256):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def name(self) -> str:
        return f"onehot_d{self._dim}"

    def embed(self, gene: str) -> np.ndarray | None:
        if not gene:
            return None
        h = hashlib.md5(gene.upper().encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % self._dim
        v = np.zeros(self._dim, dtype=np.float32)
        v[idx] = 1.0
        return v


class RandomGeneEmbedding(GeneEmbeddingProvider):
    """Deterministic random embedding (seeded by gene symbol). Useful
    for unit tests of downstream KNN / RF / UCB code."""

    def __init__(self, dim: int = 128):
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def name(self) -> str:
        return f"random_d{self._dim}"

    def embed(self, gene: str) -> np.ndarray | None:
        if not gene:
            return None
        h = hashlib.md5(gene.upper().encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=self._dim).astype(np.float32)


__all__ = ["OnehotGeneEmbedding", "RandomGeneEmbedding"]
