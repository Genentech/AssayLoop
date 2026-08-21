"""Abstract interface for a gene-embedding provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class GeneEmbeddingProvider(ABC):
    """Return one numpy vector per gene symbol."""

    @abstractmethod
    def embed(self, gene: str) -> np.ndarray | None:
        """Return the embedding for ``gene`` or ``None`` if unknown."""

    def embed_batch(self, genes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised embedding lookup.

        Returns ``(X, known_mask)`` where ``X`` is shape ``(N, D)`` and
        ``known_mask`` is a 0/1 array. Unknown genes get a zero vector.
        """
        first = None
        for g in genes:
            v = self.embed(g)
            if v is not None:
                first = v
                break
        if first is None:
            return np.zeros((len(genes), 1)), np.zeros(len(genes), dtype=bool)
        D = first.shape[0]
        X = np.zeros((len(genes), D), dtype=np.float32)
        mask = np.zeros(len(genes), dtype=bool)
        for i, g in enumerate(genes):
            v = self.embed(g)
            if v is not None:
                X[i] = v
                mask[i] = True
        return X, mask

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Subclasses MUST set this once known."""
        raise NotImplementedError

    def mean_embedding(self) -> np.ndarray | None:
        """Global mean embedding over all known genes, or ``None`` if the
        provider can't supply one. Subtracting this de-anisotropises text-style
        embeddings (e.g. GenePT/ada), whose vectors otherwise sit in a narrow
        high-cosine cone. Callers should treat ``None`` as "don't centre"."""
        return None

    def sample_matrix(self, n: int, *, seed: int = 0) -> np.ndarray | None:
        """Return ``(<=n, D)`` embeddings sampled from the provider's gene
        universe, for estimating a random-batch baseline. ``None`` if the
        provider can't enumerate its universe."""
        return None

    def name(self) -> str:
        return self.__class__.__name__
