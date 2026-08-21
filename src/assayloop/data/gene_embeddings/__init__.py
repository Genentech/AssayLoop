"""Gene-embedding providers for assayloop models.

`GeneEmbeddingProvider` is the abstract interface. Implementations:

- `PresageGeneEmbedding` (default): reads the 3.4 GB PRESAGE cache.
- `OnehotGeneEmbedding`: for smoke tests. Each gene gets a one-hot column on a
  stable index, so every gene is orthogonal to every other and any metric or
  model built on "which genes are similar" is meaningless. Nothing falls back
  to this automatically when PRESAGE is missing -- pass it explicitly.
- `RandomGeneEmbedding`: deterministic random embeddings, useful for
  unit tests of downstream models.
"""

from .base import GeneEmbeddingProvider
from .fallback import OnehotGeneEmbedding, RandomGeneEmbedding
from .presage import (
    PRESAGE_SOURCES,
    PresageGeneEmbedding,
    presage_concat,
)
from .ortholog import map_mgi_to_hgnc

__all__ = [
    "GeneEmbeddingProvider",
    "PresageGeneEmbedding",
    "OnehotGeneEmbedding",
    "RandomGeneEmbedding",
    "PRESAGE_SOURCES",
    "presage_concat",
    "map_mgi_to_hgnc",
]
