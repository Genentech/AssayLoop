"""Amortized, context-conditioned gene-ranking model.

A small Transformer encoder is trained offline on the BioGRID public train
split to predict per-gene relevance scores given a screen description and the
already-observed ``(gene, hit)`` active-learning context. At inference it slots
into the assayloop AL loop as a :class:`~assaybench.core.Model`.

Submodules:

- ``text_embed`` : screen-description text embeddings (Azure / local) + cache.
- ``data``       : gene vocab, MSE targets, context sampling, Dataset/collate.
- ``model``      : ``RankerNet`` (encoder + tied gene-embedding head).
- ``train``      : training loop, wandb logging, DAgger rounds, checkpoints.
"""

from __future__ import annotations

__all__ = ["text_embed", "data", "model", "train"]
