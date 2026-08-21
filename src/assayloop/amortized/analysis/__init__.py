"""Post-hoc analysis of a trained amortized ranker.

``embeddings`` compares the model's learned gene-embedding table against
PRESAGE knowledge sources, recovering known biology and flagging candidate
novel, screen-driven associations. ``loaders`` supplies the checkpoint and
knowledge-source readers it runs on.
"""

from __future__ import annotations

__all__ = ["loaders", "embeddings"]
