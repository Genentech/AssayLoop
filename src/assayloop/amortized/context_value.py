"""Offline counterfactual context-value diagnostic for the amortized ranker.

Measures how much giving the model an *informative* observed-gene context (a mix
that includes some true hits) improves its ranking of the **remaining** genes,
relative to an empty context. This is a single-step, no-AL-loop proxy for
"does observing hits help the model find the unseen hits."

For each screen we:
  1. pick an oracle context of ~``ctx_size`` observed genes (half true hits,
     half non-hits), leaving at least one hit remaining,
  2. score the remaining genes with an empty context (R0) and with the oracle
     context (R1),
  3. compare average precision (AP) and hits@batch of the remaining true hits.

``lift = AP(R1) - AP(R0)``. A mean lift near zero (and ~50% of screens positive)
means the model is effectively ignoring the AL context.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

log = logging.getLogger("assayloop.amortized.context_value")


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP of ``labels`` (0/1) ranked by ``scores`` (higher = better)."""
    from sklearn.metrics import average_precision_score

    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def context_value(
    *,
    checkpoint: str,
    screen_set: str = "public",
    device: str = "auto",
    n_screens: int = 60,
    ctx_size: int = 50,
    batch_size: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the offline context-value summary for a trained ranker."""
    from assaybench.core.types import Observation
    from ..models.amortized_ranker import AmortizedRankerModel
    from ..tasks import load_screens

    rng = random.Random(seed)
    model = AmortizedRankerModel(checkpoint=checkpoint, device=device)

    screens = sorted(load_screens(target_set=screen_set), key=lambda s: s.dataset_name)
    if n_screens and 0 < n_screens < len(screens):
        screens = rng.sample(screens, n_screens)

    ap_empty: list[float] = []
    ap_ctx: list[float] = []
    h_empty: list[float] = []
    h_ctx: list[float] = []
    lifts: list[float] = []
    n_pos = 0

    for s in screens:
        genes = [str(g) for g in s.genes]
        hits = [1 if h else 0 for h in s.hits]
        n = min(len(genes), len(hits))
        genes, hits = genes[:n], hits[:n]
        hit_pos = [i for i in range(n) if hits[i]]
        non_pos = [i for i in range(n) if not hits[i]]
        # Need >=2 hits (>=1 for context, >=1 remaining) and some non-hits.
        if len(hit_pos) < 2 or not non_pos:
            continue

        n_ctx_hits = min(len(hit_pos) - 1, max(1, ctx_size // 2))
        ctx_hit_idx = rng.sample(hit_pos, n_ctx_hits)
        n_ctx_non = min(len(non_pos), max(0, ctx_size - n_ctx_hits))
        ctx_non_idx = rng.sample(non_pos, n_ctx_non) if n_ctx_non > 0 else []
        ctx_idx = set(ctx_hit_idx) | set(ctx_non_idx)

        remaining = [i for i in range(n) if i not in ctx_idx]
        rem_genes = [genes[i] for i in remaining]
        rem_labels = np.array([hits[i] for i in remaining], dtype=np.float64)
        if rem_labels.sum() == 0:
            continue

        ctx = s.context() if hasattr(s, "context") else {}
        obs = [
            Observation(candidate=genes[i], label={"hit": bool(hits[i])})
            for i in ctx_idx
        ]

        p0 = model.predict([], rem_genes, ctx)
        p1 = model.predict(obs, rem_genes, ctx)
        s0 = np.array([p0.scores[g] for g in rem_genes], dtype=np.float64)
        s1 = np.array([p1.scores[g] for g in rem_genes], dtype=np.float64)

        a0 = _average_precision(rem_labels, s0)
        a1 = _average_precision(rem_labels, s1)
        k = min(batch_size, len(rem_genes))
        hh0 = float(rem_labels[np.argsort(-s0)[:k]].sum())
        hh1 = float(rem_labels[np.argsort(-s1)[:k]].sum())

        ap_empty.append(a0)
        ap_ctx.append(a1)
        h_empty.append(hh0)
        h_ctx.append(hh1)
        lifts.append(a1 - a0)
        if a1 - a0 > 0:
            n_pos += 1

    used = len(lifts)
    if used == 0:
        log.warning("context_value: no eligible screens (need >=2 hits each).")
    return {
        "screen_set": screen_set,
        "encoder_type": model.encoder_type,
        "ctx_size": ctx_size,
        "n_screens": used,
        "mean_ap_empty": float(np.mean(ap_empty)) if ap_empty else None,
        "mean_ap_ctx": float(np.mean(ap_ctx)) if ap_ctx else None,
        "mean_ap_lift": float(np.mean(lifts)) if lifts else None,
        "frac_positive": (n_pos / used) if used else None,
        "mean_hits_at_batch_empty": float(np.mean(h_empty)) if h_empty else None,
        "mean_hits_at_batch_ctx": float(np.mean(h_ctx)) if h_ctx else None,
        "mean_hits_at_batch_lift": (
            float(np.mean(np.array(h_ctx) - np.array(h_empty))) if h_ctx else None
        ),
    }


__all__ = ["context_value"]
