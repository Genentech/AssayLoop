"""Score a JSONL of (screen, ranked_genes) predictions.

Scores prediction files produced outside assayloop against the public
screen set's ground truth.
Each row in the input file must have at least ``dataset_name`` and
``ranked_genes`` (a list of gene symbols). We compute, per row:

- ``adjusted_ndcg@{10,50,100}`` and ``ndcg@{10,50,100}`` against the
  ground-truth relevance scores from the loaded public screen set.
- ``hits_in_top_k`` for several K.

Aggregate metrics are written to ``out_path`` (mean over screens).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..tasks import ScreenRecord, load_screens

log = logging.getLogger("assayloop.baselines.score")

_K_VALUES = [10, 50, 100]


def _index_screens(screens: list[ScreenRecord]) -> dict[str, ScreenRecord]:
    return {s.dataset_name: s for s in screens}


def _hits_in_top_k(ranked: list[str], hit_set: set[str], k: int) -> int:
    return sum(1 for g in ranked[:k] if g in hit_set)


def _score_row(row: dict[str, Any], screen: ScreenRecord) -> dict[str, Any]:
    ranked: list[str] = list(row.get("ranked_genes") or [])
    hit_set = {g for g, h in zip(screen.genes, screen.hits) if h}

    out: dict[str, Any] = {
        "dataset_name": screen.dataset_name,
        "split": screen.split,
        "organism": screen.organism,
        "n_ranked": len(ranked),
        "total_hits": screen.total_hits,
        "library_size": len(screen.genes),
    }
    for k in _K_VALUES:
        out[f"hits_in_top_{k}"] = int(_hits_in_top_k(ranked, hit_set, k))

    # Adjusted nDCG via assaybench.benchmark.metrics.RankingMetrics.
    try:
        from assaybench.benchmark.metrics import RankingMetrics  # public API

        rm = RankingMetrics(
            k_values=_K_VALUES,
            metric_groups=["adjusted_ndcg", "ndcg"],
            use_gene_mapper=False,
        )
        # ground truth: all (gene, score) pairs from the screen.
        result = rm.evaluate(
            predicted_genes=ranked,
            ground_truth_genes=list(screen.genes),
            relevance_scores=[float(x) for x in screen.relevance_scores],
            organism=screen.organism or "Homo sapiens",
        )
        for k in _K_VALUES:
            for key in (f"adjusted_ndcg@{k}", f"ndcg@{k}"):
                if key in result:
                    out[key] = float(result[key])
    except Exception as e:  # noqa: BLE001
        log.warning("RankingMetrics failed for %s: %s", screen.dataset_name, e)

    return out


def score_file(
    predictions_path: str | Path,
    *,
    out_path: str | Path = "output/baselines/scores.json",
    screen_set: str = "paper_test",
) -> dict[str, Any]:
    """Score a JSONL of predictions; write summary to ``out_path``."""
    in_path = Path(predictions_path)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    screens = load_screens(target_set=screen_set)
    by_name = _index_screens(screens)

    per_screen: list[dict[str, Any]] = []
    missing: list[str] = []

    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("dataset_name")
            if not name:
                continue
            screen = by_name.get(name)
            if screen is None:
                missing.append(name)
                continue
            per_screen.append(_score_row(row, screen))

    # Aggregate (mean over screens).
    agg: dict[str, float] = {}
    if per_screen:
        keys = set().union(*[set(r.keys()) for r in per_screen])
        for k in sorted(keys):
            vals = [r.get(k) for r in per_screen if isinstance(r.get(k), (int, float))]
            if vals and not isinstance(vals[0], bool):
                agg[f"mean_{k}"] = float(sum(vals) / len(vals))

    payload = {
        "input": str(in_path),
        "n_predictions": len(per_screen),
        "n_missing_screens": len(missing),
        "missing_screens": missing[:20],
        "aggregate": agg,
        "per_screen": per_screen,
    }
    out_p.write_text(json.dumps(payload, indent=2))
    return {k: v for k, v in payload.items() if k != "per_screen"} | {
        "out_path": str(out_p),
    }


__all__ = ["score_file"]
