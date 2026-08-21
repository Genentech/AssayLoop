"""Regenerate the paper's curated 20-screen test and validation manifests.

The selector operates on the public ``Genentech/assaybench`` BioGRID dataset
and can build curated 20-screen subsets for either ``yearfold0 == "test"`` or
``yearfold0 == "validation"``.

Selection signal
----------------

For every candidate screen we read the top-100 ranked gene lists emitted by:

- ``gemini-3-pro`` (the LLM signal used for the curated-set floor), and
- Oracle kNN, Embedding kNN, and phenotype-hit-frequency baselines (retained
  in the YAML as audit metrics, not as eligibility floors).

We score each screen by AnDCG@100 with ``assaybench.RankingMetrics``.

Selection criteria
------------------

1. Genome-wide CRISPR library: ``num_genes`` in ``[18000, 22000]``.
2. Enough positive signal for AL curves: ``num_hits >= 50``.
3. Not trivially saturated: ``hit_rate <= 0.15``.
4. Signal floor: ``gemini-3-pro`` AnDCG@100 must be at least ``0.05``. This
   drops screens on which no method -- LLM or learned -- gets any traction,
   where an active-learning curve would measure noise rather than method
   quality. It is a selection criterion, so it is applied identically to
   every method the paper compares.
5. Phenotype stratification: slots are allocated by ``cleaned_phenotype`` in
   proportion to the eligible pool, with a soft cap of six screens per
   phenotype. The cap is relaxed only when the eligible pool cannot otherwise
   satisfy ``n_target``.
6. Within each phenotype, greedy max-min TF-IDF diversity over screen
   descriptions, lightly weighted by Gemini AnDCG@100.

There is intentionally no LLM-vs-baseline winner balancing. Baseline scores,
``best_method``, and ``llm_lead`` remain in the output so the set can be
audited after selection.

Usage::

    uv run python -m assayloop.scripts.select_default_public_screens \
        --split test --n-target 20

    uv run python -m assayloop.scripts.select_default_public_screens \
        --split validation --n-target 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from assaybench.data.screen_sets import manifest_path

from assayloop import config

log = logging.getLogger("assayloop.scripts.select_default_public_screens")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AssayBench's committed baseline predictions (the `benchmarking/predictions`
# tree of the assaybench source checkout). Not part of the installed wheel:
# set ASSAYBENCH_PREDICTIONS to a clone.
_PREDICTIONS_ROOT = Path(
    os.getenv("ASSAYBENCH_PREDICTIONS", "")
    or (config.OUTPUT_PATH / "assaybench_predictions")
)

_LLM_METHOD = "gemini-3-pro"
_LLM_PATH = _PREDICTIONS_ROOT / "llm" / "gemini-3-pro.json"

_BASELINE_METHODS = {
    "oracle_knn": _PREDICTIONS_ROOT / "knn" / "Oracle__kNN.json",
    "embedding_knn": _PREDICTIONS_ROOT / "knn" / "Embedding__kNN.json",
    "phenotype_hit_freq": _PREDICTIONS_ROOT
    / "baselines"
    / "baseline__phenotype-hit-freq.json",
}

_PREDICTION_SPLIT_BY_YEARFOLD = {
    "test": "test",
    "validation": "val",
    "train": "train",
}

_NUM_GENES_MIN = 18_000
_NUM_GENES_MAX = 22_000
_NUM_HITS_MIN = 50
_MAX_HIT_RATE = 0.15
_MIN_GEMINI_ANDCG = 0.05
_MAX_PER_PHENOTYPE = 6
_QUALITY_WEIGHT = 0.5


def _manifest_name(split_value: str) -> str:
    """The shipped manifest this split corresponds to."""
    if split_value == "test":
        return "assayloop-test"
    if split_value == "validation":
        return "assayloop-validation"
    return f"assayloop-{split_value}"


def _default_output_path(split_value: str) -> Path:
    """Where a regenerated manifest is written.

    Not the shipped location. The manifests the code reads live inside
    assaybench (:mod:`assaybench.data.screen_sets`) because the screen list is
    part of the benchmark definition; writing there from here would mean
    writing into an installed package. Copy the file over the shipped one to
    adopt a regenerated set -- :func:`main` prints the destination.
    """
    out_dir = config.OUTPUT_PATH / "screen_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_manifest_name(split_value)}.yaml"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_predictions(path: Path, prediction_split: str) -> dict[str, list[str]]:
    """Return ``dataset_name -> top-100 predicted genes`` for one split."""
    with path.open() as f:
        data = json.load(f)

    out: dict[str, list[str]] = {}
    for ds_name, entries in data["records_by_dataset"].items():
        for entry in entries:
            if entry.get("split") != prediction_split:
                continue
            preds = entry.get("predicted_genes") or []
            if not preds:
                continue
            out[ds_name] = list(preds[:100])
            break
    return out


def _load_public_split_screens(split_value: str) -> list[dict[str, Any]]:
    """Load rows from the public AssayBench dataset for one ``yearfold0``."""
    from datasets import load_dataset

    ds = load_dataset("Genentech/assaybench", "biogrid")["train"]
    rows: list[dict[str, Any]] = []
    for ex in ds:
        if ex.get("yearfold0") != split_value:
            continue
        rows.append(dict(ex))
    return rows


def _prediction_split_for_yearfold(split_value: str) -> str:
    try:
        return _PREDICTION_SPLIT_BY_YEARFOLD[split_value]
    except KeyError as e:
        valid = ", ".join(sorted(_PREDICTION_SPLIT_BY_YEARFOLD))
        raise ValueError(f"Unsupported split {split_value!r}; expected one of {valid}") from e


# ---------------------------------------------------------------------------
# Per-screen scoring
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _score_value(value: Any, default: float = 0.0) -> float:
    return float(value) if _is_finite_number(value) else default


def _score_screens(
    rows: list[dict[str, Any]],
    preds_by_method: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Compute per-screen AnDCG@100 for each method."""
    from assaybench import RankingMetrics

    rm = RankingMetrics(
        k_values=[100],
        use_gene_mapper=False,
        metric_groups=["adjusted_ndcg"],
    )

    method_names = list(preds_by_method.keys())
    records: list[dict[str, Any]] = []

    for ex in rows:
        ds_name = ex["dataset_name"]
        gt_genes = ex["relevance_genes"]
        gt_scores = ex["relevance_scores"]
        n_genes = len(gt_genes)
        n_hits = sum(1 for h in ex.get("hit") or [] if h)

        scores: dict[str, float] = {}
        for method in method_names:
            preds = preds_by_method[method].get(ds_name)
            if preds is None:
                scores[method] = float("nan")
                continue
            try:
                result = rm.evaluate(preds, gt_genes, gt_scores)
                scores[method] = float(result.get("adjusted_ndcg@100", float("nan")))
            except Exception as e:  # noqa: BLE001
                log.warning("metric failed for %s on %s: %s", method, ds_name, e)
                scores[method] = float("nan")

        records.append(
            {
                "dataset_name": ds_name,
                "phenotype": ex.get("phenotype", "") or "",
                "cleaned_phenotype": ex.get("cleaned_phenotype", "") or "",
                "cell_line": ex.get("cell_line", "") or "",
                "cell_type": ex.get("cell_type", "") or "",
                "library_methodology": ex.get("library_methodology", "") or "",
                "library_type": ex.get("library_type", "") or "",
                "screen_type": ex.get("screen_type", "") or "",
                "screen_category": ex.get("screen_category", "") or "",
                "screen_rationale": ex.get("screen_rationale", "") or "",
                "condition_clause": ex.get("condition_clause", "") or "",
                "condition_name": ex.get("condition_name", "") or "",
                "duration": ex.get("duration", "") or "",
                "experimental_setup": ex.get("experimental_setup", "") or "",
                "author": ex.get("author", "") or "",
                "source_id": ex.get("source_id", "") or "",
                "ranking_rationale": ex.get("ranking_rationale", "") or "",
                "significance_criteria": ex.get("significance_criteria", "") or "",
                "num_genes": int(n_genes),
                "num_hits": int(n_hits),
                "andcg_at_100": scores,
            }
        )
    return records


def _load_scored_records(split_value: str) -> list[dict[str, Any]]:
    prediction_split = _prediction_split_for_yearfold(split_value)

    log.info("Loading AssayBench public split yearfold0=%r ...", split_value)
    rows = _load_public_split_screens(split_value)
    log.info("  %d rows", len(rows))

    log.info("Loading predictions for prediction split %r ...", prediction_split)
    preds: dict[str, dict[str, list[str]]] = {}
    preds[_LLM_METHOD] = _load_predictions(_LLM_PATH, prediction_split)
    log.info("  %s: %d preds", _LLM_METHOD, len(preds[_LLM_METHOD]))
    for method, path in _BASELINE_METHODS.items():
        preds[method] = _load_predictions(path, prediction_split)
        log.info("  %s: %d preds", method, len(preds[method]))

    log.info("Scoring AnDCG@100 per screen per method ...")
    return _score_screens(rows, preds)


# ---------------------------------------------------------------------------
# Filtering and selection
# ---------------------------------------------------------------------------


def _eligible(rec: dict[str, Any]) -> bool:
    if not (_NUM_GENES_MIN <= rec["num_genes"] <= _NUM_GENES_MAX):
        return False
    if rec["num_hits"] < _NUM_HITS_MIN:
        return False
    if rec["num_hits"] / max(rec["num_genes"], 1) > _MAX_HIT_RATE:
        return False
    gemini = rec["andcg_at_100"].get(_LLM_METHOD)
    if not _is_finite_number(gemini) or float(gemini) < _MIN_GEMINI_ANDCG:
        return False
    return True


def _llm_lead(rec: dict[str, Any]) -> bool:
    """True iff Gemini strictly beats every finite baseline score."""
    scores = rec["andcg_at_100"]
    llm = scores.get(_LLM_METHOD)
    if not _is_finite_number(llm):
        return False
    others = [
        scores[method]
        for method in _BASELINE_METHODS
        if _is_finite_number(scores.get(method))
    ]
    if not others:
        return True
    return float(llm) > max(float(v) for v in others)


def _annotate_selection_metrics(records: list[dict[str, Any]]) -> None:
    for rec in records:
        scores = rec["andcg_at_100"]
        rec["gemini_andcg"] = _score_value(scores.get(_LLM_METHOD))
        rec["gemini_signal_pass"] = rec["gemini_andcg"] >= _MIN_GEMINI_ANDCG

        finite_pairs = [
            (method, float(value))
            for method, value in scores.items()
            if _is_finite_number(value)
        ]
        if finite_pairs:
            best_method, best_andcg = max(finite_pairs, key=lambda kv: kv[1])
        else:
            best_method, best_andcg = None, 0.0
        rec["best_method"] = best_method
        rec["best_andcg"] = best_andcg
        rec["llm_lead"] = _llm_lead(rec)


def _description(rec: dict[str, Any]) -> str:
    parts = [
        rec.get("phenotype") or "",
        rec.get("condition_clause") or "",
        rec.get("cell_line") or "",
        rec.get("cell_type") or "",
        rec.get("library_methodology") or "",
    ]
    return " | ".join(p for p in parts if p)


def _max_min_select(
    candidates: list[dict[str, Any]],
    n_target: int,
    *,
    quality_key: str = "gemini_andcg",
    quality_weight: float = _QUALITY_WEIGHT,
) -> list[dict[str, Any]]:
    """Quality-weighted greedy max-min on TF-IDF over screen descriptions."""
    if n_target <= 0:
        return []
    if len(candidates) <= n_target:
        return list(candidates)

    def quality_for(item: dict[str, Any]) -> float:
        return _score_value(item.get(quality_key))

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_distances
    except ImportError:
        if quality_weight <= 0:
            return sorted(candidates, key=lambda c: (_description(c), c["dataset_name"]))[:n_target]
        return sorted(candidates, key=lambda c: -quality_for(c))[:n_target]

    descs = [_description(c) for c in candidates]
    vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
    x = vec.fit_transform(descs)
    distances = cosine_distances(x)

    quality = np.array([quality_for(c) for c in candidates], dtype=float)
    q_min, q_max = float(quality.min()), float(quality.max())
    if q_max > q_min:
        quality_norm = (quality - q_min) / (q_max - q_min)
    else:
        quality_norm = np.ones_like(quality)

    if quality_weight <= 0:
        seed = int(distances.mean(axis=1).argmax())
    else:
        seed = int(quality.argmax())
    chosen = [seed]
    while len(chosen) < n_target:
        min_dists = distances[chosen].min(axis=0)
        score = min_dists + quality_weight * quality_norm
        score[chosen] = -np.inf
        chosen.append(int(score.argmax()))
    return [candidates[i] for i in chosen]


def _allocate(
    n_target: int,
    weights: dict[str, float],
    caps: dict[str, int] | None = None,
) -> dict[str, int]:
    """Distribute slots across buckets by largest remainder with caps."""
    caps = caps or {}
    if not weights or n_target <= 0:
        return {k: 0 for k in weights}

    total_capacity = sum(caps.get(k, n_target) for k in weights)
    n_target = min(n_target, total_capacity)

    total_weight = sum(weights.values()) or 1.0
    raw = {k: n_target * (w / total_weight) for k, w in weights.items()}
    slots = {k: min(int(math.floor(raw[k])), caps.get(k, n_target)) for k in weights}

    while sum(slots.values()) < n_target:
        receivers = [k for k in weights if slots[k] < caps.get(k, n_target)]
        if not receivers:
            break
        key = max(receivers, key=lambda k: (raw[k] - math.floor(raw[k]), weights[k]))
        slots[key] += 1
        raw[key] = math.floor(raw[key])

    return slots


def select_screens(
    records: list[dict[str, Any]],
    n_target: int,
) -> list[dict[str, Any]]:
    """Run split-local phenotype-stratified selection."""
    _annotate_selection_metrics(records)
    eligible = [r for r in records if _eligible(r)]
    log.info(
        "Eligibility: %d / %d screens pass size/hits/rate + Gemini signal filters",
        len(eligible),
        len(records),
    )
    if not eligible:
        return []
    if len(eligible) < n_target:
        log.warning(
            "Eligible pool has only %d screens; requested %d",
            len(eligible),
            n_target,
        )

    pheno_counts = Counter(r["cleaned_phenotype"] or "(unknown)" for r in eligible)
    pheno_caps = {k: min(_MAX_PER_PHENOTYPE, v) for k, v in pheno_counts.items()}
    pheno_alloc = _allocate(
        n_target,
        {k: float(v) for k, v in pheno_counts.items()},
        caps=pheno_caps,
    )
    log.info("Phenotype distribution in eligible pool: %s", dict(pheno_counts))
    log.info("Phenotype allocation: %s", pheno_alloc)

    quality_weight = _QUALITY_WEIGHT
    chosen: list[dict[str, Any]] = []
    chosen_names: set[str] = set()
    for pheno, slots in sorted(pheno_alloc.items(), key=lambda kv: (-kv[1], kv[0])):
        if slots <= 0:
            continue
        pool = [
            r
            for r in eligible
            if (r["cleaned_phenotype"] or "(unknown)") == pheno
            and r["dataset_name"] not in chosen_names
        ]
        selected = _max_min_select(
            pool,
            slots,
            quality_key="gemini_andcg",
            quality_weight=quality_weight,
        )
        log.info("  %s: %d slots; chose %d from %d pool", pheno, slots, len(selected), len(pool))
        chosen.extend(selected)
        chosen_names.update(r["dataset_name"] for r in selected)

    if len(chosen) < min(n_target, len(eligible)):
        remaining = [r for r in eligible if r["dataset_name"] not in chosen_names]
        top_up = _max_min_select(
            remaining,
            min(n_target, len(eligible)) - len(chosen),
            quality_key="gemini_andcg",
            quality_weight=quality_weight,
        )
        log.info("Top-up chose %d additional screens from %d remaining", len(top_up), len(remaining))
        chosen.extend(top_up)

    return chosen[:n_target]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _summarize_for_yaml(rec: dict[str, Any]) -> dict[str, Any]:
    scores = rec["andcg_at_100"]
    return {
        "dataset_name": rec["dataset_name"],
        "phenotype": rec["phenotype"],
        "cleaned_phenotype": rec["cleaned_phenotype"],
        "cell_line": rec["cell_line"],
        "cell_type": rec["cell_type"],
        "library_methodology": rec["library_methodology"],
        "library_type": rec["library_type"],
        "screen_category": rec["screen_category"],
        "condition_name": rec["condition_name"],
        "condition_clause": rec["condition_clause"],
        "duration": rec["duration"],
        "author": rec["author"],
        "source_id": rec["source_id"],
        "num_genes": rec["num_genes"],
        "num_hits": rec["num_hits"],
        "andcg_at_100": {
            k: (round(float(v), 4) if _is_finite_number(v) else None)
            for k, v in scores.items()
        },
        "gemini_signal_pass": bool(rec.get("gemini_signal_pass")),
        "best_method": rec.get("best_method"),
        "llm_lead": bool(rec.get("llm_lead")),
    }


def _build_output_obj(
    chosen: list[dict[str, Any]],
    *,
    split_value: str,
    n_target: int,
) -> dict[str, Any]:
    row_count_note = "334 entries" if split_value == "test" else "validation entries"
    floor_text = f"Gemini AnDCG@100 >= {_MIN_GEMINI_ANDCG:.2f}"
    return {
        "version": 2,
        "description": (
            f"Sequential-screening {split_value} set used in the AssayLoop "
            f"paper: {n_target} genome-wide screens curated from the "
            f"AssayBench public {split_value} split (Genentech/assaybench, "
            f"biogrid, yearfold0='{split_value}', {row_count_note}). This is "
            f"NOT the whole public {split_value} split -- it is a "
            f"diversity-stratified subset with a {floor_text} signal floor, "
            "so screens no method can beat random on do not dominate the "
            "average. Baselines are scored for audit only; no "
            "LLM-vs-baseline winner balancing is applied. The selection "
            "script is scripts/select_default_public_screens.py in the "
            "AssayLoop repo."
        ),
        "source": {
            "dataset": "Genentech/assaybench",
            "config": "biogrid",
            "split_field": "yearfold0",
            "split_value": split_value,
        },
        "selection_criteria": {
            "num_genes_min": _NUM_GENES_MIN,
            "num_genes_max": _NUM_GENES_MAX,
            "num_hits_min": _NUM_HITS_MIN,
            "max_hit_rate": _MAX_HIT_RATE,
            "gemini_signal_floor": True,
            "min_gemini_andcg": _MIN_GEMINI_ANDCG,
            "winner_balance": False,
            "max_per_phenotype": _MAX_PER_PHENOTYPE,
            "max_per_phenotype_is_soft": True,
            "quality_metric": f"{_LLM_METHOD} adjusted_ndcg@100",
            "quality_weight": _QUALITY_WEIGHT,
            "scoring_methods": {
                "llm": _LLM_METHOD,
                "baselines": list(_BASELINE_METHODS.keys()),
            },
            "stratification": (
                "Per cleaned_phenotype, allocate slots proportional to the "
                "eligible split distribution with a soft cap of six per phenotype "
                "(relaxed only if needed to reach n_target); "
                "within each phenotype, diversify via greedy max-min on TF-IDF "
                "of (phenotype | condition_clause | cell_line | cell_type | "
                "library_methodology), lightly weighted by gemini-3-pro "
                "AnDCG@100. No LLM-vs-baseline winner balancing is applied."
            ),
            "n_target": n_target,
        },
        "screens": [_summarize_for_yaml(r) for r in chosen],
    }


def _print_markdown_summary(chosen: list[dict[str, Any]]) -> None:
    cols = [_LLM_METHOD] + list(_BASELINE_METHODS.keys())
    header = (
        ["dataset_name", "phenotype", "cell_line", "num_genes", "num_hits"]
        + cols
        + ["gemini_pass", "best_method", "llm_lead"]
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for rec in chosen:
        scores = rec["andcg_at_100"]
        row = [
            rec["dataset_name"],
            (rec["cleaned_phenotype"] or "")[:30],
            (rec["cell_line"] or "")[:18],
            f"{rec['num_genes']:>5d}",
            f"{rec['num_hits']:>5d}",
        ] + [
            f"{scores[m]:.3f}" if _is_finite_number(scores.get(m)) else "NA"
            for m in cols
        ] + [
            "yes" if rec.get("gemini_signal_pass") else "no",
            str(rec.get("best_method") or ""),
            "yes" if rec.get("llm_lead") else "no",
        ]
        print("| " + " | ".join(row) + " |")


def _dump_per_screen_tsv(records: list[dict[str, Any]], path: Path) -> None:
    cols = [_LLM_METHOD] + list(_BASELINE_METHODS.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(
            "\t".join(
                [
                    "dataset_name",
                    "cleaned_phenotype",
                    "cell_line",
                    "num_genes",
                    "num_hits",
                ]
                + cols
            )
            + "\n"
        )
        for rec in records:
            scores = rec["andcg_at_100"]
            f.write(
                "\t".join(
                    [
                        rec["dataset_name"],
                        rec["cleaned_phenotype"],
                        rec["cell_line"],
                        str(rec["num_genes"]),
                        str(rec["num_hits"]),
                    ]
                    + [
                        f"{scores[m]:.4f}" if _is_finite_number(scores.get(m)) else ""
                        for m in cols
                    ]
                )
                + "\n"
            )
    log.info("Wrote per-screen TSV to %s", path)


def _write_selection(
    records: list[dict[str, Any]],
    *,
    split_value: str,
    n_target: int,
    output: Path,
    dry_run: bool,
) -> int:
    log.info("Selecting %d screens for split %s ...", n_target, split_value)
    chosen = select_screens(records, n_target)
    if not chosen:
        log.error("No screens selected for split %s", split_value)
        return 1

    log.info("Selected %d screens", len(chosen))
    _print_markdown_summary(chosen)

    out_obj = _build_output_obj(
        chosen,
        split_value=split_value,
        n_target=n_target,
    )

    if dry_run:
        print("\n--- DRY RUN (no file written) ---")
        print(yaml.safe_dump(out_obj, sort_keys=False))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(out_obj, sort_keys=False))
        log.info("Wrote %s (%d screens)", output, len(chosen))
        log.info(
            "This is not the manifest the code reads. To adopt it, copy over "
            "the shipped one at %s",
            manifest_path(_manifest_name(split_value)),
        )
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output YAML path. Defaults to "
            "$ASSAYLOOP_OUTPUT/screen_sets/assayloop-<split>.yaml; copy it "
            "over the shipped manifest in assaybench to adopt it."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Public yearfold0 split to curate.",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=20,
        help="Number of screens to select.",
    )
    parser.add_argument(
        "--per-screen-tsv",
        type=Path,
        default=None,
        help="Optional path to dump per-screen AnDCG@100 for audit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = _load_scored_records(args.split)
    if args.per_screen_tsv:
        _dump_per_screen_tsv(records, args.per_screen_tsv)
    return _write_selection(
        records,
        split_value=args.split,
        n_target=args.n_target,
        output=args.output or _default_output_path(args.split),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
