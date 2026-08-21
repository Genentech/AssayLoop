"""Shared index over completed sweeps and runs, for the results tables.

Every results artifact in the paper is assembled from the same raw material:
the per-screen ``final_metrics`` written by each sweep, plus the per-step
``acquired_batch`` lists. This module is the one place that knows how to find
and read them:

- :func:`_load_all_sweeps` / :func:`load_sweep_index` walk the local and
  shared sweep directories and build a lookup keyed by tag, ``(tag, acq)``,
  sweep id, and model name.
- :func:`resolve_row` turns one of those lookup keys into a sweep dict,
  preferring the most complete run when several match.
- :func:`per_screen_stats` gives (mean, sd) across screens for a metric.
- :func:`compute_diversity` / :func:`compute_diversity_from_batches` compute
  Vendi diversity and pathway overlap from the acquired batches.
- :func:`_load_jsonl_as_sweep` adapts fine-tuned-LLM prediction dumps
  (``ASSAYLOOP_LLM_PREDICTIONS``) into the same sweep shape.

The table generators built on this are
:mod:`assayloop.scripts.full_genome_table` (the paper's main results table,
run with ``--min-screen-freq 2``) and
:mod:`assayloop.scripts.export_recovery_curves`.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from assayloop import config  # noqa: E402  (after the sys.path insert above)

# Search order: your own results, then the optional shared directory, then
# the downloaded published bundle (ASSAYLOOP_RESULTS / ASSAYLOOP_SHARED_PATH
# / ASSAYLOOP_PUBLISHED). Published comes last so that a sweep you re-ran
# yourself wins over the shipped copy of it.
SWEEP_DIRS = [
    config.RESULTS_PATH / "sweeps",
    config.SHARED_PATH / "sweeps",
    config.PUBLISHED_PATH / "sweeps",
]
RUN_DIRS = [
    config.RESULTS_PATH / "runs",
    config.SHARED_PATH / "runs",
    config.PUBLISHED_PATH / "runs",
]

# Fine-tuned-LLM prediction dumps; set ASSAYLOOP_LLM_PREDICTIONS.
QWEN_PREDICTIONS = config.LLM_PREDICTIONS_PATH

# ---------------------------------------------------------------------------
# Sweep index
# ---------------------------------------------------------------------------

def _load_all_sweeps() -> list[tuple[str, str, dict]]:
    results = []
    for sweep_dir in SWEEP_DIRS:
        if not sweep_dir.is_dir():
            continue
        for sdir in sorted(sweep_dir.iterdir()):
            sp = sdir / "sweep.json"
            if not sp.is_file():
                continue
            try:
                s = json.loads(sp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            s["_source_dir"] = str(sweep_dir)
            results.append((str(sweep_dir), sdir.name, s))
    return results


def _find_run_result(run_id: str) -> Path | None:
    for rd in RUN_DIRS:
        fp = rd / run_id / "result.json"
        if fp.is_file():
            return fp
    return None


def load_sweep_index(all_sweeps: list) -> dict:
    by_tag: dict[str, list] = defaultdict(list)
    by_model: dict[str, list] = defaultdict(list)
    by_tag_acq: dict[tuple[str, str], list] = defaultdict(list)
    by_sweep_id: dict[str, list] = defaultdict(list)

    for src, sid, s in all_sweeps:
        cfg = s.get("config", {})
        tag = cfg.get("tag", "")
        model = cfg.get("model", "")
        acq = cfg.get("acq", "")
        entry = (src, sid, s)
        if tag:
            by_tag[tag].append(entry)
        by_model[model].append(entry)
        by_tag_acq[(tag, acq)].append(entry)
        by_sweep_id[sid].append(entry)

    return {
        "by_tag": by_tag,
        "by_model": by_model,
        "by_tag_acq": by_tag_acq,
        "by_sweep_id": by_sweep_id,
    }


def _pick_best(entries: list) -> tuple | None:
    if not entries:
        return None
    for src, sid, s in entries:
        ss = s.get("config", {}).get("screen_set", "")
        if ss == "public":
            return (src, sid, s)
    for src, sid, s in entries:
        ss = s.get("config", {}).get("screen_set", "")
        if "validation" not in ss:
            return (src, sid, s)
    return entries[0]


def _load_jsonl_as_sweep(path: Path, fmt: str) -> tuple | None:
    if not path.is_file():
        return None
    screens = []
    with open(path) as fh:
        for line in fh:
            screens.append(json.loads(line))
    if not screens:
        return None

    per_screen = []
    for s in screens:
        if fmt == "handoff":
            em = s.get("episode_metrics", {})
            fm = {
                "n_hits_vs_random": em.get("n_hits_vs_random"),
                "hits_auc_normalized": em.get("hits_auc_normalized"),
                "frac_hits": em.get("frac_hits"),
                "shortfall_frac": em.get("mean_shortfall", 0.0),
            }
        else:
            total_hits = s.get("total_hits", 0)
            lib_size = s.get("library_size", 1)
            rounds = s.get("per_round", [])
            cum_hits_final = rounds[-1]["cum_hits"] if rounds else 0
            cum_genes_final = rounds[-1]["cum_genes"] if rounds else 0
            rand_exp = cum_genes_final * total_hits / lib_size if lib_size > 0 else 0
            nauc = None
            if rounds and total_hits > 0 and lib_size > 0:
                xs = [0.0] + [r["cum_genes"] / lib_size for r in rounds]
                ys = [0.0] + [r["cum_hits"] / total_hits for r in rounds]
                auc = float(np.trapezoid(ys, xs))
                frac_budget = cum_genes_final / lib_size
                best_frac = min(total_hits, cum_genes_final) / lib_size
                auc_best = float(np.trapezoid(
                    [0.0, min(1.0, cum_genes_final / total_hits), 1.0]
                    if cum_genes_final >= total_hits
                    else [0.0, cum_genes_final / total_hits],
                    [0.0, best_frac, frac_budget]
                    if cum_genes_final >= total_hits
                    else [0.0, frac_budget],
                ))
                nauc = auc / auc_best if auc_best > 0 else None
            round_picks = s.get("round_picks", [])
            sf_fracs = []
            for j, r in enumerate(rounds):
                req = round_picks[j] if j < len(round_picks) else 100
                acquired = r.get("new_genes", req)
                sf_fracs.append(max(0, req - acquired) / req if req > 0 else 0)
            fm = {
                "n_hits_vs_random": cum_hits_final / rand_exp if rand_exp > 0 else None,
                "hits_auc_normalized": nauc,
                "frac_hits": cum_hits_final / total_hits if total_hits > 0 else None,
                "shortfall_frac": float(np.mean(sf_fracs)) if sf_fracs else None,
            }
        per_screen.append({
            "screen_name": s.get("dataset_name", ""),
            "run_id": "",
            "final_metrics": fm,
        })

    # Stash per-screen batches for diversity scoring (no run dirs to read from).
    jsonl_batches = []
    for s in screens:
        rg = s.get("round_genes", [])
        if rg and isinstance(rg[0], dict):
            batches = []
            for r in rg:
                # LLM format: submitted/new_hits/new_misses
                # Handoff format: acquired/hits/misses
                in_lib = (r.get("new_hits", []) + r.get("new_misses", [])
                          or r.get("hits", []) + r.get("misses", []))
                fallback = r.get("submitted", []) or r.get("acquired", [])
                batches.append(in_lib if in_lib else fallback)
            jsonl_batches.append(batches)
        else:
            jsonl_batches.append([])

    sd = {"per_screen": per_screen, "config": {"tag": str(path.stem), "screen_set": "public"},
          "_jsonl_batches": jsonl_batches}
    return (str(path.parent), str(path.name), sd)


def resolve_row(lookup_key, index: dict):
    if lookup_key is None:
        return None
    if isinstance(lookup_key, str):
        return _pick_best(index["by_tag"].get(lookup_key, []))
    if isinstance(lookup_key, tuple):
        kind = lookup_key[0]
        if kind == "model":
            return _pick_best(index["by_model"].get(lookup_key[1], []))
        if kind == "tag+acq":
            return _pick_best(index["by_tag_acq"].get((lookup_key[1], lookup_key[2]), []))
        if kind == "sweep_id":
            entries = index["by_sweep_id"].get(lookup_key[1], [])
            return entries[0] if entries else None
        if kind == "jsonl":
            return _load_jsonl_as_sweep(lookup_key[1], lookup_key[2])
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_screen_stats(sd: dict, key: str) -> tuple[float | None, float | None]:
    vals = []
    for ps in sd.get("per_screen", []):
        fm = ps.get("final_metrics") or {}
        v = fm.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def compute_diversity(run_ids: list[str], scorer) -> dict[str, float]:
    vendi_per_run, pathway_per_run = [], []
    for rid in run_ids:
        fp = _find_run_result(rid)
        if fp is None:
            continue
        try:
            r = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        v_steps, p_steps = [], []
        for step in r.get("steps", []):
            batch = step.get("acquired_batch")
            if not batch or len(batch) < 2:
                continue
            scores = scorer.score([], None, None, [], acquired_batch=batch)
            if "batch_vendi_ratio" in scores:
                v_steps.append(scores["batch_vendi_ratio"])
            if "batch_pathway_overlap_vs_random" in scores:
                p_steps.append(scores["batch_pathway_overlap_vs_random"])
        if v_steps:
            vendi_per_run.append(float(np.mean(v_steps)))
        if p_steps:
            pathway_per_run.append(float(np.mean(p_steps)))
    out = {}
    if vendi_per_run:
        out["vendi_mean"] = float(np.mean(vendi_per_run))
        out["vendi_std"] = float(np.std(vendi_per_run))
    if pathway_per_run:
        out["pathway_mean"] = float(np.mean(pathway_per_run))
        out["pathway_std"] = float(np.std(pathway_per_run))
    return out


def compute_diversity_from_batches(jsonl_batches: list[list[list[str]]], scorer) -> dict[str, float]:
    vendi_per_screen, pathway_per_screen = [], []
    for screen_batches in jsonl_batches:
        v_steps, p_steps = [], []
        for batch in screen_batches:
            if not batch or len(batch) < 2:
                continue
            scores = scorer.score([], None, None, [], acquired_batch=batch)
            if "batch_vendi_ratio" in scores:
                v_steps.append(scores["batch_vendi_ratio"])
            if "batch_pathway_overlap_vs_random" in scores:
                p_steps.append(scores["batch_pathway_overlap_vs_random"])
        if v_steps:
            vendi_per_screen.append(float(np.mean(v_steps)))
        if p_steps:
            pathway_per_screen.append(float(np.mean(p_steps)))
    out = {}
    if vendi_per_screen:
        out["vendi_mean"] = float(np.mean(vendi_per_screen))
        out["vendi_std"] = float(np.std(vendi_per_screen))
    if pathway_per_screen:
        out["pathway_mean"] = float(np.mean(pathway_per_screen))
        out["pathway_std"] = float(np.std(pathway_per_screen))
    return out


def get_run_ids(sd: dict) -> list[str]:
    return [ps["run_id"] for ps in sd.get("per_screen", []) if ps.get("run_id")]


