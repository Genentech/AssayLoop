"""Compute diversity metrics from stored acquired_batch data and compare across sweeps.

Reads result.json files for interesting sweeps, computes batch diversity
metrics per step using the BatchDiversity scorer, averages across steps
within each run, then compares across sweep conditions.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from assayloop.metrics.batch_diversity import BatchDiversity  # noqa: E402

SWEEPS_DIR = ROOT / "output" / "sweeps"
RUNS_DIR = ROOT / "output" / "runs"

INTERESTING = {
    # LLMs
    "claude-haiku-4.5-20251001": "LLM: Haiku",
    "claude-sonnet-4-6": "LLM: Sonnet",
    "claude-opus-4.8": "LLM: Opus",
    "qwen3.6-27b": "LLM: Qwen-27B",
    # Handoff (BPMF→transformer)
    "handoff-trainsel-n1": "Handoff n=1",
    "handoff-trainsel-n3": "Handoff n=3",
    # BPMF-init transformers (bpmf-objective)
    "gf-bpmfobj-bpmfinitK1-s5": "BPMFinit K=1",
    "gf-bpmfobj-bpmfinitK8-s5": "BPMFinit K=8",
    "gf-bpmfobj-bpmfinitK64-s5": "BPMFinit K=64",
    "gf-bpmfobj-bpmfinitK256-s5": "BPMFinit K=256",
    # Hits-objective BPMF-init
    "gf-hits-bpmfinitK1-s5": "HitsInit K=1",
    "gf-hits-bpmfinitK64-s5": "HitsInit K=64",
    "gf-hits-bpmfinitK256-s5": "HitsInit K=256",
    # Key baselines
    "gf-bpmf-train-hits": "BPMF-train-hits",
    "gf-bpmf-train-hits-rl-redo": "BPMF-train-hits-RL",
    "gf-mf-train-hits-shell-rl": "MF-shell-RL",
    "fast-sweep-1": "Fast baseline",
    "transformer-leaveout": "Transformer LOPO",
}

DIVERSITY_KEYS = [
    "batch_vendi",
    "batch_vendi_ratio",
    "batch_diversity",
    "batch_diversity_vs_random",
    "batch_pathway_diversity",
    "batch_pathway_coverage",
    "batch_pathway_overlap_vs_random",
]


def collect_sweep_runs() -> dict[str, list[str]]:
    """tag -> list of run_ids (taking first sweep per tag if duplicates)."""
    tag_to_runs: dict[str, list[str]] = {}
    for sdir in sorted(SWEEPS_DIR.iterdir()):
        sp = sdir / "sweep.json"
        if not sp.is_file():
            continue
        try:
            s = json.loads(sp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tag = s.get("config", {}).get("tag", "")
        if tag not in INTERESTING:
            continue
        if tag in tag_to_runs:
            continue
        run_ids = [ps["run_id"] for ps in s.get("per_screen", []) if ps.get("run_id")]
        tag_to_runs[tag] = run_ids
    return tag_to_runs


def compute_diversity_for_run(
    run_id: str, scorer: BatchDiversity
) -> dict[str, list[float]] | None:
    """Return {metric_key: [val_step0, val_step1, ...]} for one run."""
    fp = RUNS_DIR / run_id / "result.json"
    if not fp.is_file():
        return None
    try:
        r = json.loads(fp.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    steps = r.get("steps", [])
    result: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        batch = step.get("acquired_batch")
        if not batch or len(batch) < 2:
            continue
        scores = scorer.score([], None, None, [], acquired_batch=batch)
        for k in DIVERSITY_KEYS:
            if k in scores:
                result[k].append(scores[k])
    return dict(result) if result else None


def main():
    from scipy import stats as sp_stats

    print("Loading BatchDiversity scorer (PRESAGE + gene sets)...")
    scorer = BatchDiversity()
    # Force-load providers so timing isn't in the per-run loop
    scorer._ensure_provider()
    scorer._ensure_gene_sets()
    print("  done.\n")

    tag_to_runs = collect_sweep_runs()
    print(f"Found {len(tag_to_runs)} sweep tags, {sum(len(v) for v in tag_to_runs.values())} total runs.\n")

    # {tag: {metric: [mean_across_steps_for_run0, mean_for_run1, ...]}}
    tag_metrics: dict[str, dict[str, list[float]]] = {}

    for tag, run_ids in sorted(tag_to_runs.items()):
        label = INTERESTING[tag]
        print(f"  Computing: {label} ({len(run_ids)} runs)...", end="", flush=True)
        per_run: dict[str, list[float]] = defaultdict(list)
        n_ok = 0
        for rid in run_ids:
            rd = compute_diversity_for_run(rid, scorer)
            if rd is None:
                continue
            n_ok += 1
            for k, vals in rd.items():
                per_run[k].append(float(np.mean(vals)))
        tag_metrics[tag] = dict(per_run)
        print(f" {n_ok} ok")

    print("\n" + "=" * 100)
    print("SUMMARY: Mean diversity (averaged across steps, then across screens)")
    print("=" * 100)

    header = f"{'Condition':<25s}"
    for k in DIVERSITY_KEYS:
        short = k.replace("batch_", "").replace("_vs_random", "/rand")
        header += f"  {short:>14s}"
    print(header)
    print("-" * len(header))

    for tag in sorted(tag_metrics.keys(), key=lambda t: INTERESTING.get(t, t)):
        label = INTERESTING[tag]
        row = f"{label:<25s}"
        data = tag_metrics[tag]
        for k in DIVERSITY_KEYS:
            vals = data.get(k, [])
            if vals:
                row += f"  {np.mean(vals):>10.4f}±{np.std(vals):.3f}"
            else:
                row += f"  {'—':>14s}"
        print(row)

    # Statistical comparison: for each metric, do a one-way ANOVA across all
    # conditions, then report effect size (eta-squared).
    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON (one-way ANOVA across all conditions)")
    print("=" * 100)

    for k in DIVERSITY_KEYS:
        short = k.replace("batch_", "")
        groups = []
        labels = []
        for tag in sorted(tag_metrics.keys()):
            vals = tag_metrics[tag].get(k, [])
            if len(vals) >= 3:
                groups.append(vals)
                labels.append(INTERESTING[tag])
        if len(groups) < 2:
            print(f"\n{short}: insufficient data for ANOVA")
            continue
        F, p = sp_stats.f_oneway(*groups)
        # eta-squared
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        print(f"\n{short}:  F={F:.2f}  p={p:.2e}  eta²={eta_sq:.3f}")

        # Rank conditions by mean
        ranked = sorted(zip(labels, [np.mean(g) for g in groups]), key=lambda x: -x[1])
        print("  Ranked (high→low): ", end="")
        for lbl, m in ranked[:5]:
            print(f"{lbl}={m:.4f}  ", end="")
        print()

    # Pairwise: LLMs vs transformers
    print("\n" + "=" * 100)
    print("PAIRWISE: LLMs vs best transformer (Mann-Whitney U)")
    print("=" * 100)

    llm_tags = [t for t in tag_metrics if t.startswith("claude-") or t.startswith("qwen")]
    transformer_tags = [t for t in tag_metrics if t not in llm_tags]

    # Pool LLM runs vs pool transformer runs
    for k in DIVERSITY_KEYS:
        short = k.replace("batch_", "")
        llm_vals = []
        for t in llm_tags:
            llm_vals.extend(tag_metrics[t].get(k, []))
        # Pick best transformer condition by mean
        best_t = None
        best_mean = -1
        for t in transformer_tags:
            vals = tag_metrics[t].get(k, [])
            if vals and np.mean(vals) > best_mean:
                best_mean = np.mean(vals)
                best_t = t
        if not llm_vals or best_t is None:
            continue
        trans_vals = tag_metrics[best_t].get(k, [])
        if len(llm_vals) < 3 or len(trans_vals) < 3:
            continue
        U, p = sp_stats.mannwhitneyu(llm_vals, trans_vals, alternative="two-sided")
        d = (np.mean(llm_vals) - np.mean(trans_vals)) / np.sqrt(
            (np.var(llm_vals) + np.var(trans_vals)) / 2
        ) if (np.var(llm_vals) + np.var(trans_vals)) > 0 else 0
        print(
            f"  {short:30s}  LLM={np.mean(llm_vals):.4f}±{np.std(llm_vals):.3f}  "
            f"best_trans({INTERESTING[best_t]})={np.mean(trans_vals):.4f}±{np.std(trans_vals):.3f}  "
            f"Cohen's d={d:.3f}  p={p:.2e}"
        )

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    # Find metric with highest eta-squared
    best_metric = None
    best_eta = -1
    for k in DIVERSITY_KEYS:
        groups = []
        for tag in sorted(tag_metrics.keys()):
            vals = tag_metrics[tag].get(k, [])
            if len(vals) >= 3:
                groups.append(vals)
        if len(groups) < 2:
            continue
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_vals - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        if eta_sq > best_eta:
            best_eta = eta_sq
            best_metric = k
    if best_metric:
        print(
            f"Best discriminating metric: {best_metric}\n"
            f"  eta² = {best_eta:.3f} (fraction of variance explained by condition)\n"
            f"  Higher eta² = more of the diversity variation is explained by which\n"
            f"  method was used, rather than by screen-to-screen noise."
        )


if __name__ == "__main__":
    main()
