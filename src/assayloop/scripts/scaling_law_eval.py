"""Evaluate scaling-law (``scl2-*``) checkpoints on the f2 gene universe.

For each ranker checkpoint we run the greedy AL loop with the candidate pool set
to the **f2 universe** (genes appearing in >=2 ``public`` screens), exactly as the
paper table does, and report:

  - ``nvr_at_10`` - the domain-adjusted NVR at a 1000-gene budget (batch 100 x 10
    steps), computed with the shared ``adjusted_nvr_value`` helper.  This is the
    number the scaling curves plot; it is identical to the paper table's metric.
  - ``mean_recall_curve`` / ``budget_to_50_recall`` - cumulative in-library recall
    per step and the fraction of the **f2 universe** screened to reach 50% recall.
    Budget is normalised by the candidate pool the policy actually draws from
    (|U| = 21,147), not by the per-screen library size D, so the axis is bounded
    by 1.0 instead of topping out at |U|/D ~ 1.15.

Because greedy is prefix-deterministic, the first 10 steps of a longer run equal a
10-step run, so we run ``--n-steps`` (default 50) once and read NVR@10 off the
first 10 steps while getting the recall curve for free.

Results are written to ``{run_dir}/scaling2_eval.json``.

Usage::

    uv run python -m assayloop.scripts.scaling_law_eval                 # 1 GPU
    uv run python -m assayloop.scripts.scaling_law_eval --parallel 8    # 8 GPUs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from assayloop import config
from assayloop.metrics.hits_auc import adjusted_nvr_value

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scaling_eval")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
EVAL_FILE = "scaling2_eval.json"
TRAJ_FILE = "scaling2_traj.json"   # cached per-screen rollout trajectories
NVR_BUDGET = 1000  # batch 100 x 10 steps
# AL rounds at which adjusted NVR is reported (budget = k * batch_size genes).
# Enables the NVR@k-vs-scale heatmaps; 10 is the headline point.
DEFAULT_K_LIST = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
# Recall thresholds (%) for "budget to reach k% recall" (fraction of the f2
# universe screened). Enables the budget-to-recall-vs-scale plots.
DEFAULT_RECALL_THRESHOLDS = [50, 60, 70, 80, 90, 95, 100]


# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------
def _parse_run(name: str) -> dict | None:
    """scl2-data-d{N}[-rl]-s{seed} | scl2-model-{tier}[-rl]-s{seed}."""
    if not name.startswith("scl2-"):
        return None
    rest = name[len("scl2-"):]
    is_rl = "-rl-" in rest
    rest = rest.replace("-rl-", "-")
    parts = rest.split("-")
    try:
        sweep = parts[0]
        seed = int(parts[-1][1:]) if parts[-1].startswith("s") else 0
        if sweep == "data":
            train_size = int(parts[1][1:])  # d{N}
            tier = "L"
        elif sweep == "model":
            tier = parts[1]
            train_size = 1349
        else:
            return None
    except (ValueError, IndexError):
        return None
    return {"sweep": sweep, "tier": tier, "train_size": train_size,
            "seed": seed, "is_rl": is_rl}


# ---------------------------------------------------------------------------
# f2 universe (built identically to the table / RL trainer)
# ---------------------------------------------------------------------------
def _build_f2_universe(screens) -> list[str]:
    from collections import Counter
    freq = Counter()
    for s in screens:
        for g in set(s.genes):
            freq[g] += 1
    return sorted(g for g in freq if freq[g] >= 2)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
#
# The expensive GPU rollout and the cheap metric derivation are separated: each
# checkpoint's per-screen cumulative trajectory (in-lib picks, forgiven picks,
# hits per AL round) is cached to TRAJ_FILE. All metrics (NVR@k, budget→recall
# at any threshold, recall curve) are then derived on CPU from that cache, so
# changing k_list / thresholds / n_steps never re-runs the model.
# ---------------------------------------------------------------------------
def _rollout_trajectories(ckpt_dir: Path, screens, universe, *,
                          n_steps: int, batch_size: int, device: str) -> list:
    """Run the greedy AL rollout and return per-screen cumulative trajectories."""
    from assayloop.experiment.runner import RunConfig, run_one_screen
    from assayloop.models.amortized_ranker import AmortizedRankerModel

    inf = AmortizedRankerModel(checkpoint=str(ckpt_dir), ckpt_file="model.pt",
                               device=device)
    cfg = RunConfig(
        screen_set="public", model="amortized_ranker", acq="greedy",
        batch_size=batch_size, n_steps=n_steps, persist=False,
        metrics=["hits_auc"], parallel=1, max_shortfall_frac=1.0,
        universe_genes=universe,
    )
    universe_set = set(universe)
    trajs = []
    for s in screens:
        try:
            res = run_one_screen(s, cfg, model_obj=inf, verbose=False)
        except Exception as e:  # noqa: BLE001
            log.warning("  eval failed on %s: %s", s.dataset_name, e)
            continue
        screen_genes = set(s.genes)
        hitset = {g for g, h in zip(s.genes, s.hits) if h}
        H, D = len(hitset), len(screen_genes)
        if H == 0 or D == 0:
            continue
        cum_n1 = cum_n2 = cum_hits = 0
        n1a, n2a, ha = [], [], []
        for rec in res.history:
            for g in rec.acquired_batch:
                if g in screen_genes:
                    cum_n1 += 1
                    if g in hitset:
                        cum_hits += 1
                elif g in universe_set:
                    cum_n2 += 1
            n1a.append(cum_n1)
            n2a.append(cum_n2)
            ha.append(cum_hits)
        if ha:
            # H_reach = hits on genes actually in the candidate pool (f2 universe);
            # recall is normalized by this so 100% is attainable at exhaustion.
            trajs.append({"name": s.dataset_name, "D": D, "H": H,
                          "H_reach": len(hitset & universe_set),
                          "n1": n1a, "n2": n2a, "hits": ha})
    del inf
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return trajs


def _metrics_from_traj(trajs, *, k_list, thresholds, batch_size,
                       universe_size: int) -> dict:
    """Derive NVR@k, budget→recall and the recall curve from cached trajectories.
    Pure CPU; cheap to recompute for new k / thresholds."""
    if not trajs or universe_size <= 0:
        return {}
    per_k = {k: [] for k in k_list}
    budget_by_t = {t: [] for t in thresholds}
    recall_curves = []
    for tr in trajs:
        D, H = tr["D"], tr["H"]
        H_reach = tr.get("H_reach", H)   # hits reachable in the candidate pool
        n1, n2, hits = tr["n1"], tr["n2"], tr["hits"]
        nsteps = len(hits)
        if not nsteps or H <= 0 or D <= 0:
            continue
        # NVR keeps the true library hit rate (all in-lib hits) as the baseline,
        # matching the paper table.
        for k in k_list:
            if k > nsteps:
                continue          # cached rollout doesn't reach this budget
            idx = k - 1
            per_k[k].append(adjusted_nvr_value(hits[idx], n1[idx], n2[idx],
                                               D, H, k * batch_size))
        # Recall is over *reachable* hits (in the f2 universe), so exhausting the
        # universe gives 100% recall for every screen -> budget→recall is fully
        # covered, no structural censoring.
        if H_reach <= 0:
            continue
        recall = [h / H_reach for h in hits]
        recall_curves.append(recall)
        for t in thresholds:
            frac = t / 100.0
            for i, r in enumerate(recall):
                if r >= frac:
                    # Numerator = genes actually picked (n1 in-library + n2
                    # out-of-library-but-in-universe); the nominal
                    # (i+1)*batch_size overshoots on the final, partially
                    # filled batch. Denominator = the f2 universe, i.e. the
                    # pool those picks are drawn from, so the ratio is <= 1.
                    budget_by_t[t].append((n1[i] + n2[i]) / universe_size)
                    break

    if not recall_curves:
        return {}
    max_steps = max(len(c) for c in recall_curves)
    mean_recall = [float(np.mean([c[i] for c in recall_curves if i < len(c)]))
                   for i in range(max_steps)]
    nvr_at_k = {str(k): float(np.mean(v)) for k, v in per_k.items() if v}
    n_scored = len(recall_curves)
    out = {
        "n_screens": n_scored,
        "nvr_at_k": nvr_at_k,
        "nvr_at_10": nvr_at_k.get("10", nvr_at_k.get(
            str(max(int(k) for k in nvr_at_k)) if nvr_at_k else "0", 0.0)),
        "mean_recall_curve": mean_recall,
        "n_steps_eval": max_steps,
        "batch_size": batch_size,
    }
    if per_k.get(10):
        v = per_k[10]
        out["nvr_at_10_sem"] = float(np.std(v) / max(np.sqrt(len(v)), 1))
    out["universe_size"] = int(universe_size)
    # budget to reach k% recall, as a fraction of the f2 universe
    # (mean over screens that reached it) + coverage.
    # Recall is over reachable hits, so with --exhaust every screen reaches every
    # threshold -> coverage == 1.0. Coverage < 1.0 means the rollout was not run
    # to exhaustion (budget-censored); re-run with --exhaust.
    budget_to_recall = {str(t): float(np.mean(v)) for t, v in budget_by_t.items() if v}
    if budget_to_recall:
        out["budget_to_recall"] = budget_to_recall
        out["budget_to_recall_reached"] = {
            str(t): len(v) / n_scored for t, v in budget_by_t.items() if v}
    if budget_by_t.get(50):
        out["budget_to_50_recall"] = float(np.mean(budget_by_t[50]))
        out["budget_to_50_recall_n"] = len(budget_by_t[50])
        out["frac_screens_reached_50"] = len(budget_by_t[50]) / n_scored
    return out


def _eval_checkpoint(ckpt_dir: Path, screens, universe, *,
                     n_steps: int, batch_size: int, device: str,
                     k_list=None, force_rollout: bool = False) -> dict:
    """Load-or-compute the trajectory cache, then derive metrics from it."""
    k_list = sorted(set(k_list or DEFAULT_K_LIST))
    thresholds = DEFAULT_RECALL_THRESHOLDS
    traj_path = ckpt_dir / TRAJ_FILE

    trajs = None
    universe_size = len(universe)
    if not force_rollout and traj_path.exists():
        try:
            cached = json.loads(traj_path.read_text())
            universe_size = int(cached.get("universe_size") or universe_size)
            # Cache is usable if its rollout is at least as long as requested
            # (a longer cache covers every shorter budget / threshold).
            if cached.get("n_steps", 0) >= n_steps and cached.get("screens"):
                trajs = cached["screens"]
                log.info("  [%s] using cached trajectory (%d steps)",
                         ckpt_dir.name, cached["n_steps"])
        except Exception as e:  # noqa: BLE001
            log.warning("  [%s] bad trajectory cache (%s); re-rolling", ckpt_dir.name, e)

    if trajs is None:
        universe_size = len(universe)
        trajs = _rollout_trajectories(ckpt_dir, screens, universe,
                                      n_steps=n_steps, batch_size=batch_size,
                                      device=device)
        if trajs:
            traj_path.write_text(json.dumps({
                "n_steps": n_steps, "batch_size": batch_size,
                "universe_size": universe_size, "screens": trajs}))

    return _metrics_from_traj(trajs, k_list=[k for k in k_list],
                              thresholds=thresholds, batch_size=batch_size,
                              universe_size=universe_size)


def _eval_one(run_dir: Path, meta: dict, screens, universe, *,
              n_steps: int, batch_size: int, device: str, k_list=None,
              force_rollout: bool = False) -> None:
    try:
        metrics = _eval_checkpoint(run_dir, screens, universe,
                                   n_steps=n_steps, batch_size=batch_size,
                                   device=device, k_list=k_list,
                                   force_rollout=force_rollout)
    except Exception as e:  # noqa: BLE001
        log.error("  FAILED %s: %s", run_dir.name, e)
        return
    if not metrics:
        log.warning("  no screens evaluated for %s", run_dir.name)
        return
    metrics.update(run_name=run_dir.name, eval_set="public", **meta)
    (run_dir / EVAL_FILE).write_text(json.dumps(metrics, indent=2))
    log.info("  [%s] nvr@10=%.3f  budget50=%s (n=%d)", run_dir.name,
             metrics["nvr_at_10"],
             f"{metrics['budget_to_50_recall']:.1%}"
             if "budget_to_50_recall" in metrics else "N/A",
             metrics["n_screens"])


def _effective_steps(universe, batch_size, n_steps, exhaust):
    """Rounds needed to exhaust the candidate universe, if --exhaust; else n_steps."""
    if exhaust:
        return int(np.ceil(len(universe) / max(batch_size, 1)))
    return n_steps


def _gpu_worker(gpu_id: int, work: list, eval_set: str,
                n_steps: int, batch_size: int, k_list=None,
                exhaust: bool = False, force_rollout: bool = False) -> None:
    from assayloop.tasks import load_screens
    device = f"cuda:{gpu_id}"
    screens = load_screens(target_set=eval_set)
    universe = _build_f2_universe(screens)
    eff_steps = _effective_steps(universe, batch_size, n_steps, exhaust)
    log.info("GPU %d: %d checkpoints, %d screens, f2 universe %d genes, %d steps",
             gpu_id, len(work), len(screens), len(universe), eff_steps)
    for i, (run_dir, meta) in enumerate(work, 1):
        log.info("GPU %d [%d/%d] %s", gpu_id, i, len(work), run_dir.name)
        _eval_one(run_dir, meta, screens, universe,
                  n_steps=eff_steps, batch_size=batch_size, device=device,
                  k_list=k_list, force_rollout=force_rollout)


def main() -> None:
    from assayloop.tasks import load_screens

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankers-dir", default=str(RANKERS_DIR))
    ap.add_argument("--prefix", default="scl2-")
    ap.add_argument("--eval-set", default="public")
    ap.add_argument("--n-steps", type=int, default=100,
                    help="AL steps per screen (NVR@10 uses the first 10; the rest "
                         "extend the recall curve for budget-to-50%%). 100 steps "
                         "(10k genes) lets nearly all seeds reach 50% recall, "
                         "avoiding survivorship bias in the budget-to-50 average.")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=1, help="GPUs to use.")
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-list", default=None,
                    help="Comma-sep AL rounds k for NVR@k (default %s). Enables "
                         "the NVR@k-vs-scale heatmaps." % ",".join(map(str, DEFAULT_K_LIST)))
    ap.add_argument("--exhaust", action="store_true",
                    help="Roll out until the f2 universe is exhausted "
                         "(n_steps = ceil(|universe|/batch)) so budget→recall is "
                         "not budget-censored. Cached, so it's a one-time cost.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute metrics even if scaling2_eval.json exists "
                         "(reuses the trajectory cache — no GPU rollout).")
    ap.add_argument("--force-rollout", action="store_true",
                    help="Also ignore the trajectory cache and re-run the GPU "
                         "rollout (implies --force).")
    args = ap.parse_args()
    if args.force_rollout:
        args.force = True
    k_list = ([int(x) for x in args.k_list.split(",")] if args.k_list
              else DEFAULT_K_LIST)
    k_list = ([int(x) for x in args.k_list.split(",")] if args.k_list
              else DEFAULT_K_LIST)

    rankers = Path(args.rankers_dir)
    run_dirs = []
    for d in sorted(rankers.iterdir()):
        if not d.name.startswith(args.prefix):
            continue
        if not (d / "summary.json").exists() or not (d / "model.pt").exists():
            continue
        if not args.force and (d / EVAL_FILE).exists():
            log.info("SKIP (evaluated): %s", d.name)
            continue
        meta = _parse_run(d.name)
        if meta is None:
            continue
        run_dirs.append((d, meta))

    run_dirs = [x for i, x in enumerate(run_dirs) if i % args.num_nodes == args.node]
    if not run_dirs:
        log.info("Nothing to evaluate on this node.")
        return
    log.info("Evaluating %d checkpoints (node %d/%d, parallel=%d)",
             len(run_dirs), args.node, args.num_nodes, args.parallel)

    if args.parallel <= 1:
        screens = load_screens(target_set=args.eval_set)
        universe = _build_f2_universe(screens)
        eff_steps = _effective_steps(universe, args.batch_size, args.n_steps,
                                     args.exhaust)
        log.info("Loaded %d screens, f2 universe %d genes, %d steps",
                 len(screens), len(universe), eff_steps)
        for i, (d, meta) in enumerate(run_dirs, 1):
            log.info("[%d/%d] %s (%s tier=%s d=%d rl=%s)", i, len(run_dirs),
                     d.name, meta["sweep"], meta["tier"], meta["train_size"],
                     meta["is_rl"])
            _eval_one(d, meta, screens, universe, n_steps=eff_steps,
                      batch_size=args.batch_size, device=args.device,
                      k_list=k_list, force_rollout=args.force_rollout)
    else:
        import torch
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        n_gpus = min(args.parallel, torch.cuda.device_count())
        chunks: list[list] = [[] for _ in range(n_gpus)]
        for i, item in enumerate(run_dirs):
            chunks[i % n_gpus].append(item)
        procs = []
        for g in range(n_gpus):
            if not chunks[g]:
                continue
            p = mp.Process(target=_gpu_worker,
                           args=(g, chunks[g], args.eval_set,
                                 args.n_steps, args.batch_size, k_list,
                                 args.exhaust, args.force_rollout))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    log.info("Done. Wrote %s per run.", EVAL_FILE)


if __name__ == "__main__":
    main()
