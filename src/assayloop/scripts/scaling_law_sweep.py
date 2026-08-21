"""Scaling-law sweep v2: two clean 1-D sweeps (data amount and model size).

Instead of a (model x data) grid, we run two independent 1-D sweeps that share a
single anchor point (the production model on all data):

  * **Data sweep** - the production model size (tier ``L``: d_model=384, K=10,
    2 heads, 3 layers), varying the number of training screens.  The BPMF gene
    embeddings used to initialize each run are (re)fit on the *same* screen
    subset the transformer trains on (see ``--phase bpmf``), fixing the leakage
    bug where BPMF saw all screens.
  * **Model sweep** - all training data (1349 screens), varying the transformer
    size across five tiers.  Each tier uses the *scaled K* BPMF embeddings
    (BPMF K = the tier's ``d_gene``), fit once on the full data (already on disk).

All runs are trained WITHOUT screen descriptions (``--no-description``) to isolate
the transformer/embedding scaling.  RL fine-tuning uses the f2 gene universe
(``--full-genome``) and selects on the domain-adjusted NVR.

Phases (run in order, each resume-safe):

    # 0. per-subset BPMF for the data sweep (K=10)
    uv run python -m assayloop.scripts.scaling_law_sweep --phase bpmf
    # 1. supervised pretraining (SFT), ~3 lanes/GPU
    uv run python -m assayloop.scripts.scaling_law_sweep --phase sup --lanes 3
    # 2. RL fine-tuning (4 GPUs each, 2 concurrent on an 8-GPU node)
    uv run python -m assayloop.scripts.scaling_law_sweep --phase rl
    # or chain all three:
    uv run python -m assayloop.scripts.scaling_law_sweep --phase all

GPU packing (single 8-GPU node): BPMF = 1 GPU each (8 concurrent); SFT ~= 1/4 GPU
(``--lanes`` per GPU, default 3); RL = 4 GPUs each (2 concurrent).  Each job is
pinned via ``CUDA_VISIBLE_DEVICES`` in its subprocess env.

Resume: jobs whose ``summary.json`` (rankers) / ``bpmf_result.pkl`` already exist
are skipped.  Jobs are enumerated **seed-outer** (every point gets seed 0 before
any point gets seed 1), so an interrupted run still yields full curves.
Ctrl-C stops launching new jobs; running subprocesses finish.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scaling_sweep")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _sigint_handler(signum, frame):
    if _shutdown.is_set():
        log.warning("Second interrupt - forcing exit")
        sys.exit(1)
    log.warning("Interrupt - finishing running job(s), launching no more. "
                "Ctrl-C again to force-quit.")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model tiers: (d_model, d_gene=K, d_hit, nhead, num_layers, dim_feedforward)
# L = production (d_model 384, K=10, 2 heads, 3 layers).  K scales ~with d_model.
MODEL_TIERS: dict[str, tuple[int, int, int, int, int, int]] = {
    "XS": (96,   3, 32, 1, 2,  256),
    "S":  (192,  5, 48, 2, 2,  512),
    "M":  (288,  8, 64, 2, 3,  768),
    "L":  (384, 10, 64, 2, 3, 1024),   # production / data-sweep anchor
    "XL": (512, 16, 64, 4, 4, 1536),
}
TIER_ORDER = ["XS", "S", "M", "L", "XL"]
PROD_TIER = "L"

# Full public_train screen count (== _subsample returns the whole pool at this N).
FULL_DATA_SIZE = 1349

# Data sweep: training-screen counts (denser at the low end for a clean log axis).
TRAIN_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1349]

# SFT seeds per data-sweep size (more seeds where variance is highest).
# Generous counts: the full sweep fit in ~7h of a 72h budget, so there is ~10x
# headroom; these ~4x the original schedule to clean up the curves.  Runs are
# resume-safe and enumerated seed-outer, so re-running just fills in new seeds.
DATA_SUP_SEEDS: dict[int, int] = {
    1: 32, 2: 32, 5: 24, 10: 24, 20: 20, 50: 20,
    100: 16, 200: 16, 400: 12, 800: 10, 1349: 10,
}
# SFT seeds per model-sweep tier.  Already tight at 20 (SEM ~0.04-0.10); 30
# resolves the small M/L non-monotonicity.
MODEL_SUP_SEEDS = 30
# RL fine-tunes the top-K SFT seeds (by validation NVR) at each point.  Bumped
# 6 -> 12: RL is the noisy part of the model-size curve (few seeds, one bad-init
# outlier per tier), so extra RL seeds help most here.
RL_TOP_K = 12

# Supervised epochs (small data needs more passes).
SUP_EPOCHS: dict[int, int] = {1: 100, 2: 100, 5: 100, 10: 100}
SUP_EPOCHS_DEFAULT = 40

BPMF_K = 10  # data sweep uses the production K for its per-subset BPMF fits.

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
BPMF_DIR = config.OUTPUT_PATH / "bpmf"

WANDB_GROUP = "scaling-law-v2"

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
SUP_FIXED = [
    "--objective", "hits",
    "--no-description",
    "--batch-size", "32",
    "--lr", "3e-4",
    "--weight-decay", "0.01",
    "--hit-context-frac", "0.25",
    "--max-context", "1024",
    "--max-targets", "4096",
    "--dagger-rounds", "0",
    "--dropout", "0",
    "--num-workers", "6",
    "--device", "cuda",
    "--val-screen-set", "public_validation",
    "--val-al-screens", "20",
    "--eval-every", "1",
    "--eval-screen-set", "public",
    "--wandb-group", WANDB_GROUP,
]

RL_FIXED = [
    "--reward-mode", "context_delta",
    "--kl-coef", "0.0",
    "--epochs", "50",
    "--group-size", "8",
    "--n-steps", "10",
    "--batch-size", "100",
    "--lr", "1e-5",
    "--gpus", "4",
    "--full-genome",
    "--init-ckpt-file", "model.pt",
    "--eval-screen-set", "public_validation",
    "--test-eval-set", "public",
    "--save-every", "5",
    "--wandb-group", WANDB_GROUP,
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def _data_sup_name(size: int, seed: int) -> str:
    return f"scl2-data-d{size}-s{seed}"


def _data_rl_name(size: int, seed: int) -> str:
    return f"scl2-data-d{size}-rl-s{seed}"


def _model_sup_name(tier: str, seed: int) -> str:
    return f"scl2-model-{tier}-s{seed}"


def _model_rl_name(tier: str, seed: int) -> str:
    return f"scl2-model-{tier}-rl-s{seed}"


# ---------------------------------------------------------------------------
# BPMF lookup
# ---------------------------------------------------------------------------
def _find_bpmf(K: int, train_size: int | None = None,
               subset_seed: int | None = None) -> str | None:
    """Locate a BPMF result pkl.

    ``train_size < FULL_DATA_SIZE`` -> the subset fit tagged ``_d{N}_s{seed}_``.
    Otherwise -> the full-data fit (whose tag has no ``_d..._s...`` segment).
    Returns the newest match, or ``None`` if absent.
    """
    if train_size is not None and train_size < FULL_DATA_SIZE:
        pat = f"bpmf_public_train_K{K}_su1_sv1_d{train_size}_s{subset_seed}_*"
        m = sorted(BPMF_DIR.glob(f"{pat}/bpmf_result.pkl"))
        return str(m[-1]) if m else None
    full_re = re.compile(rf"^bpmf_public_train_K{K}_su1_sv1_\d{{8}}_\d{{6}}$")
    cands = sorted(p for p in BPMF_DIR.glob(f"bpmf_public_train_K{K}_su1_sv1_*")
                   if full_re.match(p.name) and (p / "bpmf_result.pkl").exists())
    return str(cands[-1] / "bpmf_result.pkl") if cands else None


def _bpmf_done(train_size: int, subset_seed: int) -> bool:
    return _find_bpmf(BPMF_K, train_size, subset_seed) is not None


def _tier_flags(tier: str) -> list[str]:
    d_model, d_gene, d_hit, nhead, num_layers, dim_ff = MODEL_TIERS[tier]
    return [
        "--d-model", str(d_model), "--d-gene", str(d_gene), "--d-hit", str(d_hit),
        "--nhead", str(nhead), "--num-layers", str(num_layers),
        "--dim-feedforward", str(dim_ff),
    ]


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------
def _build_bpmf_cmd(train_size: int, subset_seed: int) -> list[str]:
    return [
        sys.executable, "-m", "assayloop.scripts.train_bpmf_gpu",
        "--target-set", "public_train",
        "--K", str(BPMF_K), "--sigma-u", "1", "--sigma-v", "1",
        "--train-size", str(train_size), "--subset-seed", str(subset_seed),
        "--output-root", str(BPMF_DIR), "--device", "cuda",
    ]


def _build_sup_cmd(run_name: str, tier: str, train_size: int, seed: int,
                   bpmf_pkl: str, sweep: str) -> list[str]:
    epochs = SUP_EPOCHS.get(train_size, SUP_EPOCHS_DEFAULT)
    tags = (f"scaling_law_v2,{sweep},tier_{tier},data_{train_size},seed_{seed}")
    cmd = [
        "assayloop", "train-ranker",
        "--run-name", run_name,
        "--train-size", str(train_size),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--wandb-tags", tags,
    ]
    cmd += _tier_flags(tier)
    cmd += SUP_FIXED
    cmd += ["--init-gene-factors", f"bpmf:{bpmf_pkl}", "--no-freeze-gene-factors"]
    return cmd


def _build_rl_cmd(run_name: str, init_dir: str, train_size: int, seed: int,
                  tier: str, sweep: str) -> list[str]:
    tags = f"scaling_law_v2,rl,{sweep},tier_{tier},data_{train_size},seed_{seed}"
    cmd = [
        "assayloop", "train-ranker-rl",
        "--run-name", run_name,
        "--init-checkpoint", init_dir,
        "--train-size", str(train_size),
        "--seed", str(seed),
        "--wandb-tags", tags,
    ]
    cmd += RL_FIXED
    return cmd


# ---------------------------------------------------------------------------
# Resume / seed selection
# ---------------------------------------------------------------------------
def _is_done(run_name: str, rankers_dir: Path) -> bool:
    return (rankers_dir / run_name / "summary.json").exists()


def _val_nvr(run_name: str, rankers_dir: Path) -> float | None:
    p = rankers_dir / run_name / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s.get("best_val_n_hits_vs_random")


def _top_sup_seeds(candidates: list[tuple[int, str]], rankers_dir: Path,
                   k: int) -> list[tuple[int, str]]:
    """From (seed, sup_run_name) candidates, return the top-k by validation NVR."""
    scored = []
    for seed, name in candidates:
        nvr = _val_nvr(name, rankers_dir)
        if nvr is not None:
            scored.append((nvr, seed, str(rankers_dir / name)))
    scored.sort(reverse=True)
    return [(seed, d) for _, seed, d in scored[:k]]


# ---------------------------------------------------------------------------
# Job construction (seed-outer for breadth-first coverage)
# ---------------------------------------------------------------------------
def _bpmf_jobs(sizes, rankers_dir) -> list[tuple[list[str], str]]:
    jobs = []
    max_seed = max(DATA_SUP_SEEDS.values())
    for seed in range(max_seed):
        for N in sizes:
            if N >= FULL_DATA_SIZE:
                continue  # full-data K=10 fit already exists on disk
            if seed >= DATA_SUP_SEEDS[N]:
                continue
            if _bpmf_done(N, seed):
                continue
            name = f"bpmf-K{BPMF_K}-d{N}-s{seed}"
            jobs.append((_build_bpmf_cmd(N, seed), name))
    return jobs


def _sup_jobs(tiers, sizes, rankers_dir) -> list[tuple[list[str], str]]:
    jobs = []
    max_seed = max(max(DATA_SUP_SEEDS.values()), MODEL_SUP_SEEDS)
    for seed in range(max_seed):
        # Data sweep (production tier L, varying data).
        for N in sizes:
            if seed >= DATA_SUP_SEEDS[N]:
                continue
            name = _data_sup_name(N, seed)
            if _is_done(name, rankers_dir):
                continue
            bpmf = _find_bpmf(BPMF_K, N, seed)
            if bpmf is None:
                log.warning("No BPMF for data d=%d s=%d (run --phase bpmf first) "
                            "- skipping %s", N, seed, name)
                continue
            jobs.append((_build_sup_cmd(name, PROD_TIER, N, seed, bpmf, "data"),
                         name))
        # Model sweep (all data, varying tier / scaled K).
        if seed < MODEL_SUP_SEEDS:
            for tier in tiers:
                name = _model_sup_name(tier, seed)
                if _is_done(name, rankers_dir):
                    continue
                K = MODEL_TIERS[tier][1]
                bpmf = _find_bpmf(K, None, None)
                if bpmf is None:
                    log.warning("No full-data BPMF K=%d for tier %s - skipping %s",
                                K, tier, name)
                    continue
                jobs.append((_build_sup_cmd(name, tier, FULL_DATA_SIZE, seed, bpmf,
                                            "model"), name))
    return jobs


def _rl_jobs(tiers, sizes, rankers_dir) -> list[tuple[list[str], str]]:
    jobs = []
    # Determine the top SFT seeds per point first (independent of RL seed order),
    # then emit RL jobs seed-outer by RL rank.
    data_best = {N: _top_sup_seeds([(s, _data_sup_name(N, s))
                                    for s in range(DATA_SUP_SEEDS[N])],
                                   rankers_dir, RL_TOP_K)
                 for N in sizes}
    model_best = {t: _top_sup_seeds([(s, _model_sup_name(t, s))
                                     for s in range(MODEL_SUP_SEEDS)],
                                    rankers_dir, RL_TOP_K)
                  for t in tiers}
    for rank in range(RL_TOP_K):
        for N in sizes:
            best = data_best[N]
            if rank >= len(best):
                if rank == 0:
                    log.warning("No completed SFT for data d=%d - skipping RL", N)
                continue
            seed, init_dir = best[rank]
            name = _data_rl_name(N, seed)
            if _is_done(name, rankers_dir):
                continue
            jobs.append((_build_rl_cmd(name, init_dir, N, seed, PROD_TIER, "data"),
                         name))
        for tier in tiers:
            best = model_best[tier]
            if rank >= len(best):
                if rank == 0:
                    log.warning("No completed SFT for tier %s - skipping RL", tier)
                continue
            seed, init_dir = best[rank]
            name = _model_rl_name(tier, seed)
            if _is_done(name, rankers_dir):
                continue
            jobs.append((_build_rl_cmd(name, init_dir, FULL_DATA_SIZE, seed, tier,
                                       "model"), name))
    return jobs


# ---------------------------------------------------------------------------
# GPU slot tables
# ---------------------------------------------------------------------------
def _slots(phase: str, n_gpus: int, lanes: int) -> list[dict[str, str]]:
    if phase == "bpmf":
        return [{"CUDA_VISIBLE_DEVICES": str(g)} for g in range(n_gpus)]
    if phase == "sup":
        return [{"CUDA_VISIBLE_DEVICES": str(g),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                for g in range(n_gpus) for _ in range(lanes)]
    if phase == "rl":
        groups = []
        for g0 in range(0, n_gpus - 3, 4):
            groups.append({"CUDA_VISIBLE_DEVICES":
                           ",".join(str(g0 + i) for i in range(4))})
        return groups or [{"CUDA_VISIBLE_DEVICES": ",".join(str(i)
                          for i in range(min(4, n_gpus)))}]
    raise ValueError(phase)


# ---------------------------------------------------------------------------
# Slot-queue scheduler
# ---------------------------------------------------------------------------
def _run_jobs(jobs, slots, rankers_dir, dry_run: bool) -> list[str]:
    total = len(jobs)
    if dry_run:
        for i, (cmd, name) in enumerate(jobs, 1):
            slot = slots[(i - 1) % len(slots)]
            print(f"[{i}/{total}] {name}  "
                  f"(CUDA_VISIBLE_DEVICES={slot.get('CUDA_VISIBLE_DEVICES')})")
            print("  " + " \\\n    ".join(cmd) + "\n")
        return []

    free = list(range(len(slots)))
    running: dict[subprocess.Popen, tuple[int, str]] = {}
    queue = list(jobs)
    failed: list[str] = []
    launched = 0

    while queue or running:
        while queue and free and not _shutdown.is_set():
            cmd, name = queue.pop(0)
            if _is_done(name, rankers_dir):
                log.info("SKIP (done at runtime): %s", name)
                continue
            si = free.pop()
            env = {**os.environ, **slots[si]}
            launched += 1
            log.info("[%d/%d slot%d cvd=%s] %s", launched, total, si,
                     slots[si].get("CUDA_VISIBLE_DEVICES"), name)
            running[subprocess.Popen(cmd, env=env)] = (si, name)
        finished = [p for p in running if p.poll() is not None]
        for p in finished:
            si, name = running.pop(p)
            free.append(si)
            if p.returncode != 0:
                log.error("FAILED (exit %d): %s", p.returncode, name)
                failed.append(name)
        if not finished:
            time.sleep(2)
        if _shutdown.is_set() and not running:
            break
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    signal.signal(signal.SIGINT, _sigint_handler)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["bpmf", "sup", "rl", "all"], default="sup")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tiers", default=None, help="Comma-sep model tiers (default all).")
    ap.add_argument("--sizes", default=None, help="Comma-sep data sizes (default all).")
    ap.add_argument("--lanes", type=int, default=3,
                    help="Concurrent SFT jobs per GPU (each ~1/4 GPU). Default 3.")
    ap.add_argument("--gpus-total", type=int, default=0,
                    help="Physical GPUs on this node (0 = autodetect).")
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--rankers-dir", default=str(RANKERS_DIR))
    args = ap.parse_args()

    rankers_dir = Path(args.rankers_dir)
    tiers = args.tiers.split(",") if args.tiers else list(TIER_ORDER)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(TRAIN_SIZES)
    for t in tiers:
        if t not in MODEL_TIERS:
            log.error("Unknown tier %s (have %s)", t, ", ".join(TIER_ORDER))
            sys.exit(1)

    n_gpus = args.gpus_total
    if n_gpus <= 0:
        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            n_gpus = 8
    n_gpus = max(n_gpus, 1)

    phases = ["bpmf", "sup", "rl"] if args.phase == "all" else [args.phase]

    overall_failed = []
    for phase in phases:
        if _shutdown.is_set():
            break
        if phase == "bpmf":
            all_jobs = _bpmf_jobs(sizes, rankers_dir)
        elif phase == "sup":
            all_jobs = _sup_jobs(tiers, sizes, rankers_dir)
        else:
            all_jobs = _rl_jobs(tiers, sizes, rankers_dir)

        # Partition across nodes (round-robin preserves the seed-outer mix).
        jobs = [j for i, j in enumerate(all_jobs) if i % args.num_nodes == args.node]
        slots = _slots(phase, n_gpus, args.lanes)

        log.info("=== phase %s: %d jobs on node %d/%d (%d total), %d slots (%s) ===",
                 phase, len(jobs), args.node, args.num_nodes, len(all_jobs),
                 len(slots), "1/GPU" if phase == "bpmf"
                 else f"{args.lanes}/GPU" if phase == "sup" else "4-GPU groups")
        if not jobs:
            log.info("Nothing to run for phase %s.", phase)
            continue
        failed = _run_jobs(jobs, slots, rankers_dir, args.dry_run)
        overall_failed += failed

    if overall_failed:
        log.error("Failed runs (%d): %s", len(overall_failed),
                  ", ".join(overall_failed))
        sys.exit(1)
    if _shutdown.is_set():
        log.info("Stopped early - re-run the same command to resume.")
    else:
        log.info("All requested phases complete.")


if __name__ == "__main__":
    main()
