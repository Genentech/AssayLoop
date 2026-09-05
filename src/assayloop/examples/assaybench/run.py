"""Option 1 driver: per-target inner loops over an evaluation set.

Iterates over a list of target screens (default: the 20 public test screens;
``--target-set paper_validation`` for the validation set, or pass
``--targets`` for explicit ``dataset_name``s) and runs one
``SequentialLoop`` per target. Candidate training screens are drawn from
``--pool-set``, the public training fold by default.

The model is :class:`Option1LLMRanker` and the acquisition is
configurable between :class:`RandomScreenAcquisition` and
:class:`LLMScreenAcquisition`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from assaybench.core import SequentialLoop

from ... import config
from ...llm.client import LLMClientConfig
from ...tracing import run_trace_scope
from .acquisition import LLMScreenAcquisition, RandomScreenAcquisition
from .metric import Option1AnDCG
from .model import Option1LLMRanker
from .task import AssayBenchScreenSelectionTask, build_pool


def run_option1(
    *,
    target_set: str = "paper_test",
    targets: list[str] | None = None,
    pool_set: str = "train",
    acquisition: str = "random",
    n_steps: int = 5,
    n_predictions: int = 100,
    seed: int = 0,
    llm: LLMClientConfig | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
):
    load_dotenv()
    out_dir = out_dir or (config.OUTPUT_PATH / "option1")
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = build_pool(target_set=pool_set)
    pool_by_name = {s.dataset_name: s for s in pool}

    if targets:
        target_names = list(targets)
    else:
        eval_set = build_pool(target_set=target_set)
        target_names = [s.dataset_name for s in eval_set]

    if verbose:
        print(f"[option1] {len(target_names)} target screens / pool size {len(pool)}")

    rows = []
    t0 = time.time()
    for i, name in enumerate(target_names):
        target = pool_by_name.get(name)
        if target is None:
            if verbose:
                print(f"  [{i + 1:>2}/{len(target_names)}] {name}: not in pool, skipping")
            continue

        task = AssayBenchScreenSelectionTask(target=target, pool=pool, seed=seed)
        model = Option1LLMRanker(target=target, llm=llm, n_predictions=n_predictions)
        if acquisition == "llm":
            acq = LLMScreenAcquisition(llm=llm)
        elif acquisition == "random":
            acq = RandomScreenAcquisition(seed=seed)
        else:
            raise ValueError(f"Unknown acquisition {acquisition!r}")

        loop = SequentialLoop(
            task=task,
            model=model,
            acquisition=acq,
            metrics=Option1AnDCG(),
            batch_size=1,
            run_id=f"option1-{name[:40].replace('/', '_')}",
            # Without this the ranker's LLM calls are made outside any trace
            # scope and no llm_calls.jsonl is written for the run.
            trace_scope=run_trace_scope,
        )
        if verbose:
            print(f"\n=== [{i + 1}/{len(target_names)}] target={name} ===")
        result = loop.run(n_steps=n_steps, verbose=verbose)
        final = result.final_metrics or {}
        rows.append({
            "target": name,
            "acquisition": acquisition,
            "n_steps": len(result.history),
            "metrics": final,
        })
        (out_dir / f"{loop.run_id}.json").write_text(
            json.dumps({
                "target": name,
                "config": result.config,
                "final_metrics": final,
            }, indent=2, default=str)
        )

    # Aggregate.
    keys = set()
    for r in rows:
        keys.update(r["metrics"].keys())
    agg = {}
    for k in sorted(keys):
        vals = [r["metrics"].get(k) for r in rows if isinstance(r["metrics"].get(k), (int, float))]
        if vals:
            agg[f"mean_{k}"] = float(sum(vals) / len(vals))
    summary = {
        "target_set": target_set,
        "n_targets": len(rows),
        "acquisition": acquisition,
        "n_steps": n_steps,
        "elapsed_s": round(time.time() - t0, 2),
        "aggregate": agg,
        "per_target": rows,
    }
    (out_dir / f"summary_{acquisition}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if verbose:
        print(f"\n[option1] {len(rows)} targets done in {summary['elapsed_s']}s")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Option 1: AssayBench screen selection")
    p.add_argument("--target-set", default="paper_test",
                   help="paper_test|paper_validation|/path/to.yaml (eval-target list)")
    p.add_argument("--targets", default=None,
                   help="Comma-separated dataset_names to use as targets "
                        "(overrides --target-set).")
    p.add_argument("--pool-set", default="train",
                   help="Pool of candidate training screens (default: the public "
                        "training fold).")
    p.add_argument("--acquisition", default="random", choices=["random", "llm"])
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--n-predictions", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    targets = None
    if args.targets:
        targets = [s.strip() for s in args.targets.split(",") if s.strip()]

    run_option1(
        target_set=args.target_set,
        targets=targets,
        pool_set=args.pool_set,
        acquisition=args.acquisition,
        n_steps=args.n_steps,
        n_predictions=args.n_predictions,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
