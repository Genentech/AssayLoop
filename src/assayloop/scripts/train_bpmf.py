#!/usr/bin/env python3
"""Train a BPMF model via Gibbs sampling and save to shared storage.

Usage:
    uv run python -m assayloop.scripts.train_bpmf
    uv run python -m assayloop.scripts.train_bpmf --target-set public_train --K 10 --n-iter 2000

``--target-set public_train`` (the 1,349-screen training fold) is what the paper's
K=10 factorisation is fit on.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


from assayloop import config

_OUTPUT_ROOT = config.OUTPUT_PATH / "bpmf"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train BPMF model via Gibbs sampling")
    parser.add_argument("--target-set", default="public_train", help="Screen set to train on (default: public_train)")
    parser.add_argument("--K", type=int, default=10, help="Latent dimension (default: 10)")
    parser.add_argument("--n-iter", type=int, default=2000, help="Total Gibbs iterations (default: 2000)")
    parser.add_argument("--burn-in", type=int, default=1000, help="Burn-in iterations to discard (default: 1000)")
    parser.add_argument("--thin", type=int, default=2, help="Thinning interval after burn-in (default: 2)")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="Prior std for screen factors (default: 1.0)")
    parser.add_argument("--sigma-v", type=float, default=1.0, help="Prior std for gene factors (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory (default: auto-generated under $ASSAYLOOP_OUTPUT/bpmf/)")
    args = parser.parse_args(argv)

    from assayloop.models.bayesian_pmf import build_hit_matrix, gibbs_bpmf
    from assayloop.tasks import load_screens

    config = {
        "target_set": args.target_set,
        "K": args.K,
        "n_iter": args.n_iter,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "sigma_u": args.sigma_u,
        "sigma_v": args.sigma_v,
        "seed": args.seed,
    }

    print(f"Loading screens from target_set={args.target_set!r} ...")
    screens = load_screens(target_set=args.target_set)
    print(f"Loaded {len(screens)} screens")

    print("Building hit matrix ...")
    Y, mask, screen_names, gene_names = build_hit_matrix(screens)
    print(f"Hit matrix: {Y.shape[0]} screens x {Y.shape[1]} genes")
    print(f"Observed entries: {mask.sum()} / {mask.size} ({100 * mask.mean():.1f}%)")
    print(f"Hit rate (observed): {Y[mask].mean():.4f}")

    print(f"\nRunning Gibbs sampler: K={args.K}, n_iter={args.n_iter}, "
          f"burn_in={args.burn_in}, thin={args.thin} ...")
    result = gibbs_bpmf(
        Y,
        K=args.K,
        sigma_u=args.sigma_u,
        sigma_v=args.sigma_v,
        n_iter=args.n_iter,
        burn_in=args.burn_in,
        thin=args.thin,
        seed=args.seed,
        mask=mask,
    )
    result.gene_names = gene_names
    result.screen_names = screen_names

    n_samples = result.V_samples.shape[0]
    print(f"\nSampling complete: {n_samples} posterior samples collected")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = _OUTPUT_ROOT / f"bpmf_{args.target_set}_K{args.K}_{ts}"

    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "bpmf_result.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved BPMFResult to {result_path}")

    config["n_screens"] = len(screen_names)
    config["n_genes"] = len(gene_names)
    config["n_posterior_samples"] = n_samples
    config["observed_frac"] = float(mask.mean())
    config["hit_rate"] = float(Y[mask].mean())
    config["timestamp"] = datetime.now(timezone.utc).isoformat()

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved config to {config_path}")

    np.savetxt(out_dir / "screen_names.txt", screen_names, fmt="%s")
    np.savetxt(out_dir / "gene_names.txt", gene_names, fmt="%s")
    print(f"Saved screen/gene name lists to {out_dir}")

    print(f"\nDone. Output directory: {out_dir}")
    print(f"Use with bpmf model:  --model-param checkpoint_path={result_path}")


if __name__ == "__main__":
    main()
