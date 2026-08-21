"""torchrun entrypoint for distributed RL fine-tuning.

Launched by ``assayloop train-ranker-rl --gpus N`` (which execs)::

    torchrun --nproc_per_node=N --module assayloop.amortized.rl_ddp \
        --kwargs-json <path>

All ``run_rl_training`` arguments are passed via a JSON file (simpler and less
error-prone than mirroring ~30 flags). Rank/world-size are read from the
torchrun environment inside ``run_rl_training``; only rank 0 evaluates and writes
artifacts. The post-training eval sweep is run by the *parent* (non-distributed)
process, so it is intentionally not done here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rl import run_rl_training


def main() -> None:
    ap = argparse.ArgumentParser(description="Distributed RL ranker worker.")
    ap.add_argument("--kwargs-json", required=True,
                    help="Path to a JSON dict of run_rl_training kwargs.")
    args = ap.parse_args()
    kwargs = json.loads(Path(args.kwargs_json).read_text())
    run_rl_training(**kwargs)


if __name__ == "__main__":
    main()
