#!/usr/bin/env python3
"""Export training screen metadata to a minimal JSON dataset for the agent sandbox.

The ``agent_ranker`` baseline ("Haiku-4.5 Agent") reads this file at
``/data/screens.json`` inside its container and analyses it programmatically to
inform its predictions. Only the public training fold is exported, so the
container holds nothing the released benchmark does not.

For each screen it writes ``dataset_name``, ``question`` (experimental
context), ``cleaned_phenotype``, ``hits`` and ``non_hits``.

Usage::

    uv run python -m assayloop.scripts.export_screen_dataset
    uv run python -m assayloop.scripts.export_screen_dataset \
        --target-set public_train --output output/datasets/public_train_screens.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from assayloop import config

DEFAULT_OUT = config.OUTPUT_PATH / "datasets" / "public_train_screens.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export screen metadata for the agent sandbox")
    parser.add_argument(
        "--target-set", default="public_train",
        help="Screen set to export (default: public_train).",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT),
        help=f"Output JSON path (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    from assayloop.tasks import load_screens

    print(f"Loading screens from target_set={args.target_set!r}...")
    screens = load_screens(target_set=args.target_set)
    print(f"Loaded {len(screens)} screens.")

    records = []
    for s in screens:
        hits = [g for g, h in zip(s.genes, s.hits) if h]
        non_hits = [g for g, h in zip(s.genes, s.hits) if not h]
        records.append({
            "dataset_name": s.dataset_name,
            "question": s.question,
            "cleaned_phenotype": s.cleaned_phenotype,
            "hits": hits,
            "non_hits": non_hits,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} screens to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
