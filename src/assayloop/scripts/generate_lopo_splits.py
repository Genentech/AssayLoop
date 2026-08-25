"""Regenerate the leave-one-phenotype-out (LOPO) cross-validation manifests.

Pools ALL public screens across train/validation/test yearfold0 partitions
(1901 screens), groups by ``cleaned_phenotype`` (5 categories), and writes 10
manifests (5 train + 5 test) named ``lopo-<phenotype>-{train,test}.yaml``.

The manifests that the code actually reads ship inside assaybench
(:mod:`assaybench.data.screen_sets`), because which screens a fold contains is
part of the benchmark definition. This script does not write there -- it would
be writing into an installed package -- so it writes to ``--out-dir`` and
prints where to copy the files if you mean to update the shipped set.

Usage::

    uv run python -m assayloop.scripts.generate_lopo_splits
    uv run python -m assayloop.scripts.generate_lopo_splits --out-dir /tmp/lopo
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from assaybench import load_screens
from assaybench.data.screen_sets import manifest_path

from assayloop import config

SLUG_MAP = {
    "Fitness / Proliferation / Viability": "fitness",
    "Drug / Chemical / Environmental Response": "drug",
    "Host-Pathogen / Infection Response": "infection",
    "Molecular Output / Reporter / Pathway Activity": "molecular",
    "Trafficking / Localization / Structural Phenotypes": "trafficking",
}

def _default_out_dir() -> Path:
    return config.OUTPUT_PATH / "screen_sets"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Where to write the manifests (default: $ASSAYLOOP_OUTPUT/screen_sets).",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all public screens (train + validation + test)...")
    all_screens = load_screens(
        split_field="yearfold0",
        split_value=["train", "validation", "test"],
        strict=False,
    )
    print(f"Total: {len(all_screens)} screens\n")

    by_pheno: dict[str, list] = defaultdict(list)
    for s in all_screens:
        by_pheno[s.cleaned_phenotype].append(s)

    print(f"{'Phenotype':<55} {'Count':>6}")
    print("-" * 63)
    for pheno in sorted(by_pheno, key=lambda p: -len(by_pheno[p])):
        print(f"  {pheno:<53} {len(by_pheno[pheno]):>6}")
    print()

    for held_out_pheno, slug in SLUG_MAP.items():
        test_screens = by_pheno[held_out_pheno]
        train_screens = [
            s for p, ss in by_pheno.items() if p != held_out_pheno for s in ss
        ]

        for suffix, screens in [("train", train_screens), ("test", test_screens)]:
            if suffix == "train":
                desc = (
                    f"LOPO fold: train on all phenotypes EXCEPT "
                    f"'{held_out_pheno}' ({len(screens)} screens from "
                    f"all yearfold0 partitions)."
                )
            else:
                desc = (
                    f"LOPO fold: held-out test set for phenotype "
                    f"'{held_out_pheno}' ({len(screens)} screens from "
                    f"all yearfold0 partitions)."
                )

            yaml_obj = {
                "version": 2,
                "description": desc,
                "source": {
                    "dataset": "Genentech/assaybench",
                    "config": "biogrid",
                    "split_field": "yearfold0",
                    "split_value": ["train", "validation", "test"],
                },
                "screens": [
                    {
                        "dataset_name": s.dataset_name,
                        "cleaned_phenotype": s.cleaned_phenotype,
                    }
                    for s in sorted(screens, key=lambda s: s.dataset_name)
                ],
            }
            out_path = out_dir / f"lopo-{slug}-{suffix}.yaml"
            out_path.write_text(
                yaml.dump(yaml_obj, default_flow_style=False, sort_keys=False)
            )
            print(f"  {out_path.name}: {len(screens)} screens")

    print(f"\nDone. Files written to {out_dir}/")
    shipped = manifest_path("lopo-drug-train").parent
    print(
        "These are NOT the manifests the code reads. To update those, copy "
        f"over the shipped ones in\n  {shipped}\n"
        "(in a source checkout: AssayBench/src/assaybench/data/screen_sets/)."
    )


if __name__ == "__main__":
    main()
