"""Emit a real LLM acquisition prompt for one screen after 1 round of context.

Reproduces exactly what `llm_single` (LLMSingleAcquisition) sends to the LLM:
the system prompt + the screen block + the Active-Learning History block built
from one warm-start round of 100 genes (with true hit/no-hit labels from the
screen). Pick a biologically interesting screen with --match.

    uv run python -m assayloop.scripts.example_llm_prompt --match venetoclax
    uv run python -m assayloop.scripts.example_llm_prompt --list  # show candidates

This one needs the screen corpus, because the point of it is to see a prompt
with real hit labels in it. For the prompt plumbing on its own -- history
rendering and reply parsing, no data, no API key -- see
``examples/sequential_llm_prompt.py`` in the assaybench repo.
"""
from __future__ import annotations

import argparse
import random

from assayloop.acquisitions.llm_single_acq import LLMSingleAcquisition
from assaybench.core.types import Observation, StepRecord
from assayloop.tasks import load_screens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="venetoclax",
                    help="substring to match in phenotype/condition/cell line")
    ap.add_argument("--target-set", default="public")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--warm-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true",
                    help="just list screens matching --match and exit")
    args = ap.parse_args()

    screens = load_screens(target_set=args.target_set)
    m = args.match.lower()

    def hay(s):
        return " ".join(str(x or "").lower() for x in (
            s.phenotype, s.cleaned_phenotype, s.condition_clause,
            s.cell_line, s.cell_type, s.dataset_name))

    matches = [s for s in screens if m in hay(s)]
    if args.list:
        for s in sorted(matches, key=lambda s: -s.total_hits)[:40]:
            print(f"{s.total_hits:5d} hits / {s.num_genes:6d} genes  "
                  f"{s.dataset_name}  |  {s.cell_line}  |  {s.phenotype[:80]}")
        print(f"\n{len(matches)} screens match {args.match!r}")
        return

    if not matches:
        raise SystemExit(f"no screens match {args.match!r}")
    # pick the matching screen with the most hits (richest 1-round signal)
    s = max(matches, key=lambda s: s.total_hits)

    # --- simulate one warm-start round of `warm_size` random genes ---
    rng = random.Random(args.seed)
    idx = list(range(len(s.genes)))
    rng.shuffle(idx)
    idx = idx[:args.warm_size]
    obs = [Observation(candidate=s.genes[i], label={"hit": bool(s.hits[i])})
           for i in idx]
    warm = StepRecord(step=1, acquired_batch=[o.candidate for o in obs],
                      new_observations=obs,
                      acquisition_trace={"warm_start": True})

    acq = LLMSingleAcquisition()
    system, user, n_req = acq._build_prompt([warm], args.batch_size, s.context())

    n_hits = sum(1 for o in obs if o.label["hit"])
    print("=" * 78)
    print(f"SCREEN: {s.dataset_name}")
    print(f"  cell line: {s.cell_line} ({s.cell_type})")
    print(f"  phenotype: {s.phenotype}")
    print(f"  library:   {s.num_genes} genes, {s.total_hits} true hits "
          f"({100*s.total_hits/s.num_genes:.1f}%)")
    print(f"  round-1 warm-start: {n_hits}/{len(obs)} hits "
          f"({100*n_hits/len(obs):.1f}%)")
    print("=" * 78)
    print("\n########## SYSTEM PROMPT ##########\n")
    print(system)
    print("\n########## USER PROMPT ##########\n")
    print(user)


if __name__ == "__main__":
    main()
