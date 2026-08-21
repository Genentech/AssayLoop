"""AssayFormer with no context -- the transformer's prior ranking, top 1000.

The LLM label ablation strips the per-round hit feedback from the prompt and
re-runs the same 10 x 100 loop. The transformer's analogue is to take its
opening ranking and stop there: one batch of 1000 with an empty observation
history, i.e. the genes it would name before seeing a single readout. Anything
the full run gains over this number is what the feedback loop bought.

EF at a fixed 1000-gene budget is order-invariant, so 1 x 1000 and 10 x 100 are
directly comparable -- the same 1000 picks scored the same way. (nAUC is *not*
order-invariant and is meaningless for a single batch, which is why this script
only reports EF.)

Runs the same checkpoint, universe and screens as the full-genome table
(`full_genome_table.py`, row "\\quad + GRPO (= AssayLoop)"): the f2 universe
(genes in >= 2 test screens) over the 20 public test screens. Results are
cached as ordinary runs under `sweep-fg-f2-fg-assayformer-noctx-NN-<screen>`,
so a second invocation is free.

Usage::

    uv run python -m assayloop.scripts.assayformer_no_context [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import numpy as np

from assayloop import config
from assayloop.experiment.runner import RunConfig, run_one_screen
from assayloop.scripts.full_genome_table import (
    HANDOFF_CKPT_FILE, HANDOFF_RANKER, _load_ranker_model,
)
from assayloop.tasks import load_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assayformer_no_context")

RUNS_DIR = config.OUTPUT_PATH / "runs"
ANALYSIS = config.OUTPUT_PATH / "analysis"
OUT = ANALYSIS / "assayformer_no_context.json"

BUDGET = 1000
MIN_SCREEN_FREQ = 2
SWEEP_NOCTX = "sweep-fg-f2-fg-assayformer-noctx"   # this script
SWEEP_FULL = "sweep-fg-f2-fg-assayloop-s19"        # the 10 x 100 table row


def _universe(screens) -> list[str]:
    """The f2 universe: genes appearing in >= 2 test screens (drops pseudogenes)."""
    freq = Counter()
    for s in screens:
        for g in set(s.genes):
            freq[g] += 1
    return sorted(g for g in {g for s in screens for g in s.genes}
                  if freq[g] >= MIN_SCREEN_FREQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    args = ap.parse_args()

    screens = load_screens(target_set="public")
    universe = _universe(screens)
    log.info("f2 universe: %d genes, %d screens", len(universe), len(screens))

    model = _load_ranker_model(HANDOFF_RANKER, ckpt_file=HANDOFF_CKPT_FILE)
    cfg = RunConfig(
        screen_set="public", model="null", acq="greedy",
        batch_size=BUDGET, n_steps=1,          # <- the whole point: one shot, no history
        persist=True, metrics=["hits_auc"],
        max_shortfall_frac=1.0, universe_genes=universe,
    )

    rows = []
    for i, screen in enumerate(screens):
        name = screen.dataset_name
        run_id = f"{SWEEP_NOCTX}-{i:02d}-{name}"
        cached = RUNS_DIR / run_id / "result.json"
        if cached.is_file() and not args.force:
            fm = json.loads(cached.read_text()).get("final_metrics", {})
            log.info("  [%2d/%d] %-28s (cached)", i + 1, len(screens), name)
        else:
            log.info("  [%2d/%d] %-28s", i + 1, len(screens), name)
            res = run_one_screen(screen, cfg, model_obj=model, verbose=False,
                                 run_id=run_id, sweep_id=SWEEP_NOCTX)
            fm = res.final_metrics if isinstance(res.final_metrics, dict) \
                else dict(res.final_metrics or {})

        full = RUNS_DIR / f"{SWEEP_FULL}-{i:02d}-{name}" / "result.json"
        ffm = json.loads(full.read_text()).get("final_metrics", {}) \
            if full.is_file() else {}
        rows.append({
            "screen": name,
            "noctx_ef": fm.get("n_hits_vs_random"),
            "full_ef": ffm.get("n_hits_vs_random"),
        })

    paired = [(r["full_ef"], r["noctx_ef"]) for r in rows
              if r["full_ef"] is not None and r["noctx_ef"] is not None]
    a = np.array([p[0] for p in paired])       # 10 x 100, with feedback
    b = np.array([p[1] for p in paired])       # 1 x 1000, no context
    d = a - b
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    summary = {
        "n_screens": len(paired),
        "with_feedback_ef": float(a.mean()),
        "no_context_ef": float(b.mean()),
        "delta": float(d.mean()),
        "paired_se": se,
        "wins": int((d > 0).sum()),
        "per_screen": rows,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))

    print(f"\n{'screen':30s} {'10x100':>8s} {'1x1000':>8s} {'delta':>8s}")
    for r in rows:
        if r["full_ef"] is None or r["noctx_ef"] is None:
            continue
        print(f"{r['screen'][:30]:30s} {r['full_ef']:8.2f} {r['noctx_ef']:8.2f} "
              f"{r['full_ef'] - r['noctx_ef']:8.2f}")
    print(f"\nAssayFormer  with feedback {a.mean():.2f}   no context {b.mean():.2f}   "
          f"delta {d.mean():+.2f} +/- {se:.2f} (paired SE, n={len(d)}), "
          f"{summary['wins']}/{len(d)} screens improve")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
