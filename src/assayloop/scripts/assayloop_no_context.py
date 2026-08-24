"""AssayLoop with no hit feedback anywhere in the pipeline.

The blind arm of the label ablation for the Gemini -> AssayFormer handoff. Both
halves are blinded, each by its own mechanism -- the same two mechanisms the
rest of :mod:`assayloop.scripts.plot_label_ablation` uses:

* **Gemini-3.1-Pro**, rounds 1-3 -- its recorded blind-prompt picks
  (``llm_single_blind``, sweep ``cd32999e``): the same 10 x 100 loop with the
  per-round hit labels stripped from the prompt.
* **AssayFormer**, rounds 4-10 -- its *opening* ranking against an empty
  observation history (``assayformer_no_context.py``), with Gemini's picks
  removed and the next 700 genes taken in score order. No readout is ever seen,
  by either half.

So this is not a handoff in the usual sense: nothing is handed over, because
there is nothing to hand over. It is the same 1000-gene budget spent by the two
components acting on their priors alone, and it is what the full AssayLoop run
(``sweep-fg-f2-fg-handoff-gemini-s19-n3``) has to beat for the feedback loop to
be earning its place.

Everything is assembled from cached runs -- the blind Gemini traces and the
no-context AssayFormer ranking both already exist -- and replayed through the
standard loop, then scored with the table's own domain-adjusted EF/nAUC
(``_adj_ef_from_run`` / ``_adj_nauc_from_run``) rather than the raw
``n_hits_vs_random``, so the with-feedback side reproduces the LaTeX row
(EF 5.66, nAUC 21.7) exactly and the difference is a like-for-like one. The
ranking is 1000 genes long and Gemini's first three rounds overlap it by at
most 252, so the 700 are always available.

Usage::

    uv run python -m assayloop.scripts.assayloop_no_context [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import numpy as np

from assayloop import config
from assayloop.acquisitions.glm_handoff_acq import GlmHandoffAcquisition
from assayloop.acquisitions.greedy_from_model import GreedyFromModel
from assayloop.amortized.warmstart import load_run_traces
from assayloop.experiment.runner import RunConfig, run_one_screen
from assayloop.scripts.full_genome_table import (
    _adj_nauc_from_run, _adj_ef_from_run)
from assayloop.tasks import load_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assayloop_no_context")

RUNS_DIR = config.OUTPUT_PATH / "runs"
ANALYSIS = config.OUTPUT_PATH / "analysis"
OUT = ANALYSIS / "assayloop_no_context.json"

BUDGET = 1000
BATCH = 100
N_STEPS = 10
N_WARM = 3                                          # Gemini rounds, as in the table row
MIN_SCREEN_FREQ = 2

SWEEP_BLIND = "sweep-fg-f2-fg-assayloop-noctx"      # this script
SWEEP_FULL = "sweep-fg-f2-fg-handoff-gemini-s19-n3"  # the table's AssayLoop row
BLIND_LLM_PREFIX = "sweep-cd32999e-"                # Gemini, hit labels stripped
NOCTX_SWEEP = "sweep-fg-f2-fg-assayformer-noctx"    # AssayFormer's opening ranking


def _universe(screens) -> list[str]:
    """The f2 universe: genes appearing in >= 2 test screens (drops pseudogenes)."""
    freq = Counter()
    for s in screens:
        for g in set(s.genes):
            freq[g] += 1
    return sorted(g for g in {g for s in screens for g in s.genes}
                  if freq[g] >= MIN_SCREEN_FREQ)


def _noctx_ranking(index: int, name: str) -> list[str]:
    """AssayFormer's opening ranking for one screen, best first."""
    p = RUNS_DIR / f"{NOCTX_SWEEP}-{index:02d}-{name}" / "result.json"
    if not p.is_file():
        return []
    steps = json.loads(p.read_text()).get("steps") or []
    return [g for s in steps for g in (s.get("acquired_batch") or [])]


def _blind_rounds(llm_rounds: list[list[str]], ranking: list[str]) -> list[list[str]]:
    """Gemini's blind first 3 rounds, then the prior ranking in batches of 100."""
    warm = [list(r) for r in llm_rounds[:N_WARM]]
    taken = {g for r in warm for g in r}
    tail = [g for g in ranking if g not in taken]
    need = N_STEPS - N_WARM
    return warm + [tail[i * BATCH:(i + 1) * BATCH] for i in range(need)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    args = ap.parse_args()

    screens = load_screens(target_set="public")
    universe = _universe(screens)
    universe_set = set(universe)
    log.info("f2 universe: %d genes, %d screens", len(universe), len(screens))

    traces = load_run_traces(RUNS_DIR, BLIND_LLM_PREFIX)
    cfg = RunConfig(
        screen_set="public", model="null", acq="greedy",
        batch_size=BATCH, n_steps=N_STEPS,
        persist=True, metrics=["hits_auc"],
        max_shortfall_frac=1.0, universe_genes=universe,
    )

    rows = []
    for i, screen in enumerate(screens):
        name = screen.dataset_name
        run_id = f"{SWEEP_BLIND}-{i:02d}-{name}"
        cached = RUNS_DIR / run_id / "result.json"
        if cached.is_file() and not args.force:
            log.info("  [%2d/%d] %-28s (cached)", i + 1, len(screens), name)
        else:
            llm_rounds = (traces.get(name) or [[]])[0]
            ranking = _noctx_ranking(i, name)
            if not llm_rounds or not ranking:
                log.warning("  [%2d/%d] %-28s SKIPPED (blind trace %s, ranking %s)",
                            i + 1, len(screens), name,
                            "ok" if llm_rounds else "missing",
                            "ok" if ranking else "missing")
                continue
            rounds = _blind_rounds(llm_rounds, ranking)
            log.info("  [%2d/%d] %-28s picks/round %s", i + 1, len(screens), name,
                     [len(r) for r in rounds])
            # n_warm = N_STEPS: every round is a replay, so `base` is never
            # reached and no model is ever consulted -- which is the point.
            acq_obj = GlmHandoffAcquisition(
                rounds=rounds, n_warm=N_STEPS, base=GreedyFromModel(seed=0))
            run_one_screen(screen, cfg, model_obj=None, acq_obj=acq_obj,
                           verbose=False, run_id=run_id, sweep_id=SWEEP_BLIND)

        # Score both arms the way the LaTeX table does: domain-adjusted EF and
        # nAUC re-derived from the pick stream, not the raw final_metrics.
        lib = set(screen.genes)
        total_hits = sum(screen.hits)

        def _score(p):
            if not p.is_file():
                return None, None
            return (
                _adj_ef_from_run(p, lib, universe_set, total_hits, len(lib), BUDGET),
                100.0 * _adj_nauc_from_run(p, lib, universe_set, total_hits,
                                           len(lib), BUDGET),
            )

        blind_ef, blind_nauc = _score(cached)
        full_ef, full_nauc = _score(
            RUNS_DIR / f"{SWEEP_FULL}-{i:02d}-{name}" / "result.json")
        rows.append({
            "screen": name,
            "blind_ef": blind_ef, "full_ef": full_ef,
            "blind_nauc": blind_nauc, "full_nauc": full_nauc,
        })

    def _paired(key_full, key_blind):
        pairs = [(r[key_full], r[key_blind]) for r in rows
                 if r[key_full] is not None and r[key_blind] is not None]
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        d = a - b
        return a, b, d, float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None

    a, b, d, se = _paired("full_ef", "blind_ef")
    na, nb, nd, nse = _paired("full_nauc", "blind_nauc")
    summary = {
        "n_screens": len(a),
        "with_feedback_ef": float(a.mean()), "blind_ef": float(b.mean()),
        "delta_ef": float(d.mean()), "paired_se_ef": se,
        "wins_ef": int((d > 0).sum()),
        "with_feedback_nauc": float(na.mean()) if len(na) else None,
        "blind_nauc": float(nb.mean()) if len(nb) else None,
        "delta_nauc": float(nd.mean()) if len(nd) else None,
        "paired_se_nauc": nse,
        "per_screen": rows,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1))

    print(f"\n{'screen':30s} {'AssayLoop':>10s} {'blind':>8s} {'delta':>8s}")
    for r in rows:
        if r["full_ef"] is None or r["blind_ef"] is None:
            continue
        print(f"{r['screen'][:30]:30s} {r['full_ef']:10.2f} {r['blind_ef']:8.2f} "
              f"{r['full_ef'] - r['blind_ef']:8.2f}")
    print(f"\nAssayLoop  EF   with feedback {a.mean():.2f}   blind {b.mean():.2f}   "
          f"delta {d.mean():+.2f} +/- {se:.2f} (paired SE, n={len(d)}), "
          f"{summary['wins_ef']}/{len(d)} screens improve")
    if len(na):
        print(f"AssayLoop  nAUC with feedback {na.mean():.1f}   blind {nb.mean():.1f}   "
              f"delta {nd.mean():+.1f} +/- {nse:.1f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
