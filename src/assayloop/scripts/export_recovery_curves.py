"""Export per-step hit-recovery curves for every method in the f2 LaTeX table.

For each (method, screen, step) it emits the cumulative acquisition state so the
enrichment / recovery curves in `full_genome_baselines_f2.tex` can be
reconstructed downstream. Picks are classified against the screen ground truth
exactly as the table does:

  n_in_library            gene is in the screen's assay library (chargeable, scorable)
  n_in_universe_not_lib   real gene in the f2 universe but not this library (FORGIVEN by EF)
  n_out_of_universe       gene not in the f2 universe (hallucinated / out-of-domain)
  cum_hits                in-library picks that are true hits

Direct gene-list LLM curves replay the raw logged response against the f2
universe, matching the table's open-vocabulary scoring. Other methods use the
genes the run acquired. The finetuned-LLM harness records the genes it actually
assayed in ``new_hits`` and ``new_misses``; see :func:`_batches_from_jsonl`.

Two files are written to output/analysis/:
  recovery_curves_by_screen.csv   one row per (method, screen, step)  [full detail]
  recovery_curves_mean.csv        one row per (method, step), averaged over screens

    uv run python -m assayloop.scripts.export_recovery_curves
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from assayloop import config
from assayloop.llm.replay import load_logged_response_texts, replay_llm_steps
from assayloop.tasks import load_screens
from assayloop.scripts.full_genome_table import (
    METHODS, HANDOFF_METHODS, JSONL_HANDOFF_METHODS, RAW_SWEEP_METHODS,
    LLM_METHODS, JSONL_METHODS, RUNS_DIR, SHARED_RUNS_DIR,
    _handoff_sweep_id,
    clean_label as _clean_label, resolve_name as _resolve_name,
)
from assayloop.scripts.results_index import (
    _load_all_sweeps, load_sweep_index, resolve_row, _find_run_result,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_recovery_curves")

ANALYSIS_DIR = config.OUTPUT_PATH / "analysis"
MIN_SCREEN_FREQ = 2
BATCH_SIZE = 100
SWEEP_TAG = f"fg-f{MIN_SCREEN_FREQ}"


def _classify_curve(gene_batches, lib, hitset, universe_set):
    """Accumulate cumulative counts over an ordered list of per-step gene lists.

    Returns a list of per-step dicts (cumulative + per-step-new counts).
    """
    n1 = n2 = n3 = cum_hits = 0
    rows = []
    prev = (0, 0)  # (cum_acquired, cum_hits) for per-step deltas
    for si, batch in enumerate(gene_batches, 1):
        for g in batch:
            if g in lib:
                n1 += 1
                if g in hitset:
                    cum_hits += 1
            elif g in universe_set:
                n2 += 1
            else:
                n3 += 1
        n_acq = n1 + n2 + n3
        rows.append({
            "step": si,
            "budget_requested": si * BATCH_SIZE,
            "n_acquired": n_acq,
            "n_in_library": n1,
            "n_in_universe_not_lib": n2,
            "n_out_of_universe": n3,
            "cum_hits": cum_hits,
            "new_acquired": n_acq - prev[0],
            "new_hits": cum_hits - prev[1],
        })
        prev = (n_acq, cum_hits)
    return rows


def _batches_from_result(fp: Path):
    rd = json.loads(fp.read_text())
    return [step.get("acquired_batch", []) for step in rd.get("steps", [])]


def _batches_from_jsonl(js: dict):
    """The genes a finetuned-LLM run actually assayed, round by round.

    Not the same as the round's ``submitted`` list. That harness walked the
    submission in order and took the first 100 genes that were in the screen's
    library and not already acquired, then stopped: everything after the cutoff,
    and every out-of-library or repeated name before it, was never assayed.
    Counting ``submitted`` therefore credits these methods with picks that cost
    nothing and revealed nothing -- 2.0x the budget on the base model by round
    ten -- and puts them on a different x axis from every other row.

    ``new_hits`` + ``new_misses`` is the harness's own record of what it took,
    so use that rather than re-deriving the cut here. It is never more than 100
    genes, which is what makes the axis comparable again.
    """
    batches = []
    for r in js.get("round_genes", []):
        if not (isinstance(r, dict) and "new_hits" in r and "new_misses" in r):
            step = r.get("step", "?") if isinstance(r, dict) else "?"
            raise ValueError(
                f"{js.get('dataset_name', '?')} round {step} records no "
                "new_hits/new_misses, only what was submitted, so which genes "
                "were assayed cannot be recovered from this file."
            )
        batches.append(list(r["new_hits"]) + list(r["new_misses"]))
    return batches


def _check_recorded_hits(method, js, batches, hitset):
    """Cross-check a reconstructed curve against the run's own hit tally.

    ``per_round`` was written by the harness that produced the batches, so its
    final ``cum_hits`` counts the same thing this script is about to recount
    from the screen ground truth. If the two disagree, the JSONL and the loaded
    screen are not describing the same experiment, and nothing downstream would
    catch it.
    """
    per_round = js.get("per_round") or []
    recorded = per_round[-1].get("cum_hits") if per_round else None
    if recorded is None:
        return
    got = sum(1 for b in batches for g in b if g in hitset)
    if int(recorded) != got:
        raise ValueError(
            f"{method} / {js.get('dataset_name', '?')}: the run recorded "
            f"{int(recorded)} cumulative hits, but its assayed genes score "
            f"{got} against this screen's ground truth."
        )


def main():
    screens = load_screens(target_set="public")
    all_genes = {g for s in screens for g in s.genes}
    freq = Counter()
    for s in screens:
        for g in set(s.genes):
            freq[g] += 1
    universe = sorted(g for g in all_genes if freq[g] >= MIN_SCREEN_FREQ)
    universe_set = set(universe)
    U = len(universe)
    log.info("universe (f%d): %d genes, %d screens", MIN_SCREEN_FREQ, U, len(screens))

    screen_by_name = {s.dataset_name: s for s in screens}

    def screen_info(s):
        lib = set(s.genes)
        hitset = {g for g, h in zip(s.genes, s.hits) if h}
        return lib, hitset

    # rows: list of dicts (method, screen, + curve fields + constants)
    out_rows = []

    def emit(method, s, batches):
        lib, hitset = screen_info(s)
        for r in _classify_curve(batches, lib, hitset, universe_set):
            out_rows.append({
                "method": method,
                "screen": s.dataset_name,
                **r,
                "total_hits": s.total_hits,
                "frac_hits": (r["cum_hits"] / s.total_hits) if s.total_hits else 0.0,
                "library_size": len(s.genes),
                "universe_size": U,
            })

    def emit_sweep(method, sweep_id, dirs=(RUNS_DIR,)):
        """Sweep-family: run dir = f'{sweep_id}-{i:02d}-{name}'."""
        found = 0
        for i, s in enumerate(screens):
            fp = None
            for d in dirs:
                cand = d / f"{sweep_id}-{i:02d}-{s.dataset_name}" / "result.json"
                if cand.is_file():
                    fp = cand
                    break
            if fp is None:
                continue
            emit(method, s, _batches_from_result(fp))
            found += 1
        return found

    # 1) METHODS (transformer/BPMF/classical/ablations) --------------------
    last = None
    for label, *_rest, sweep_suffix in METHODS:
        clean = _clean_label(label)
        name, last = _resolve_name(clean, last)
        sweep_id = f"sweep-{SWEEP_TAG}-{sweep_suffix}"
        n = emit_sweep(name, sweep_id)
        log.info("%-40s %3d screens", name, n)

    random_sweep = f"sweep-{SWEEP_TAG}-fg-random-uniform"
    n = emit_sweep("Random", random_sweep)
    log.info("%-40s %3d screens", "Random", n)

    # 2) RAW_SWEEP (BioBO, Haystacks) -- RUNS or SHARED --------------------
    last = None
    for label, sweep_id in RAW_SWEEP_METHODS:
        name, last = _resolve_name(_clean_label(label), last)
        n = emit_sweep(name, sweep_id, dirs=(RUNS_DIR, SHARED_RUNS_DIR))
        log.info("%-40s %3d screens", name, n)

    # 3) HANDOFF + JSONL_HANDOFF (sweep result.json) ----------------------
    last = None
    for label, *_r, sweep_suffix in HANDOFF_METHODS:
        name, last = _resolve_name(_clean_label(label), last)
        n = emit_sweep(name, _handoff_sweep_id(SWEEP_TAG, sweep_suffix))
        log.info("%-40s %3d screens", name, n)
    last = None
    for label, _path, _ck, _ckpt_file, _nw, sweep_suffix in JSONL_HANDOFF_METHODS:
        name, last = _resolve_name(_clean_label(label), last)
        n = emit_sweep(name, _handoff_sweep_id(SWEEP_TAG, sweep_suffix))
        log.info("%-40s %3d screens", name, n)

    # 4) LLM_METHODS (resolve sweep -> per_screen run_id -> result.json) --
    all_sweeps = _load_all_sweeps()
    index = load_sweep_index(all_sweeps)
    last = None
    for label, lookup_key in LLM_METHODS:
        name, last = _resolve_name(_clean_label(label), last)
        hit = resolve_row(lookup_key, index)
        if hit is None:
            log.warning("%-40s NOT FOUND (%s)", name, lookup_key)
            continue
        _, _, sd = hit
        is_direct_llm = str(sd.get("config", {}).get("acq", "")).startswith(
            "llm_single"
        )
        found = 0
        for ps in sd.get("per_screen", []):
            sn = ps.get("screen_name", "")
            s = screen_by_name.get(sn)
            rid = ps.get("run_id", "")
            fp = _find_run_result(rid) if rid else None
            if s is None or fp is None:
                continue
            if is_direct_llm:
                run_data = json.loads(fp.read_text())
                steps = run_data.get("steps") or []
                batches = replay_llm_steps(
                    steps,
                    universe,
                    batch_size=BATCH_SIZE,
                    response_texts=load_logged_response_texts(
                        fp.parent, expected_steps=len(steps)
                    ),
                )
            else:
                batches = _batches_from_result(fp)
            emit(name, s, batches)
            found += 1
        log.info("%-40s %3d screens", name, found)

    # 5) JSONL_METHODS (finetuned LLMs) -----------------------------------
    last = None
    for label, jsonl_path, _fmt in JSONL_METHODS:
        name, last = _resolve_name(_clean_label(label), last)
        if not Path(jsonl_path).is_file():
            log.warning("%-40s NOT FOUND (%s)", name, jsonl_path)
            continue
        found = 0
        with open(jsonl_path) as fh:
            for line in fh:
                js = json.loads(line)
                s = screen_by_name.get(js.get("dataset_name", ""))
                if s is None:
                    continue
                batches = _batches_from_jsonl(js)
                _check_recorded_hits(name, js, batches, screen_info(s)[1])
                emit(name, s, batches)
                found += 1
        log.info("%-40s %3d screens", name, found)

    # ---- write per-screen detail ----
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["method", "screen", "step", "budget_requested", "n_acquired",
            "n_in_library", "n_in_universe_not_lib", "n_out_of_universe",
            "cum_hits", "new_acquired", "new_hits", "frac_hits",
            "total_hits", "library_size", "universe_size"]
    p1 = ANALYSIS_DIR / "recovery_curves_by_screen.csv"
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in cols})
    log.info("wrote %s (%d rows)", p1, len(out_rows))

    # ---- write per-method mean-over-screens curve ----
    agg = defaultdict(lambda: defaultdict(list))
    order = []
    for r in out_rows:
        key = (r["method"], r["step"])
        if r["method"] not in order:
            order.append(r["method"])
        agg[key]["n_acquired"].append(r["n_acquired"])
        agg[key]["n_in_library"].append(r["n_in_library"])
        agg[key]["n_in_universe_not_lib"].append(r["n_in_universe_not_lib"])
        agg[key]["n_out_of_universe"].append(r["n_out_of_universe"])
        agg[key]["cum_hits"].append(r["cum_hits"])
        agg[key]["frac_hits"].append(r["frac_hits"])
    p2 = ANALYSIS_DIR / "recovery_curves_mean.csv"
    mcols = ["method", "step", "budget_requested", "n_screens",
             "mean_n_acquired", "mean_n_in_library", "mean_n_in_universe_not_lib",
             "mean_n_out_of_universe", "mean_cum_hits", "mean_frac_hits",
             "sem_frac_hits"]
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(mcols)
        for method in order:
            for step in range(1, 11):
                a = agg.get((method, step))
                if not a:
                    continue
                fh = np.array(a["frac_hits"], float)
                w.writerow([
                    method, step, step * BATCH_SIZE, len(fh),
                    f"{np.mean(a['n_acquired']):.2f}",
                    f"{np.mean(a['n_in_library']):.2f}",
                    f"{np.mean(a['n_in_universe_not_lib']):.2f}",
                    f"{np.mean(a['n_out_of_universe']):.2f}",
                    f"{np.mean(a['cum_hits']):.3f}",
                    f"{np.mean(fh):.4f}",
                    f"{(np.std(fh, ddof=1) / np.sqrt(len(fh))) if len(fh) > 1 else 0.0:.4f}",
                ])
    log.info("wrote %s (%d methods)", p2, len(order))


if __name__ == "__main__":
    main()
