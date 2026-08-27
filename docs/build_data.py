#!/usr/bin/env python3
"""Build the JSON the AssayLoop site reads, from the analysis outputs.

The site has no build step at serve time: every page fetches a static JSON from
``assets/data/``. This script writes those files, and it is the only thing that
does. Run it after regenerating the results table and the recovery curves::

    # 1. the results table, which also dumps the JSON this reads
    python -m assayloop.scripts.full_genome_table --min-screen-freq 2 \\
        --json output/tables/full_genome_baselines_f2.json

    # 2. the recovery curves
    python -m assayloop.scripts.export_recovery_curves

    # 3. this
    python docs/build_data.py

Nothing here falls back. A missing input raises and names the path it wanted,
because the failure mode otherwise is a site that renders a complete-looking
leaderboard with rows quietly absent -- which is indistinguishable, to a
reader, from those methods not having been evaluated.

The one way to publish a table with a gap in it is ``--allow-unavailable``,
which makes you name each unevaluated method and write the note the site shows
in their place. That is the opposite of a silent fallback: the gap ends up on
the page rather than in the dashes.
"""
from __future__ import annotations

import argparse
import ast
import math
import csv
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_data")

DOCS = Path(__file__).resolve().parent
DATA_OUT = DOCS / "assets" / "data"
REPO = DOCS.parent

# Methods that export_recovery_curves.py computes but the paper's table omits.
# They still get a recovery curve on the site, so they need a family; naming
# them here rather than defaulting keeps "not in the table" a deliberate state
# instead of the result of a failed lookup.
EXTRA_METHOD_FAMILIES = {
    "Transformer (SVD-Sphere)": ("ablation", "AssayFormer (SVD-Sphere)"),
    "SVD-Sphere + GRPO": ("ablation", "SVD-Sphere + GRPO"),
}

# Curve columns carried to the site, mapped from the CSV's mean_* names.
MEAN_COLUMNS = {
    "frac_hits": "mean_frac_hits",
    "sem_frac_hits": "sem_frac_hits",
    "cum_hits": "mean_cum_hits",
    "n_acquired": "mean_n_acquired",
    "in_library": "mean_n_in_library",
    "in_universe_not_lib": "mean_n_in_universe_not_lib",
    "out_of_universe": "mean_n_out_of_universe",
}
BY_SCREEN_COLUMNS = {
    "frac_hits": "frac_hits",
    "cum_hits": "cum_hits",
    "n_acquired": "n_acquired",
    "out_of_universe": "n_out_of_universe",
}


class MissingInput(FileNotFoundError):
    """An input this script needs has not been generated yet."""


def _require(path: Path, what: str, how: str) -> Path:
    if not path.is_file():
        raise MissingInput(
            f"{what} not found at {path}.\n  Generate it: {how}")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _num(row: dict, col: str, path: Path) -> float:
    """Parse a numeric CSV cell, refusing to guess at a blank one."""
    raw = row.get(col)
    if raw is None:
        raise MissingInput(f"{path} has no column {col!r}; columns are "
                           f"{sorted(row)}")
    if raw == "":
        raise MissingInput(
            f"{path}: empty {col!r} for method {row.get('method')!r} at step "
            f"{row.get('step')!r}. An empty cell is not a zero.")
    return float(raw)


# --------------------------------------------------------------- results

#: Join keys that carry a code-internal name into the served JSON. Applied to
#: the serialised text of every output, after all the joins are done, so the
#: rename cannot desync results.json from the recovery curves keyed by the same
#: string. Values must not collide with an existing key.
SERVED_KEY_RENAMES = {
    "BPMF + GRPO (NVR)": "BPMF + GRPO (EF-terminal)",
}


def _rename_keys(text: str) -> str:
    for old, new in SERVED_KEY_RENAMES.items():
        text = text.replace(old, new)
    return text


#: Rows whose displayed label the site overrides, keyed by the table's join
#: key. Values are (display, name) -- the indented table cell and the
#: standalone name legends and tooltips use.
#:
#: The table's standalone names are composed as "<section's opening row> +
#: <this row>", which is right for a second-level row and wrong for a third.
#: `+ GRPO` continues `+ BPMF Embeddings`, but the composition reaches past it
#: to `Transformer` -- displayed as "AssayFormer (random embs.)" -- and so
#: named the paper's headline model "AssayFormer (random embs.) + GRPO", which
#: says it uses the one initialisation it does not. Naming the row here rather
#: than teaching the composition to be cumulative keeps the change on the site:
#: `_DISPLAY` in full_genome_table.py feeds the paper's LaTeX from the same
#: strings, and the indentation already disambiguates the row there.
LABEL_OVERRIDES = {
    "Transformer + GRPO (= AssayLoop)": (
        "+ BPMF + GRPO (= AssayFormer)",
        "AssayFormer + BPMF + GRPO (= AssayFormer)",
    ),
    # Same artifact one row up, and the same contradiction -- a name cannot say
    # both "random embs." and "+ BPMF Embeddings". The cell keeps its short
    # form; only the standalone name changes.
    "Transformer + BPMF Embeddings": (
        "+ BPMF Embeddings",
        "AssayFormer + BPMF",
    ),
}


def _relabel(payload: dict) -> None:
    """Apply LABEL_OVERRIDES, refusing to override a row that is not there."""
    by_key = {r["key"]: r for r in payload["rows"]}
    unknown = [k for k in LABEL_OVERRIDES if k not in by_key]
    if unknown:
        raise MissingInput(
            f"LABEL_OVERRIDES names rows that are not in the table: "
            f"{', '.join(unknown)}.\n"
            "  Keys must match the `key` field exactly. A rename in "
            "full_genome_table.py's METHODS list will land here.")
    for key, (display, name) in LABEL_OVERRIDES.items():
        row = by_key[key]
        log.info("results: relabelled %r -> %r", row["name"], name)
        row["display"] = display
        row["name"] = name


def _strip_internals(payload: dict) -> None:
    """Drop the table's LaTeX and code-facing fields from what gets served.

    `full_genome_table.py --json` is written for the paper as much as for the
    site: every row carries the ``\\hfill``-laden LaTeX cell it renders into,
    and the EF note explains that the counts on disk are stored under the
    metric's older internal name. Both are right for the repo and wrong for
    the site, which is written for someone who has read the paper and has no
    reason to meet a second name for EF or a stray TeX macro in a JSON file
    they opened out of curiosity. The keys stay as they are -- they join the
    results to the recovery curves -- but nothing here is displayed.
    """
    notes = payload.get("metric_notes", {})
    if "ef" in notes:
        notes["ef"] = ("Enrichment factor: hit rate relative to random, "
                       "domain-adjusted.")
    for row in payload["rows"]:
        row.pop("label", None)


def build_results(table_json: Path, allowed: list[str], note: str | None) -> dict:
    """Pass the table's --json through, after checking it is complete.

    A row with no result is a build error by default. ``allowed`` is the escape
    hatch, and it is deliberately a narrow one: you name every unevaluated
    method on the command line and say why in ``note``, and the note is written
    into the payload so ``results.html`` can put it on the page. The gap is
    then visible to the reader, which is the whole point -- an unevaluated row
    that renders as bare dashes is indistinguishable from a method that was run
    and scored nothing.
    """
    payload = json.loads(table_json.read_text())
    rows = payload["rows"]
    # Before the name-keyed checks below, so --allow-unavailable takes the name
    # the site shows rather than the one the table composed.
    _relabel(payload)

    by_name = {r["name"]: r for r in rows}
    unknown = [n for n in allowed if n not in by_name]
    if unknown:
        raise MissingInput(
            f"--allow-unavailable names rows that are not in {table_json}: "
            f"{', '.join(unknown)}.\n"
            "  Names must match the `name` field exactly.")
    stale = [n for n in allowed if by_name[n]["available"]]
    if stale:
        raise MissingInput(
            f"--allow-unavailable names rows that now have results: "
            f"{', '.join(stale)}.\n"
            "  Drop them from the flag; an allowance nobody removes is how a "
            "future gap gets waved through.")

    missing = [r["name"] for r in rows if not r["available"]]
    unwaived = [n for n in missing if n not in allowed]
    if unwaived:
        raise MissingInput(
            f"{table_json} has {len(unwaived)} of {len(rows)} rows with no "
            f"result: {', '.join(unwaived)}.\n"
            "  The site would show these as dashes, which reads as 'evaluated "
            "and scored nothing' rather than 'not evaluated'. Re-run\n"
            "  full_genome_table.py with ASSAYLOOP_RESULTS pointing at a "
            "complete set of sweeps -- or, if the run genuinely cannot be "
            "reproduced,\n  pass --allow-unavailable with each name above and "
            "--unavailable-note explaining why, which puts the gap on the page."
        )
    if missing and not note:
        raise MissingInput(
            "--allow-unavailable needs --unavailable-note. The note is what "
            "the site shows the reader in place of the missing numbers; "
            "without it the rows are just dashes again.")

    payload["unavailable"] = {
        "names": missing,
        "keys": [by_name[n]["key"] for n in missing],
        "note": note if missing else None,
    }
    _strip_internals(payload)
    if missing:
        log.warning("results: %d of %d rows were not evaluated (%s); the site "
                    "will label them and show the note",
                    len(missing), len(rows), ", ".join(missing))
    log.info("results: %d rows, %d families", len(rows),
             len({r["family"] for r in rows}))
    return payload


def build_diversity(results: dict) -> dict:
    """The pathway-diversity slice, plus who dashed out and why."""
    unavailable = set(results["unavailable"]["names"])
    rows = []
    for r in results["rows"]:
        # ef rides along so the page can ask whether diversity costs
        # enrichment without a second fetch of results.json.
        row = {k: r[k] for k in
               ("key", "display", "name", "family", "section", "indent",
                "ep_b", "ep_s", "ep_d", "vendi", "pathway", "ef")}
        # The diagnostics are what let the page say *why* a scope dashed out
        # instead of just showing a dash and leaving the reader to guess.
        row["ep_diagnostics"] = r["ep_diagnostics"]
        row["available"] = r["available"]
        rows.append(row)
    # An unevaluated method has null EP for the same reason it has null
    # everything: nobody ran it. Listing it under "below the retention floor"
    # would attribute the dash to the metric's contract instead.
    dashed = {
        scope: [r["name"] for r in rows
                if r[scope] is None and r["name"] not in unavailable]
        for scope in ("ep_b", "ep_s", "ep_d")
    }
    for scope, names in dashed.items():
        if names:
            log.info("diversity: %s is null (below the retention floor) for %s",
                     scope, ", ".join(names))
    return {
        "schema": 1,
        "reference_counts": results["ep_reference_counts"],
        "retention_floor": results["ep_retention_floor"],
        "notes": {k: results["metric_notes"][k] for k in
                  ("ep_b", "ep_s", "ep_d", "vendi", "pathway", "ef")},
        "below_floor": dashed,
        "unavailable": results["unavailable"],
        "rows": rows,
    }


# -------------------------------------------------------------- recovery

def _family_lookup(results: dict) -> dict[str, tuple[str, str]]:
    by_key = {r["key"]: (r["family"], r["name"]) for r in results["rows"]}
    return {**EXTRA_METHOD_FAMILIES, **by_key}


def build_recovery_mean(path: Path, results: dict) -> dict:
    rows = _read_csv(path)
    fams = _family_lookup(results)

    methods: dict[str, dict] = {}
    steps: list[int] = []
    for row in rows:
        m = row["method"]
        step = int(row["step"])
        if step not in steps:
            steps.append(step)
        entry = methods.setdefault(m, {k: [] for k in MEAN_COLUMNS})
        entry.setdefault("budget", []).append(int(row["budget_requested"]))
        entry.setdefault("n_screens", int(row["n_screens"]))
        for out_col, csv_col in MEAN_COLUMNS.items():
            entry[out_col].append(_num(row, csv_col, path))

    unknown = sorted(m for m in methods if m not in fams)
    if unknown:
        raise MissingInput(
            f"{path} has methods with no family: {unknown}.\n"
            "  They are neither in the results table nor in "
            "EXTRA_METHOD_FAMILIES at the top of this script. Add them there "
            "with a deliberate family rather than letting the site colour them "
            "by accident."
        )

    out = {}
    for m, entry in methods.items():
        family, name = fams[m]
        out[m] = {"family": family, "name": name,
                  "in_table": m not in EXTRA_METHOD_FAMILIES,
                  "budget_overshoot": _budget_overshoot(entry), **entry}
    over = {m: e["budget_overshoot"] for m, e in out.items()
            if e["budget_overshoot"] > OVERSHOOT_TOL}
    if over:
        log.warning("recovery_mean: %d method(s) acquired more genes than the "
                    "budget allows; the curve page plots them at what they "
                    "assayed and names them: %s", len(over),
                    ", ".join(f"{m} ({r:.2f}x)" for m, r in sorted(over.items())))
    log.info("recovery_mean: %d methods x %d steps", len(out), len(steps))
    return {"schema": 3, "overshoot_tolerance": OVERSHOOT_TOL,
            "steps": sorted(steps), "methods": out}


#: A method may acquire slightly more genes than it asked for -- a batch that
#: comes back with duplicates resolved, say -- but not meaningfully more. Past
#: this ratio the overrun is a fact about the method worth stating on the page,
#: not rounding error.
OVERSHOOT_TOL = 1.05


def _budget_overshoot(entry: dict) -> float:
    """Worst ratio of genes actually acquired to genes the budget allowed.

    Nothing currently overshoots, and the page's note is hidden accordingly.
    The check stays because the last thing that tripped it was a real bug and
    not a property of any method: the finetuned-LLM curves were being built from
    what those runs *submitted* rather than what their harness assayed, which
    ran the base model to 1.29x budget and the SFT row to 2.01x. If a future
    export reintroduces something like that, the page should say so on its face
    rather than quietly drawing one method on a longer x axis than the rest.
    """
    return max((a / b for a, b in zip(entry["n_acquired"], entry["budget"])
                if b), default=0.0)


def build_random_reference(mean: dict, by_screen: dict) -> dict:
    """What picking genes at random would recover: the chance line.

    Every plotted method spends the same budget out of the same |U|-gene
    candidate pool, which makes this exact rather than simulated. A uniform
    draw of ``n`` genes from ``U`` contains any given gene with probability
    ``n/|U|``, and a screen's hits are a subset of its library, so for screen
    ``s``::

        E[fraction of hits recovered] = n / |U|
        E[hits found]                 = n |H_s| / |U|
        E[picks outside the library]  = n (|U| - |L_s|) / |U|

    This is the EF = 1 line, and not by coincidence. EF forgives picks that are
    real genes outside the screen's library, so the uniform policy's effective
    budget is only the ``n |L_s| / |U|`` picks that landed inside it, and
    ``(n |H_s| / |U|) / (n |L_s| / |U| · |H_s| / |L_s|) = 1`` exactly. A curve
    above this line is beating chance in the same sense the results table's EF
    column means, which is why the page can draw one line and let it carry both
    readings.

    Note the fraction line is *not* ``n / |L_s|``. A policy drawing from one
    screen's library alone recovers slightly more hits per round, because it
    never spends a pick outside -- and scores EF 1 as well. Both are chance;
    this is the one facing the same pool as the methods drawn beside it.
    """
    budgets = {tuple(e["budget"]) for e in mean["methods"].values()}
    if len(budgets) != 1:
        raise MissingInput(
            "the recovery curves disagree about the per-round budget: "
            f"{sorted(budgets)}. The chance line is drawn against one budget "
            "grid, and a single line cannot be right for two of them."
        )
    grid = [float(n) for n in budgets.pop()]
    totals = by_screen["screen_totals"]
    if not totals:
        raise MissingInput(
            "recovery_by_screen has no screen_totals, so |U|, |L| and |H| are "
            "unknown and the chance line cannot be computed. Re-run "
            "export_recovery_curves.py."
        )

    def frac(t: dict, n: float) -> float:
        return n / t["universe_size"]

    def hits(t: dict, n: float) -> float:
        return n * t["total_hits"] / t["universe_size"]

    def outside(t: dict, n: float) -> float:
        return n * (t["universe_size"] - t["library_size"]) / t["universe_size"]

    def over_screens(term) -> list[float]:
        return [sum(term(t, n) for t in totals.values()) / len(totals)
                for n in grid]

    log.info("random reference: %.1f%% of hits at %d genes assayed",
             100 * over_screens(frac)[-1], int(grid[-1]))
    return {
        "n_acquired": grid,
        "frac_hits": over_screens(frac),
        "cum_hits": over_screens(hits),
        "in_universe_not_lib": over_screens(outside),
        # Per screen too: the explorer's one-screen view compares against that
        # screen's own chance line, which differs from the mean's wherever the
        # screen's library or hit count does.
        "by_screen": {
            s: {"n_acquired": grid,
                "frac_hits": [frac(t, n) for n in grid],
                "cum_hits": [hits(t, n) for n in grid]}
            for s, t in totals.items()
        },
    }


def build_recovery_by_screen(path: Path, results: dict) -> dict:
    rows = _read_csv(path)
    fams = _family_lookup(results)

    # method -> screen -> {column: [per step]}
    curves: dict[str, dict[str, dict]] = {}
    screens: list[str] = []
    for row in rows:
        m, s = row["method"], row["screen"]
        if s not in screens:
            screens.append(s)
        entry = curves.setdefault(m, {}).setdefault(
            s, {k: [] for k in BY_SCREEN_COLUMNS})
        for out_col, csv_col in BY_SCREEN_COLUMNS.items():
            entry[out_col].append(_num(row, csv_col, path))

    unknown = sorted(m for m in curves if m not in fams)
    if unknown:
        raise MissingInput(f"{path} has methods with no family: {unknown}")

    # Per-screen totals, so the page can show "38 of 169 hits" not just a
    # fraction. Constant per screen, so read them off any row.
    totals = {}
    for row in rows:
        totals.setdefault(row["screen"], {
            "total_hits": int(row["total_hits"]),
            "library_size": int(row["library_size"]),
            "universe_size": int(row["universe_size"]),
        })

    log.info("recovery_by_screen: %d methods x %d screens",
             len(curves), len(screens))
    return {"schema": 2, "screens": screens, "screen_totals": totals,
            "methods": curves}


# --------------------------------------------------------------- summary

def build_summary(results: dict, recovery: dict) -> dict:
    """Headline numbers for the landing page's stat grid.

    Derived, never typed in by hand: if a rerun moves the best EF, the site
    moves with it instead of quoting a stale figure next to a fresh table.
    """
    scored = [r for r in results["rows"] if r["ef"] is not None]
    best = max(scored, key=lambda r: r["ef"])
    best_fh = max(scored, key=lambda r: r["frac"])
    steps = recovery["steps"]
    return {
        "schema": 1,
        # The stat grid's "methods compared" is the count that actually has a
        # number, not the row count: a row nobody could evaluate was not
        # compared to anything.
        "n_methods": len(scored),
        "n_methods_in_table": len(results["rows"]),
        "n_methods_unavailable": len(results["unavailable"]["names"]),
        "n_methods_with_curves": len(recovery["methods"]),
        "n_test_screens": results["n_screens"],
        "universe_size": results["universe_size"],
        "budget": results["budget"],
        "n_rounds": len(steps),
        "batch_size": results["budget"] // len(steps) if steps else None,
        "best_ef": {"value": best["ef"], "method": best["name"],
                    "family": best["family"]},
        "best_frac_hits": {"value": best_fh["frac"], "method": best_fh["name"],
                           "family": best_fh["family"]},
    }


# ---------------------------------------------------- interactive figures

# Four of the paper's figures are drawn from data files that the analysis
# scripts already leave on disk. Where that is true the site plots the data
# rather than shipping a picture of it: a sunburst you can click into beats a
# PDF of a sunburst, and a heatmap with a real hover beats squinting at a
# 6 pt row label. The static PDFs stay in assets/figures/ and stay linked, so
# nothing is lost -- but the page leads with the interactive version.
#
# Everything below reads only from --analysis-dir, and every one of them is
# required: a missing input is a build failure, not a page that quietly falls
# back to the static image.


#: Node id of the sunburst's synthetic root. Anything the Reactome vocabulary
#: cannot also be; the dunder is there to make a collision obvious if it ever
#: happens rather than to be pretty.
ROOT = "__all__"


def _tint(hex_color: str, amount: float) -> str:
    """Blend toward white; amount 0 is the hue itself, 1 is white."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(round(c + (255 - c) * amount) for c in (r, g, b))
    return "#%02x%02x%02x" % mix


def build_sunburst(path: Path, script: Path) -> dict:
    """Reactome coverage per method, as the paper's Figure 6 draws it.

    The cache holds all 29 Reactome roots and all 210 level-2 groups, which is
    not what the figure shows and not what is readable: the tail is a comb of
    slivers. The figure keeps the ``N_CATS`` largest roots by weight *pooled
    over every method* -- so a hue means the same category in every panel --
    folds the rest into "Other", and inside each category draws only the
    ``N_SUB`` largest groups that clear ``MIN_SUB`` of the panel, pooling that
    tail too. Those constants and the palette are read out of the figure's own
    script so the site cannot drift from the paper.

    Values are millionths of the panel as **integers**, and each category's
    value is the integer sum of the children emitted under it. Integers are the
    correctness requirement, not tidiness: Plotly's ``branchvalues: "total"``
    renders nothing at all -- silently, no error, an empty box -- if any parent
    comes out below its children, and in floating point ``round(a, 6) +
    round(b, 6)`` routinely lands an ulp above ``round(a + b, 6)``. Plotly only
    ever uses these as ratios, so the unit does not reach the reader.
    """
    scale = 1_000_000
    lit = _literal_from_script(script, ("METHODS", "N_CATS", "N_SUB",
                                        "MIN_SUB", "HUES", "OTHER"))
    n_cats, n_sub, min_sub = lit["N_CATS"], lit["N_SUB"], lit["MIN_SUB"]
    hues, other_hue = lit["HUES"], lit["OTHER"]

    raw = json.loads(path.read_text())
    # The figure's panel order is an argument -- random, then the classical
    # baselines, then the systems -- so carry it rather than whatever order the
    # cache happens to serialise in.
    order = [name for name, _ in lit["METHODS"] if name in raw]
    order += [name for name in raw if name not in order]
    for name, d in raw.items():
        for key in ("by_cat", "by_sub", "sub_cat", "eff_pathways"):
            if key not in d:
                raise MissingInput(
                    f"{path}: method {name!r} has no {key!r}. Re-run "
                    "`python -m assayloop.scripts.plot_pathway_sunburst`.")

    # Shared category order: rank by each method's *normalised* share so a
    # method with more annotated picks does not get a bigger vote.
    pooled: dict[str, float] = {}
    for d in raw.values():
        tot = sum(d["by_cat"].values())
        for cat, w in d["by_cat"].items():
            pooled[cat] = pooled.get(cat, 0.0) + w / tot
    kept = [c for c, _ in sorted(pooled.items(), key=lambda kv: -kv[1])][:n_cats]
    colour_of = dict(zip(kept, hues))
    colour_of["Other"] = other_hue
    cats = [*kept, "Other"]
    keptset = set(kept)

    methods = {}
    for name in order:
        d = raw[name]
        total = sum(d["by_cat"].values())
        # A synthetic root under everything. Two jobs: Plotly draws the root as
        # a disc at the centre, which is the hole the paper's figure puts the
        # EP-D number in -- a sunburst with several roots has no hole at all,
        # it is a pie -- and it gives the drill-down somewhere to zoom back out
        # to. Its value is filled in below, once the categories are known.
        ids, labels, parents, values, colours = [ROOT], [name], [""], [0], ["#ffffff"]
        for cat in cats:
            # Children first: the parent's value is whatever they sum to.
            if cat == "Other":
                sel = [(s, w) for s, w in d["by_sub"].items()
                       if d["sub_cat"].get(s, "Other") not in keptset]
            else:
                sel = [(s, w) for s, w in d["by_sub"].items()
                       if d["sub_cat"].get(s) == cat]
            sel.sort(key=lambda sw: -sw[1])
            shares = [(s, w / total) for s, w in sel]
            n_head = min(n_sub, sum(1 for _, w in shares if w >= min_sub))
            head, tail = shares[:n_head], shares[n_head:]

            kids = [(s, round(w * scale),
                     _tint(colour_of[cat], 0.24 if i % 2 else 0.44))
                    for i, (s, w) in enumerate(head)]
            if tail:
                kids.append((f"{len(tail)} smaller groups",
                             round(sum(w for _, w in tail) * scale),
                             _tint(colour_of[cat], 0.62)))
            kids = [k for k in kids if k[1] > 0]
            if not kids:
                continue

            ids.append(cat)
            labels.append(cat)
            parents.append(ROOT)
            values.append(sum(w for _, w, _ in kids))
            colours.append(colour_of[cat])
            for sub, w, colour in kids:
                # Plotly keys nodes by id, so a group name that repeats across
                # categories (the pooled tail does, every time) still resolves.
                ids.append(f"{cat}/{sub}")
                labels.append(sub)
                parents.append(cat)
                values.append(w)
                colours.append(colour)

        values[0] = sum(v for p, v in zip(parents, values) if p == ROOT)
        methods[name] = {
            "ids": ids, "labels": labels, "parents": parents,
            "values": values, "colors": colours,
            "eff_pathways": d["eff_pathways"],
            "ep_batch": d.get("ep_batch"), "ep_screen": d.get("ep_screen"),
            "n_picks": d.get("n_picks"), "n_annotated": d.get("n_annotated"),
        }
    log.info("sunburst: %d methods, %d categories, %d nodes each (max)",
             len(methods), len(cats), max(len(m["ids"]) for m in methods.values()))
    return {"schema": 1, "scale": scale, "root": ROOT, "order": order,
            "categories": cats, "colors": colour_of, "methods": methods}


def build_pathway_heatmap(path: Path) -> dict:
    """Per-method share of each Reactome root, as a log2 ratio to Random.

    The paper's heatmap is over- and under-representation, so the plotted
    quantity is log2(share / share under Random) -- 0 is "same as chance",
    +1 is twice as much of that biology.
    """
    raw = json.loads(path.read_text())
    if "Random" not in raw:
        raise SystemExit(
            f"{path}: no 'Random' row, so there is nothing to normalise "
            "against. Re-run `python -m assayloop.scripts."
            "plot_llm_pathway_heatmap`.")
    base = raw["Random"]["share"]
    methods = [m for m in raw if m != "Random"]
    # Order the columns by how much of it Random sees, so the dense biology is
    # on the left and the heatmap reads left-to-right by prevalence.
    cats = sorted(base, key=lambda c: -base[c])
    z, hover = [], []
    for m in methods:
        share = raw[m]["share"]
        row, hrow = [], []
        for c in cats:
            s, b = share.get(c, 0.0), base[c]
            if not b:
                raise SystemExit(
                    f"{path}: Random has zero share of {c!r}, so the ratio for "
                    "every method is undefined. Drop the category upstream.")
            # A method with no picks in a category is genuinely absent, not
            # missing data: floor the ratio rather than emitting -inf, and say
            # so in the hover.
            row.append(round(math.log2(s / b), 4) if s > 0 else None)
            hrow.append(f"{c}<br>{s * 100:.1f}% of picks "
                        f"(random: {b * 100:.1f}%)")
        z.append(row)
        hover.append(hrow)
    return {"schema": 1, "categories": cats, "methods": methods,
            "z": z, "hover": hover,
            "eff_pathways": {m: raw[m]["eff_pathways"] for m in methods},
            "unique_genes": {m: raw[m]["n_unique"] for m in methods},
            "annotated_fraction": {
                m: raw[m]["n_annotated"] / max(raw[m]["n_picks"], 1)
                for m in methods
            }}


def build_lopo(path: Path) -> dict:
    """Leave-one-phenotype-out EF per fold, per method."""
    rows = _read_csv(path)
    series = {"knn": "kNN baseline", "sup": "AssayFormer (supervised)",
              "rl_best": "AssayFormer + GRPO"}
    for r in rows:
        for col in ("label", "n_test", *series):
            if col not in r:
                raise SystemExit(
                    f"{path}: no {col!r} column. Re-run "
                    "`python -m assayloop.scripts.plot_lopo_results`.")
    return {
        "schema": 1,
        "folds": [r["label"] for r in rows],
        "n_test": [int(r["n_test"]) for r in rows],
        "series": [{"key": k, "label": lab,
                    "values": [_num(r, k, path) for r in rows]}
                   for k, lab in series.items()],
    }


def build_gene_matrices(path: Path) -> dict:
    """How well each embedding recovers each gene-relationship database.

    One record per (initialisation, training stage) with an AUROC and an
    average precision against STRING, CORUM, SIGNOR and Reactome.
    """
    rows = json.loads(path.read_text())
    dbs = ["STRING", "CORUM", "SIGNOR", "REACTOME"]
    out = []
    for r in rows:
        for db in dbs:
            for suffix in ("auroc", "ap", "n_pos"):
                if f"{db}_{suffix}" not in r:
                    raise SystemExit(
                        f"{path}: record {r.get('init')}/{r.get('stage')} has "
                        f"no {db}_{suffix}. Re-run "
                        "`python -m assayloop.scripts.analyze_gene_matrices`.")
        out.append({
            "init": r["init"], "stage": r["stage"],
            "auroc": {db: r[f"{db}_auroc"] for db in dbs},
            "ap": {db: r[f"{db}_ap"] for db in dbs},
        })
    # Preserve first-seen order for both axes: the scripts emit them in the
    # paper's order, and sorting alphabetically would scramble the training
    # stages (Raw emb. -> SFT -> RL) into nonsense.
    inits, stages = [], []
    for r in out:
        if r["init"] not in inits:
            inits.append(r["init"])
        if r["stage"] not in stages:
            stages.append(r["stage"])
    return {"schema": 1, "databases": dbs, "inits": inits, "stages": stages,
            "n_pos": {db: rows[0][f"{db}_n_pos"] for db in dbs},
            "records": out}


# Each embedding initialisation, as `gene_matrix_analysis.json` names it, and
# the two rows of the results table that carry its downstream score. The figure
# script keeps this join as two hardcoded dicts of EF values; reading it from
# the table instead means re-running the sweeps updates the site.
INIT_ROWS = {
    "Random":    ("AssayFormer (Random)",      "Random + GRPO"),
    "BPMF":      ("AssayFormer (BPMF)",        "BPMF + GRPO"),
    "MF":        ("AssayFormer (MF)",          "MF + GRPO"),
    "MF-Sphere": ("AssayFormer (MF-Sphere)",   "MF-Sphere + GRPO"),
    "SVD":       ("AssayFormer (SVD)",         "SVD + GRPO"),
    "GenePT":    ("AssayFormer (GenePT-PCA)",  "GenePT-PCA + GRPO"),
    "K562":      ("AssayFormer (K562-PCA)",    "K562-PCA + GRPO"),
}


def build_init_story(path: Path, results: dict) -> dict:
    """Textbook-biology recovery against downstream task performance.

    x is a property of the raw embedding -- the mean AUROC over STRING, CORUM,
    SIGNOR and Reactome -- so it is fixed across training stages; y is the
    enrichment factor AssayFormer reaches from that initialisation, supervised
    and after GRPO. The point of the figure is that the two are anticorrelated.
    """
    rows = json.loads(path.read_text())
    dbs = ["STRING", "CORUM", "SIGNOR", "REACTOME"]
    by_name = {r["name"]: r for r in results["rows"]}

    points = []
    for init, (sup_row, rl_row) in INIT_ROWS.items():
        raw = [r for r in rows if r["init"] == init and r["stage"] == "Raw emb."]
        if not raw:
            raise MissingInput(
                f"{path}: no 'Raw emb.' record for init {init!r}, so it has no "
                "x coordinate. Re-run `python -m "
                "assayloop.scripts.analyze_gene_matrices`.")
        for name in (sup_row, rl_row):
            if name not in by_name:
                raise MissingInput(
                    f"the results table has no row named {name!r}, which is "
                    f"where the {init!r} initialisation's downstream score "
                    "comes from. Re-run full_genome_table.py with the "
                    "ablation rows included.")
        points.append({
            "init": init,
            "auroc": round(sum(raw[0][f"{db}_auroc"] for db in dbs) / len(dbs), 5),
            "per_db": {db: raw[0][f"{db}_auroc"] for db in dbs},
            "supervised": by_name[sup_row]["ef"],
            "rl": by_name[rl_row]["ef"],
        })
    return {"schema": 1, "databases": dbs, "points": points}


def _literal_from_script(path: Path, names: Sequence[str]) -> dict:
    """Read module-level literal assignments out of a script without running it.

    The figure scripts are top-level scripts: importing one renders its figure.
    Some of what the site needs -- the label-ablation numbers, the sunburst's
    palette and cutoffs -- lives in literals those scripts define on the way, so
    parse the source and evaluate just those assignments. Slower to write than
    an import and much better behaved: no matplotlib, no file written, and no
    chance of the figure's side effects landing in whatever `$ASSAYLOOP_OUTPUT`
    happens to point at when the site is built.
    """
    try:
        tree = ast.parse(path.read_text())
    except OSError as err:
        raise MissingInput(
            f"cannot read {path}, which is where the site reads "
            f"{', '.join(names)} from.\n  {err}") from err

    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)

    absent = [n for n in names if n not in found]
    if absent:
        raise MissingInput(
            f"{path.name} no longer defines {', '.join(absent)} as a "
            "module-level literal, so the site cannot read it out of the "
            "script that draws the paper's version.\n"
            "  Either restore the literal or give the script a --json dump and "
            "read that instead.")
    return found


def build_label_ablation(path: Path) -> dict:
    """With vs. without hit feedback, read out of the figure's own script.

    Unlike everything else in this section there is no analysis file: the
    numbers live in a module-level ``DATA`` dict in `plot_label_ablation.py`,
    cached from the sweeps the baselines table draws on. Reading them from
    there is what keeps the site and the paper's figure from drifting -- a copy
    here would be a second place to update and a first place to forget.
    """
    lit = _literal_from_script(path, ("DATA", "SEPARATE"))
    data, separate = lit["DATA"], set(lit["SEPARATE"])

    rows = []
    for name, panels in data.items():
        ef = panels.get("EF")
        if ef is None:
            raise MissingInput(
                f"plot_label_ablation.DATA[{name!r}] has no EF entry, so there "
                "is no dumbbell to draw for it.")
        with_fb, without_fb, sterr = ef
        nauc = panels.get("nAUC")
        rows.append({
            "name": name,
            # The two arms are blinded by different mechanisms and the figure
            # rules them apart; the site needs the same split.
            "group": "no readout at all" if name in separate
                     else "hit labels stripped from the prompt",
            "with_feedback": with_fb,
            "without_feedback": without_fb,
            "gap": round(with_fb - without_fb, 4),
            # Paired standard error over the 20 shared test screens. Not drawn
            # in the paper's figure (the whiskers overlap the dots); on the web
            # it goes in the hover, where it costs nothing.
            "gap_stderr": sterr,
            "nauc": None if nauc is None
                    else {"with_feedback": nauc[0], "without_feedback": nauc[1],
                          "gap_stderr": nauc[2]},
        })
    return {"schema": 1, "metric": "Enrichment factor (EF)", "rows": rows}


def build_influence(path: Path) -> dict:
    """Context-conditional influence of each probe gene on each target.

    The paper's figure draws only the fifteen pairs it discusses, as two bar
    panels. The run computed the whole probe x target grid, so the site draws
    the grid and marks the pairs the paper picked out -- the same numbers, with
    the rest of the matrix visible around them.
    """
    raw = json.loads(path.read_text())
    inf = raw.get("influence")
    if not inf:
        raise SystemExit(
            f"{path}: no 'influence' block. Re-run `python -m "
            "assayloop.scripts.paper_influence_figure` (needs an "
            "OPENAI_API_KEY to recompute from scratch).")
    probes = sorted(inf)
    # Every probe was scored against the same target list; if that ever stops
    # being true the matrix has holes and a heatmap would hide them.
    targets = sorted(inf[probes[0]])
    for p in probes:
        if sorted(inf[p]) != targets:
            raise SystemExit(
                f"{path}: probe {p!r} was scored against a different target "
                "set than {probes[0]!r}, so the influence matrix is ragged.")

    featured = raw.get("featured_pairs", {})
    # (probe, target) -> the paper's one-phrase reading of that pair. Newlines
    # in the source are line breaks for the matplotlib label; the site wraps.
    notes = {}
    for direction, pairs in featured.items():
        for probe, target, ann in pairs:
            notes[f"{probe}\t{target}"] = {
                "direction": direction, "note": " ".join(ann.split()),
            }

    z = [[round(inf[p][t], 4) for t in targets] for p in probes]
    return {"schema": 1, "probes": probes, "targets": targets, "z": z,
            "featured": notes}


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-json", type=Path,
                    default=REPO / "output" / "tables"
                    / "full_genome_baselines_f2.json",
                    help="Output of full_genome_table.py --json")
    ap.add_argument("--analysis-dir", type=Path,
                    default=REPO / "output" / "analysis",
                    help="Where export_recovery_curves.py wrote its CSVs")
    ap.add_argument("--out-dir", type=Path, default=DATA_OUT,
                    help="Where to write the site's JSON")
    ap.add_argument("--allow-unavailable", nargs="+", metavar="NAME", default=[],
                    help="Method names (the table JSON's `name` field) that "
                         "have no result and should be published as "
                         "explicitly not evaluated. Requires "
                         "--unavailable-note. Every unevaluated row must be "
                         "named; anything left out is still an error.")
    ap.add_argument("--unavailable-note", default=None,
                    help="One sentence shown on results.html explaining why "
                         "the --allow-unavailable rows have no numbers.")
    args = ap.parse_args(argv)

    table_json = _require(
        args.table_json, "Results table JSON",
        "python -m assayloop.scripts.full_genome_table --min-screen-freq 2 "
        f"--json {args.table_json}")
    mean_csv = _require(
        args.analysis_dir / "recovery_curves_mean.csv", "Mean recovery curves",
        "python -m assayloop.scripts.export_recovery_curves")
    screen_csv = _require(
        args.analysis_dir / "recovery_curves_by_screen.csv",
        "Per-screen recovery curves",
        "python -m assayloop.scripts.export_recovery_curves")

    # Inputs for the interactive versions of the paper's figures. Each names
    # the script that writes it, so a missing one is a one-line fix rather
    # than a hunt.
    figure_inputs = {
        "pathway_heatmap.json": (
            build_pathway_heatmap, "llm_pathway_heatmap_data.json",
            "LLM pathway heatmap data",
            "python -m assayloop.scripts.plot_llm_pathway_heatmap"),
        "lopo.json": (
            build_lopo, "lopo_summary.csv",
            "Leave-one-phenotype-out summary",
            "python -m assayloop.scripts.plot_lopo_results"),
        "gene_matrices.json": (
            build_gene_matrices, "gene_matrix_analysis.json",
            "Gene-relationship recovery",
            "python -m assayloop.scripts.analyze_gene_matrices"),
        "influence.json": (
            build_influence, "paper_influence_data.json",
            "Context-conditional influence",
            "python -m assayloop.scripts.paper_influence_figure"),
    }

    results = build_results(table_json, args.allow_unavailable,
                            args.unavailable_note)
    recovery_mean = build_recovery_mean(mean_csv, results)
    recovery_by_screen = build_recovery_by_screen(screen_csv, results)
    # The chance line is a function of both: the budget grid comes from the
    # curves, |U| / |L| / |H| from the per-screen totals. It is split across the
    # two files the way everything else is -- means here, per screen there -- so
    # the landing page, which fetches only the mean, still gets one.
    reference = build_random_reference(recovery_mean, recovery_by_screen)
    recovery_by_screen["random_reference"] = reference.pop("by_screen")
    recovery_mean["random_reference"] = reference
    outputs = {
        "results.json": results,
        "diversity.json": build_diversity(results),
        "recovery_mean.json": recovery_mean,
        "recovery_by_screen.json": recovery_by_screen,
        "summary.json": build_summary(results, recovery_mean),
    }
    for name, (fn, src, what, how) in figure_inputs.items():
        outputs[name] = fn(_require(args.analysis_dir / src, what, how))
    # Three that do not fit the (builder, one input file) shape above: the init
    # story joins an analysis file to the results table, the label ablation's
    # numbers live in its own script rather than in a file at all, and the
    # sunburst needs both its cache and the cutoffs the paper's figure applies
    # to it.
    scripts = DOCS.parent / "src" / "assayloop" / "scripts"
    in_repo = "it ships in this repo; run from a checkout, not a wheel"
    outputs["init_story.json"] = build_init_story(
        _require(args.analysis_dir / "gene_matrix_analysis.json",
                 "Gene-relationship recovery",
                 "python -m assayloop.scripts.analyze_gene_matrices"),
        results)
    outputs["label_ablation.json"] = build_label_ablation(
        _require(scripts / "plot_label_ablation.py",
                 "Label-ablation figure script", in_repo))
    outputs["sunburst.json"] = build_sunburst(
        _require(args.analysis_dir / "pathway_sunburst_data.json",
                 "Pathway sunburst data",
                 "python -m assayloop.scripts.plot_pathway_sunburst"),
        _require(scripts / "plot_pathway_sunburst.py",
                 "Pathway sunburst figure script", in_repo))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        p = args.out_dir / name
        # Compact separators: these are fetched over the wire, not read by hand.
        p.write_text(_rename_keys(json.dumps(payload, separators=(",", ":"))))
        log.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
