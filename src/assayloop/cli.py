"""assayloop command-line interface.

Running the loop:

- ``run``                  : one (model, acq) pair over a screen set.
- ``sweep``                : a grid over models x acquisitions x seeds.
- ``eval-ranker``          : evaluate a trained AssayFormer checkpoint.
- ``eval-ranker-handoff``  : LLM warm start handed off to AssayFormer.
- ``score-existing``       : score predictions from a JSONL file.

Training:

- ``train-bpmf``           : BPMF factorisation (the gene-embedding init).
- ``train-ranker``         : supervised AssayFormer training.
- ``train-ranker-rl``      : GRPO fine-tuning of a trained AssayFormer.
- ``collect-dataset``      : teacher rollouts for AssayLLM SFT.
- ``warm-text-cache``      : pre-embed screen descriptions.

Inspecting and reproducing:

- ``figure``               : regenerate a paper figure or table.
- ``select-public-screens``: rebuild the public screen split.
- ``trace``                : replay a persisted run.
- ``info``                 : print resolved config (paths, LLM, etc.).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from . import config

app = typer.Typer(help="assayloop: modular active-learning loops for screen prediction.")
console = Console()

logging.basicConfig(
    level=os.getenv("ASSAYLOOP_LOG", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _csv(x: Optional[str]) -> list[str]:
    if not x:
        return []
    return [s.strip() for s in x.split(",") if s.strip()]


def _parse_kv_params(params: Optional[list[str]]) -> dict[str, Any]:
    """Parse a list of 'key=value' strings into a dict with auto-typed values."""
    if not params:
        return {}
    out: dict[str, Any] = {}
    for item in params:
        if "=" not in item:
            raise typer.BadParameter(f"Expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if v.lower() in ("true", "yes"):
            out[k] = True
        elif v.lower() in ("false", "no"):
            out[k] = False
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


# Acquisitions that pick policy-driven batches from step 1 onward
# without needing labelled examples to bootstrap. Warm-start is pure
# wasted budget for these (random samples replace would-be informed
# picks) — we warn the user but still honour their request.
_ZERO_SHOT_ACQS = {
    "llm_single",
    "llm_single_blind",
}


def _warn_if_warm_start_wasted(warm_start: int, acqs: list[str]) -> None:
    if warm_start <= 0:
        return
    zero_shot = [a for a in acqs if a.lower() in _ZERO_SHOT_ACQS]
    if not zero_shot:
        return
    console.print(
        f"[yellow]warning:[/yellow] --warm-start={warm_start} is set, but "
        f"the acquisition{'s' if len(zero_shot) > 1 else ''} "
        f"{', '.join(zero_shot)} {'are' if len(zero_shot) > 1 else 'is'} "
        f"zero-shot. The first {warm_start} picks will be RANDOM "
        f"ground-truth reveals instead of policy-driven ones, which "
        f"strictly hurts the cumulative-hits curve for the LLM. "
        f"Consider --warm-start 0 unless you're comparing against a "
        f"classical model (knn/rf) in the same sweep that needs "
        f"labelled examples to bootstrap."
    )


# Acquisitions whose ``make_acquisition`` accepts an ``llm``
# LLMClientConfig override.
_LLM_CLIENT_ACQS = {
    "llm_single",
    "llm_single_blind",
    "llm",
}


def _llm_with_empty_policy(client, *, retry_on_empty: bool):
    """Return ``client`` with its empty-answer retry policy forced.

    The same ``--lm-config`` feeds both ``run`` (where the model is allowed to
    return an empty answer) and ``collect-dataset`` (where every SFT batch must
    carry at least one gene). We make that distinction *command-scoped* rather
    than config-scoped by overriding ``retry_on_empty`` here. Transient-error
    retries (``max_retries``) are untouched, so the "retry timeouts forever"
    behaviour still applies to both.
    """
    from dataclasses import replace

    if client is None:
        return None
    try:
        return replace(client, retry_on_empty=retry_on_empty)
    except TypeError:
        # Not a dataclass we can rewrite — leave it as-is.
        return client


def _load_lm_config_or_exit(lm_config: Optional[str]):
    """Load an assaybench collect-*.yaml into an AssayBenchLM (or exit)."""
    if not lm_config:
        return None
    from .llm.assaybench_config import load_lm_config

    try:
        loaded = load_lm_config(lm_config)
    except Exception as e:
        console.print(
            f"[red]error:[/red] failed to load --lm-config {lm_config!r}: "
            f"{type(e).__name__}: {e}"
        )
        raise typer.Exit(1)
    c = loaded.client.resolved()
    console.print(
        f"[green]lm-config[/green] {loaded.path.name} "
        f"(model_name={loaded.model_name}): provider={c.provider} "
        f"model={c.model} base_url={c.base_url} max_tokens={c.max_tokens} "
        f"temp={c.temperature} top_p={c.top_p} top_k={c.top_k} "
        f"min_p={c.min_p} presence_penalty={c.presence_penalty} "
        f"repetition_penalty={c.repetition_penalty} "
        f"thinking={'off' if c.disable_thinking else 'on'}"
    )
    return loaded


# ---------------------------------------------------------------------------
# run / sweep
# ---------------------------------------------------------------------------


@app.command()
def run(
    model: str = typer.Option("knn", help="See make_model() in experiment/runner.py for the full list."),
    acq: str = typer.Option("greedy", help="random|greedy|ucb|bio_ucb|llm_single (open-vocab, AssayBench-style)|llm_single_blind (ablation: history shown without hit labels). The LLM->AssayFormer handoff is its own command, eval-ranker-handoff."),
    screen_set: str = typer.Option("public", help="public|public_train|public_val|public_test|/path/to.yaml"),
    screen: Optional[str] = typer.Option(None, help="Comma-sep dataset_names to run (overrides screen_set)"),
    batch_size: int = typer.Option(100),
    n_steps: Optional[int] = typer.Option(None, help="Cap on AL steps"),
    seed: int = typer.Option(0),
    warm_start: int = typer.Option(
        0,
        help=(
            "# of randomly revealed genes before step 1. Useful for "
            "bootstrapping classical models (knn/rf) that can't predict "
            "without labels. Set to 0 for zero-shot LLM acquisitions — "
            "warm-start strictly hurts their cumulative-hits curve by "
            "replacing policy picks with random samples."
        ),
    ),
    metrics: str = typer.Option("hits_auc,andcg,batch_hits", help="Comma-sep metric names"),
    tag: str = typer.Option(""),
    lm_config: Optional[str] = typer.Option(
        None,
        "--lm-config",
        help="Path to an AssayBench collect-*.yaml (see configs/lm/). Its 'lm' block "
        "(model, api_base, sampling preset, thinking) configures the LLM "
        "acquisition. Only applies to llm_single/llm_single_blind. "
        "If --tag is empty, the config's model_name is used as the tag.",
    ),
    parallel: int = typer.Option(1, "--parallel", "-j", help="# of screens to run concurrently (thread pool). 1=sequential."),
    max_shortfall_frac: float = typer.Option(1.0, "--max-shortfall-frac", help="Abort a screen if the running acquisition shortfall fraction (requested - actually acquired) exceeds this. Set to 1.0 (or higher) to disable."),
    system_prompt_file: Optional[str] = typer.Option(
        None,
        "--system-prompt-file",
        help="Path to a .txt holding the system prompt to inject into the "
        "llm_single/llm_single_blind acquisition. Overrides the "
        "acquisition's default prompt.",
    ),
    model_param: Optional[list[str]] = typer.Option(None, "--model-param", help="Model override as key=value (repeatable)"),
    no_persist: bool = typer.Option(False, "--no-persist"),
    sweep_id: Optional[str] = typer.Option(None, "--sweep-id", help="Fixed sweep ID (required for --resume to match prior run directories)."),
    resume: bool = typer.Option(False, "--resume", help="Skip screens whose result.json already exists from a prior run."),
    full_genome: bool = typer.Option(False, "--full-genome", help="Expand candidate pool to the union of genes across all screens in the set, filtered by --min-screen-freq (fair comparison with open-vocab LLMs)."),
    min_screen_freq: int = typer.Option(2, "--min-screen-freq", help="With --full-genome, keep only genes measured in at least this many screens. The default 2 is the paper's f2 universe (21,147 genes for --screen-set public); 0 keeps the plain union (22,174), which admits pseudogenes and per-library assembly artefacts."),
    verbose: bool = typer.Option(True),
):
    """Run a single (model, acq) configuration across the resolved screens."""
    from .experiment.runner import RunConfig, run_sweep

    _warn_if_warm_start_wasted(warm_start, [acq])

    acq_overrides: dict = {}
    loaded_lm = _load_lm_config_or_exit(lm_config)
    if loaded_lm is not None:
        if acq.lower() in _LLM_CLIENT_ACQS:
            # A `run` lets the model emit an empty answer if it wants to.
            acq_overrides["llm"] = _llm_with_empty_policy(
                loaded_lm.client, retry_on_empty=False
            )
        else:
            console.print(
                f"[yellow]warning:[/yellow] --lm-config is set but acq "
                f"{acq!r} does not take an LLM client; ignoring it."
            )
        if not tag and loaded_lm.model_name:
            tag = loaded_lm.model_name

    if system_prompt_file:
        if acq.lower() in _LLM_CLIENT_ACQS:
            try:
                acq_overrides["system_prompt"] = Path(
                    system_prompt_file
                ).read_text(encoding="utf-8")
            except OSError as e:
                console.print(
                    f"[red]error:[/red] could not read --system-prompt-file "
                    f"{system_prompt_file!r}: {e}"
                )
                raise typer.Exit(1)
        else:
            console.print(
                f"[yellow]warning:[/yellow] --system-prompt-file is set but acq "
                f"{acq!r} does not take a system prompt; ignoring it."
            )

    universe = None
    if full_genome:
        from .tasks import gene_universe, load_screens as _load_screens
        all_screens = _load_screens(target_set=screen_set, dataset_names=_csv(screen) or None)
        universe = gene_universe(all_screens, min_screen_freq=min_screen_freq)
        console.print(
            f"[cyan]full-genome[/cyan]: {len(universe)} genes in candidate "
            f"universe (freq >= {min_screen_freq})"
        )

    cfg = RunConfig(
        screen_set=screen_set,
        dataset_names=_csv(screen) or None,
        model=model,
        acq=acq,
        batch_size=batch_size,
        n_steps=n_steps,
        warm_start_size=warm_start,
        seed=seed,
        model_overrides=_parse_kv_params(model_param),
        acq_overrides=acq_overrides,
        metrics=_csv(metrics) or ["hits_auc", "andcg", "batch_hits", "batch_diversity"],
        tag=tag,
        persist=not no_persist,
        parallel=parallel,
        max_shortfall_frac=(None if max_shortfall_frac >= 1.0 else max_shortfall_frac),
        universe_genes=universe,
    )
    sweep = run_sweep(cfg, sweep_id=sweep_id, verbose=verbose, resume=resume)
    _print_sweep(sweep)


@app.command()
def sweep(
    models: str = typer.Option("null,knn,rf", help="Comma-sep model names"),
    acqs: str = typer.Option("random,greedy,ucb", help="Comma-sep acquisition names"),
    seeds: str = typer.Option("0", help="Comma-sep RNG seeds"),
    screen_set: str = typer.Option("public", help="public|public_train|public_val|public_test|/path/to.yaml"),
    screen: Optional[str] = typer.Option(None),
    batch_size: int = typer.Option(100),
    n_steps: Optional[int] = typer.Option(None),
    warm_start: int = typer.Option(
        0,
        help=(
            "# of randomly revealed genes before step 1. Set to 0 for "
            "zero-shot LLM acquisitions — warm-start strictly hurts "
            "them. Useful only when classical baselines (knn/rf) are in "
            "the grid and need labels to bootstrap."
        ),
    ),
    metrics: str = typer.Option("hits_auc,andcg"),
    tag: str = typer.Option(""),
    lm_config: Optional[str] = typer.Option(
        None,
        "--lm-config",
        help="Path to an AssayBench collect-*.yaml (see configs/lm/). Its 'lm' block "
        "configures every LLM acquisition in the grid "
        "(llm_single/llm_single_blind). If --tag is empty, the "
        "config's model_name is used as the tag.",
    ),
    parallel: int = typer.Option(1, "--parallel", "-j", help="# of screens to run concurrently (thread pool). 1=sequential."),
    max_shortfall_frac: float = typer.Option(0.5, "--max-shortfall-frac", help="Abort a screen if the running acquisition shortfall fraction (requested - actually acquired) exceeds this. Set to 1.0 (or higher) to disable."),
    model_param: Optional[list[str]] = typer.Option(None, "--model-param", help="Model override as key=value (repeatable)"),
    no_persist: bool = typer.Option(False, "--no-persist"),
    verbose: bool = typer.Option(True),
):
    """Run a model x acquisition x seed grid sweep."""
    from .experiment.runner import RunConfig, run_sweep

    sweep_id = f"sweep-{uuid.uuid4().hex[:8]}"
    rows = []
    model_names = _csv(models)
    acq_names = _csv(acqs)
    seed_vals = [int(s) for s in _csv(seeds)] or [0]
    model_overrides = _parse_kv_params(model_param)

    _warn_if_warm_start_wasted(warm_start, acq_names)

    loaded_lm = _load_lm_config_or_exit(lm_config)
    if loaded_lm is not None:
        if not any(a.lower() in _LLM_CLIENT_ACQS for a in acq_names):
            console.print(
                "[yellow]warning:[/yellow] --lm-config is set but no acq in "
                f"{acq_names} takes an LLM client; ignoring it."
            )
        if not tag and loaded_lm.model_name:
            tag = loaded_lm.model_name

    for m, a, s in itertools.product(model_names, acq_names, seed_vals):
        acq_overrides: dict = {}
        if loaded_lm is not None and a.lower() in _LLM_CLIENT_ACQS:
            # A sweep is just batched `run`s — allow empty answers too.
            acq_overrides["llm"] = _llm_with_empty_policy(
                loaded_lm.client, retry_on_empty=False
            )
        cfg = RunConfig(
            screen_set=screen_set,
            dataset_names=_csv(screen) or None,
            model=m,
            acq=a,
            batch_size=batch_size,
            n_steps=n_steps,
            warm_start_size=warm_start,
            seed=s,
            model_overrides=model_overrides,
            acq_overrides=acq_overrides,
            metrics=_csv(metrics) or ["hits_auc", "andcg", "batch_hits", "batch_diversity"],
            tag=tag,
            persist=not no_persist,
            parallel=parallel,
            max_shortfall_frac=(None if max_shortfall_frac >= 1.0 else max_shortfall_frac),
        )
        sweep_res = run_sweep(cfg, sweep_id=f"{sweep_id}-{m}-{a}-s{s}", verbose=verbose)
        rows.append((m, a, s, sweep_res))

    _print_sweep_grid(rows)


def _print_sweep(sweep) -> None:
    table = Table(title=f"Sweep {sweep.sweep_id}")
    table.add_column("Screen")
    table.add_column("Steps", justify="right")
    table.add_column("HitsAUC", justify="right")
    table.add_column("frac_hits", justify="right")
    for s in sweep.per_screen:
        m = s.final_metrics or {}
        table.add_row(
            s.screen_name,
            str(s.n_steps),
            f"{m.get('hits_auc', 0):.4f}",
            f"{m.get('frac_hits', 0):.3f}",
        )
    console.print(table)
    console.print("Aggregate:", sweep.aggregate)


def _print_sweep_grid(rows):
    table = Table(title="Sweep grid")
    table.add_column("Model")
    table.add_column("Acq")
    table.add_column("Seed", justify="right")
    table.add_column("mean_hits_auc", justify="right")
    table.add_column("mean_frac_hits", justify="right")
    table.add_column("n_screens", justify="right")
    for m, a, s, sweep in rows:
        agg = sweep.aggregate
        table.add_row(
            m, a, str(s),
            f"{agg.get('mean_hits_auc', 0):.4f}",
            f"{agg.get('mean_frac_hits', 0):.3f}",
            f"{int(agg.get('n_screens', 0))}",
        )
    console.print(table)


@app.command("collect-dataset")
def collect_dataset(
    screen_set: str = typer.Option(
        "public_train",
        help="public|public_train|public_val|public_test|/path/to.yaml. "
        "public_train = biogrid train fold (held out from the public test "
        "set used for evaluation).",
    ),
    screen: Optional[str] = typer.Option(
        None, help="Comma-sep dataset_names to restrict to (overrides screen_set)."
    ),
    n_traces: int = typer.Option(
        3, "--n-traces",
        help="# of stochastic teacher traces to collect per screen. Each "
        "trace uses a distinct seed; variation comes from sampling at "
        "temperature 1.0.",
    ),
    acq: str = typer.Option(
        "llm_single",
        help="LLM acquisition to trace: llm_single (open-vocab) | llm_single_blind.",
    ),
    lm_config: Optional[str] = typer.Option(
        "configs/lm/collect-GLM-5.1.yaml", "--lm-config",
        help="assaybench collect-*.yaml configuring the teacher LLM "
        "(default: GLM-5.1).",
    ),
    batch_size: int = typer.Option(100),
    n_steps: int = typer.Option(10, help="AL steps per trace."),
    warm_start: int = typer.Option(
        0, "--warm-start-size",
        help="# of random genes revealed before step 1 (warm-start variant). "
        "0 = pure cold-start AL. Set to batch_size for a random first batch, "
        "or k*batch_size for k random batches.",
    ),
    seed: int = typer.Option(0, help="Base RNG seed; trace i uses seed+i."),
    metrics: str = typer.Option("hits_auc,andcg,batch_hits,batch_diversity"),
    out: str = typer.Option(
        "output/datasets/sft_traces.jsonl", "--out",
        help="Output JSONL path (overwritten at start unless --resume).",
    ),
    resume: bool = typer.Option(
        False, "--resume/--no-resume",
        help="Resume by scanning --out: keep (screen, trace) pairs that already "
        "reached the final step, drop partial pairs, and only run the missing "
        "ones. Without this the output file is truncated and everything re-runs.",
    ),
    parallel: int = typer.Option(
        1, "--parallel", "-j",
        help="# of (screen, trace) runs to execute concurrently (thread pool).",
    ),
    max_shortfall_frac: float = typer.Option(
        1.0, "--max-shortfall-frac",
        help="Abort a run if the running acquisition shortfall fraction "
        "exceeds this. 1.0 (default) disables — we keep partial traces.",
    ),
    verbose: bool = typer.Option(True),
):
    """Generate a post-training (SFT / distillation) dataset of teacher traces.

    Runs the teacher LLM across the resolved screens ``n_traces`` times each
    and writes one chat-messages record per AL step (reasoning inline as
    ``<think>...</think>`` + the gene list), tagged with the per-batch metrics
    for downstream filtering. Collection runs are not persisted as sweeps
    (no result.json, no sweep entry) — the dataset file is the output.
    """
    import concurrent.futures as _cf
    from pathlib import Path

    from .experiment.runner import RunConfig, run_one_screen
    from .tasks import load_screens

    if acq.lower() not in _LLM_CLIENT_ACQS:
        console.print(
            f"[red]error:[/red] acq {acq!r} is not an LLM acquisition; "
            f"choose one of {sorted(_LLM_CLIENT_ACQS)}."
        )
        raise typer.Exit(1)

    loaded_lm = _load_lm_config_or_exit(lm_config)
    teacher_label = (loaded_lm.model_name if loaded_lm else None) or acq

    try:
        screens = load_screens(
            dataset_names=_csv(screen) or None, target_set=screen_set
        )
    except Exception as e:
        console.print(
            f"[red]error:[/red] failed to resolve screens for "
            f"screen_set={screen_set!r}: {type(e).__name__}: {e}"
        )
        raise typer.Exit(1)
    if not screens:
        console.print(
            f"[red]error:[/red] no screens resolved for screen_set={screen_set!r}."
        )
        raise typer.Exit(1)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: scan the existing dataset to decide which (screen, trace) pairs are
    # already complete (have a record for every step 1..n_steps). Complete pairs
    # are kept verbatim; partial pairs are dropped (their stale records removed)
    # and re-run, since AL steps are sequential and can't be resumed mid-trace.
    # Without --resume we start fresh and truncate so we never append to stale data.
    complete_keys: set[tuple[str, int]] = set()
    n_resumed_records = 0
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        from .data.sft_export import pair_is_complete, scan_sft_records

        by_pair, misc_lines = scan_sft_records(out_path)
        complete_keys = {
            k for k, items in by_pair.items()
            if pair_is_complete({s for s, _ in items}, n_steps)
        }
        keep_lines = list(misc_lines)
        for k in complete_keys:
            keep_lines.extend(line for _, line in by_pair[k])
        n_resumed_records = len(keep_lines)
        out_path.write_text(
            "\n".join(keep_lines) + ("\n" if keep_lines else ""), encoding="utf-8"
        )
    else:
        out_path.write_text("", encoding="utf-8")

    msf = None if max_shortfall_frac >= 1.0 else max_shortfall_frac

    def _make_cfg(s_seed: int, t_idx: int) -> RunConfig:
        acq_overrides: dict = {}
        if loaded_lm is not None:
            # SFT collection enforces a non-empty answer: retry empty
            # completions (up to the client's max_empty_retries cap).
            acq_overrides["llm"] = _llm_with_empty_policy(
                loaded_lm.client, retry_on_empty=True
            )
        return RunConfig(
            screen_set=screen_set,
            model="null",  # llm_single is open-vocab; no internal model needed.
            acq=acq,
            batch_size=batch_size,
            n_steps=n_steps,
            warm_start_size=warm_start,
            seed=s_seed,
            acq_overrides=acq_overrides,
            metrics=_csv(metrics) or ["hits_auc", "andcg", "batch_hits", "batch_diversity"],
            persist=False,
            parallel=1,
            max_shortfall_frac=msf,
            collect_traces=True,
            trace_out=str(out_path),
            trace_idx=t_idx,
            teacher_label=teacher_label,
        )

    all_pairs = [(scr, t) for scr in screens for t in range(n_traces)]
    pairs = [
        (scr, t) for (scr, t) in all_pairs
        if (scr.dataset_name, t) not in complete_keys
    ]
    n_skipped = len(all_pairs) - len(pairs)
    console.print(
        f"[cyan]collect-dataset[/cyan]: {len(screens)} screen(s) x "
        f"{n_traces} trace(s) = {len(all_pairs)} run(s); teacher={teacher_label}; "
        f"out={out_path}"
    )
    if resume:
        console.print(
            f"[cyan]resume[/cyan]: {n_skipped} complete pair(s) skipped "
            f"({n_resumed_records} record(s) kept), {len(pairs)} run(s) to (re)do."
        )
    if not pairs:
        console.print("[green]Nothing to do[/] — all (screen, trace) pairs complete.")
        raise typer.Exit(0)

    def _run_pair(pair):
        scr, t = pair
        cfg = _make_cfg(seed + t, t)
        try:
            run_one_screen(scr, cfg, verbose=verbose)
            return (scr.dataset_name, t, None)
        except Exception as e:  # keep going; one bad screen shouldn't kill the run
            return (scr.dataset_name, t, f"{type(e).__name__}: {e}")

    errors: list[tuple[str, int, str]] = []
    if parallel > 1:
        with _cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            results = list(ex.map(_run_pair, pairs))
    else:
        results = [_run_pair(p) for p in pairs]
    for name, t, err in results:
        if err:
            errors.append((name, t, err))
            console.print(f"[yellow]run failed[/yellow] {name} trace {t}: {err}")

    n_examples = sum(
        1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    manifest = {
        "out": str(out_path),
        "screen_set": screen_set,
        "fold": "yearfold0=train" if screen_set == "public_train" else screen_set,
        "teacher": teacher_label,
        "lm_config": lm_config,
        "acq": acq,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "warm_start_size": warm_start,
        "base_seed": seed,
        "n_screens": len(screens),
        "n_traces": n_traces,
        "resume": resume,
        "n_runs_total": len(all_pairs),
        "n_runs_skipped_complete": n_skipped,
        "n_runs": len(pairs),
        "n_failed_runs": len(errors),
        "n_examples": n_examples,
        "screens": [s.dataset_name for s in screens],
        "failed": [{"screen": n, "trace": t, "error": e} for n, t, e in errors],
    }
    manifest_path = out_path.parent / (out_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.print(
        f"[green]Wrote[/] {n_examples} SFT example(s) -> {out_path}\n"
        f"[green]Manifest[/] -> {manifest_path}"
        + (f"\n[yellow]{len(errors)} run(s) failed[/yellow]" if errors else "")
    )


@app.command("score-existing")
def score_existing(
    predictions_path: str = typer.Argument(...),
    out_path: str = typer.Option("output/baselines/scores.json"),
):
    """Score a JSONL of (screen, ranked_genes) predictions against ground truth."""
    from .baselines.score_existing_predictions import score_file

    res = score_file(predictions_path, out_path=out_path)
    rprint(json.dumps(res, indent=2))


@app.command("select-public-screens")
def select_public_screens(
    n_target: int = typer.Option(
        20, help="Number of screens to select (15-20 recommended)."
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Public AssayBench yearfold0 split to curate: test or validation.",
    ),
    out_path: Optional[str] = typer.Option(
        None,
        help="Output YAML path. Defaults to the standard path for the split.",
    ),
    per_screen_tsv: Optional[str] = typer.Option(
        None,
        help="Optional path to dump per-screen AnDCG@100 across every selected split "
        "screen as TSV (audit trail).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Re-run the publishable public-AssayBench screen selector.

    Picks high-signal, diverse public screens stratified by phenotype.
    gemini-3-pro AnDCG@100 >= 0.05 is the signal floor; baseline scores are
    retained only for audit, and no winner balancing is applied. Writes a
    regenerated ``assayloop-test`` (``--split test``) or
    ``assayloop-validation`` (``--split validation``) manifest under
    ``$ASSAYLOOP_OUTPUT/screen_sets/``; the manifests the code reads ship in
    ``assaybench.data.screen_sets`` and are updated by copying over them.
    """
    import sys

    from .scripts.select_default_public_screens import main as _select_main

    saved = sys.argv[:]
    sys.argv = [
        "select_default_public_screens",
        "--n-target", str(n_target),
        "--split", split,
    ]
    if out_path:
        sys.argv += ["--output", out_path]
    if per_screen_tsv:
        sys.argv += ["--per-screen-tsv", per_screen_tsv]
    if dry_run:
        sys.argv.append("--dry-run")
    try:
        rc = _select_main()
    finally:
        sys.argv = saved
    raise typer.Exit(rc)


@app.command()
def info():
    """Print resolved configuration (paths, vLLM endpoint, trace dir)."""
    payload = {
        "LLM_PROVIDER": config.LLM_PROVIDER,
        "VLLM_BASE_URL": config.VLLM_BASE_URL,
        "VLLM_MODEL_NAME": config.VLLM_MODEL_NAME,
        "SCRATCH_PATH": str(config.SCRATCH_PATH),
        "OUTPUT_PATH": str(config.OUTPUT_PATH),
        "PRESAGE_CACHE_PATH": str(config.PRESAGE_CACHE_PATH),
        "LLM_TRACE_PATH": str((config.OUTPUT_PATH / "runs").resolve()),
    }
    rprint(payload)


@app.command()
def trace(
    run_id: str = typer.Argument(..., help="Run id (e.g. sweep-abc123-00-1708)."),
    limit: int = typer.Option(5, "--limit", "-n", help="Show the most-recent N calls."),
    full: bool = typer.Option(False, "--full", help="Print the entire prompt+response (default truncates)."),
):
    """View the local LLM-call trace for a run.

    Each call captures the full message history, response, finish_reason,
    latency, and token usage. Traces are written to
    ``$OUTPUT_PATH/runs/<run_id>/llm_calls.jsonl``; nothing is uploaded.
    """
    import json as _json

    path = config.OUTPUT_PATH / "runs" / run_id / "llm_calls.jsonl"
    if not path.is_file():
        rprint(f"[yellow]No trace file at {path}[/]")
        rprint("Either the run made no LLM calls (null/random baselines),")
        rprint("or it ran before local tracing was added.")
        raise typer.Exit(code=1)
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            calls.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue
    rprint(f"[bold]{run_id}[/]: {len(calls)} LLM call(s) in {path}")
    show = calls[-limit:]
    for i, c in enumerate(show, start=len(calls) - len(show) + 1):
        rprint(f"\n[bold cyan]--- call #{i} ---[/]")
        rprint(f"  model        : {c.get('model')}")
        rprint(f"  finish_reason: {c.get('finish_reason')}")
        rprint(f"  latency_s    : {c.get('latency_s'):.2f}" if isinstance(c.get("latency_s"), (int, float)) else f"  latency_s    : {c.get('latency_s')}")
        usage = c.get("usage") or {}
        if usage:
            rprint(f"  tokens in/out: {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}")
        if c.get("error"):
            rprint(f"  [red]error[/]      : {c['error']}")
        resp = c.get("response_text") or ""
        reasoning = c.get("reasoning_text") or ""
        cap = 10_000 if full else 500
        if reasoning and not resp:
            rprint(f"  [red]empty content; reasoning leaked through ({len(reasoning)} chars):[/]")
            rprint(f"    {reasoning[:cap]}")
        else:
            rprint(f"  response ({len(resp)} chars):")
            rprint(f"    {resp[:cap]}{'...' if len(resp) > cap else ''}")


def _safe_delta(a: Any, b: Any) -> float | None:
    """``a - b`` when both are numeric, else None (for metric comparisons)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def _eval_ranker_sweep(
    *,
    checkpoint: str,
    screen_set: str,
    batch_size: int,
    n_steps: int,
    parallel: int,
    tag: str,
    seed: int,
    device: str,
    ckpt_file: str = "model.pt",
    ignore_context: bool = False,
    verbose: bool = True,
):
    """Run the amortized ranker through the AL loop over a screen set and
    persist it as a sweep, like any other run type.

    ``ignore_context`` runs the diagnostic ablation where the model never sees
    the accumulated AL observations (static description-only ranking)."""
    from .experiment.runner import RunConfig, run_sweep

    overrides: dict[str, Any] = {
        "checkpoint": checkpoint, "device": device, "ckpt_file": ckpt_file,
    }
    if ignore_context:
        overrides["ignore_context"] = True
    cfg = RunConfig(
        screen_set=screen_set,
        model="amortized_ranker",
        acq="greedy",
        batch_size=batch_size,
        n_steps=n_steps,
        seed=seed,
        model_overrides=overrides,
        metrics=["hits_auc", "andcg", "batch_hits"],
        persist=True,
        parallel=parallel,
        tag=(tag or "amortized_ranker") + ("-noctx" if ignore_context else ""),
    )
    return run_sweep(cfg, verbose=verbose)


# Default GLM-5.1 val/test traces used to seed the handoff. Looked up under
# ASSAYLOOP_SHARED_PATH/runs; override with --warm-start-eval-dir.
_HANDOFF_RUNS_DIR = str(config.SHARED_PATH / "runs")
_HANDOFF_PREFIX_BY_SET = {
    "public": "sweep-e0dd3203-",
    "public_validation": "sweep-ad6af08f-",
}


def _eval_ranker_handoff_sweep(
    *,
    checkpoint: str,
    screen_set: str,
    batch_size: int,
    n_steps: int,
    parallel: int,
    tag: str,
    seed: int,
    device: str,
    warm_dir: str,
    warm_prefix: str,
    n_by_screen: dict[str, int] | None,
    default_n: int,
    ckpt_file: str = "model.pt",
    verbose: bool = True,
):
    """Run the GLM-5.1 -> transformer handoff through the AL loop and persist
    the sweep: GLM replays its recorded first ``n`` rounds (per screen), then
    the ranker continues. ``n_by_screen`` overrides ``default_n`` per screen."""
    from .acquisitions.glm_handoff_acq import GlmHandoffAcquisition
    from .acquisitions.greedy_from_model import GreedyFromModel
    from .amortized.warmstart import WarmStart
    from .experiment.runner import RunConfig, run_sweep

    ws = WarmStart.from_run_dirs(warm_dir, warm_prefix)

    def acq_factory(screen):
        traces = ws.traces.get(screen.dataset_name) or []
        rounds = traces[0] if traces else []
        n = (n_by_screen or {}).get(screen.dataset_name, default_n)
        return GlmHandoffAcquisition(rounds, n, GreedyFromModel(seed=seed))

    cfg = RunConfig(
        screen_set=screen_set,
        model="amortized_ranker",
        acq="greedy",
        batch_size=batch_size,
        n_steps=n_steps,
        seed=seed,
        model_overrides={
            "checkpoint": checkpoint,
            "device": device,
            "ckpt_file": ckpt_file,
        },
        metrics=["hits_auc", "andcg", "batch_hits"],
        persist=True,
        parallel=parallel,
        tag=tag,
    )
    return run_sweep(cfg, verbose=verbose, acq_factory=acq_factory)


@app.command("train-ranker")
def train_ranker(
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Artifact dir (default output/rankers/<run-name>)."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name (wandb + artifact dir)."),
    train_screen_set: str = typer.Option(
        "public_train",
        "--train-screen-set",
        help=(
            "Screen set used for training. Default is the full public biogrid "
            "train fold. Pass a YAML path for custom splits (e.g. LOPO folds)."
        ),
    ),
    train_size: Optional[int] = typer.Option(None, "--train-size", help="Subsample N train screens (default: all)."),
    val_screen_set: str = typer.Option(
        "public_validation",
        "--val-screen-set",
        help=(
            "Screen set used for checkpoint selection / in-training validation. "
            "Default is the curated 20-screen validation set. Use public_val "
            "for the full public validation fold."
        ),
    ),
    val_size: Optional[int] = typer.Option(
        None,
        "--val-size",
        help=(
            "Subsample N screens from --val-screen-set (default: all; 0 = all)."
        ),
    ),
    epochs: int = typer.Option(10, help="Epochs per DAgger round."),
    batch_size: int = typer.Option(32, help="Training minibatch size (screens per step)."),
    lr: float = typer.Option(1e-3, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, "--weight-decay", help="AdamW weight decay."),
    d_model: int = typer.Option(256, help="Transformer width."),
    d_gene: int = typer.Option(256, help="Learned gene-embedding dim."),
    d_hit: int = typer.Option(32, "--d-hit", help="Hit-label embedding dim."),
    num_layers: int = typer.Option(2, help="Transformer encoder layers."),
    nhead: int = typer.Option(4, help="Attention heads."),
    dim_feedforward: int = typer.Option(512, "--dim-feedforward", help="Transformer feed-forward width."),
    dropout: float = typer.Option(0.1, help="Transformer dropout."),
    encoder: str = typer.Option(
        "transformer", "--encoder",
        help="Encoder: transformer | modernbert-embed | modernbert-text.",
    ),
    bert_model: str = typer.Option(
        "answerdotai/ModernBERT-base", "--bert-model",
        help="HF backbone for the modernbert-* encoders.",
    ),
    bert_lr: float = typer.Option(2e-5, "--bert-lr", help="Backbone LR (modernbert-* encoders)."),
    freeze_bert: bool = typer.Option(False, "--freeze-bert", help="Freeze the backbone (CPU smoke test)."),
    bert_pool: str = typer.Option("cls", "--bert-pool", help="Text-encoder pooling: cls | mean."),
    text_max_tokens: int = typer.Option(2048, "--text-max-tokens", help="Max tokens for the text encoder."),
    use_description: bool = typer.Option(
        True, "--description/--no-description",
        help="Include the screen description in the model input. --no-description trains the ablation (embedding encoders use a constant query token; the text encoder renders observed genes only).",
    ),
    max_context: int = typer.Option(1024, help="Max observed-gene context tokens."),
    max_targets: int = typer.Option(4096, help="Max supervised genes per screen (hits always kept)."),
    hit_context_frac: float = typer.Option(0.5, help="Prob. of forcing hits into the sampled context."),
    leave_out: bool = typer.Option(
        True, "--leave-out/--no-leave-out",
        help="Supervise only held-out genes (exclude the observed context from targets), making the context necessary. --no-leave-out restores the old context-independent objective.",
    ),
    init_gene_factors: Optional[str] = typer.Option(
        None, "--init-gene-factors",
        help="Initialise the gene-embedding table (bilinear head) from an external geometry, decoupled from the loss. This is stage 1 of training (paper 4.1.1) and the axis of the embedding-initialisation ablation (paper 6.7.1). SPEC is 'bpmf:<result.pkl>' (BPMF posterior-mean V, forces d_gene=K), 'presage:<source>' (e.g. presage:genept; PCA-reduced to --d-gene), 'mf[:<reg>[:raw]]' (pickle-free BPMF surrogate: masked ALS matrix *completion* of the observed gene x screen hit matrix to --d-gene, ridge reg default 0.1 -- proper missing-data handling, unlike svd), 'svd[:<norm>[:raw]]' (truncated SVD of the marginal-centered train hit matrix to --d-gene; zero-fills unobserved entries), or 'random[:<std>]' (Gaussian control geometry at --d-gene; std default 1.0). For mf/svd the factors are by default projected onto a constant-norm shell (BPMF-like uniform per-gene norm, so ranking is direction- i.e. context-driven); append ':raw' to keep the heavy-tailed ALS/SVD magnitudes. Composable with --objective hits (hard-hit loss). Pair with --freeze-gene-factors to hold the geometry fixed.",
    ),
    freeze_gene_factors: bool = typer.Option(
        True, "--freeze-gene-factors/--no-freeze-gene-factors",
        help="[--init-gene-factors] freeze the initialised gene factors (and bias) so the encoder must infer the screen latent û from context (default: freeze). Use --no-freeze-gene-factors to let V keep training from the initialisation (kept out of weight decay).",
    ),
    objective: str = typer.Option(
        "relevance", "--objective",
        help="Training target: 'relevance' = static unmasked relevance score (MSE, default); 'hits' = teacher-free self-supervised in-context hit prediction (soft-target BCE on the genes' own binary hit labels); 'bpmf' = train the bilinear head *directly* with BPMF's own objective — probit likelihood P(hit)=Φ(û·V_g) + Gaussian priors on the amortized screen latent û and the gene factors V (teacher-free, fully differentiable on GPU). Pair 'hits'/'bpmf' with --disable-gene-bias for the pure bilinear bottleneck.",
    ),
    bpmf_sigma_u: float = typer.Option(
        1.0, "--bpmf-sigma-u",
        help="[--objective bpmf] prior std σ_u on the screen latent û (BPMF's sigma_u; default 1.0). Smaller = stronger shrinkage.",
    ),
    bpmf_sigma_v: float = typer.Option(
        1.0, "--bpmf-sigma-v",
        help="[--objective bpmf] prior std σ_v on the gene factors V (BPMF's sigma_v; default 1.0). Applied explicitly, so V is kept out of AdamW weight decay.",
    ),
    bpmf_fit_sigma: bool = typer.Option(
        False, "--bpmf-fit-sigma",
        help="[--objective bpmf] learn σ_u, σ_v jointly (their log-σ normalisers give a proper MLE) instead of holding them fixed.",
    ),
    bpmf_variational: bool = typer.Option(
        False, "--bpmf-variational",
        help="[--objective bpmf] amortized *variational* BPMF: the encoder emits (µ, logσ²) for û, training samples û~N(µ,σ_q²) (reparam) and uses KL to the prior instead of the MAP penalty. Inference still uses the posterior mean. MAP (point estimate) when off.",
    ),
    disable_gene_bias: bool = typer.Option(
        False, "--disable-gene-bias",
        help="Zero and freeze the per-gene bias so the ranking must come from û·V_g (gene factors V stay trainable). Removes the static-prior shortcut. Use a small --d-gene (e.g. 16) for the bilinear bottleneck.",
    ),
    gene_norm_reg: float = typer.Option(
        0.0, "--gene-norm-reg",
        help="Sphere/shell regularizer strength: add coef * mean_g (||V_g|| - r)^2 over the batch's trained gene factors, pulling per-gene norms toward a common radius r so ranking is decided by *direction* (context-controllable), not by a few high-norm genes. Reproduces the mf/svd shell-init geometry for the from-scratch --objective bpmf head. 0 = off. No-op when the gene table is frozen.",
    ),
    gene_norm_target: float = typer.Option(
        0.0, "--gene-norm-target",
        help="[--gene-norm-reg] target shell radius r. 0 (default) uses the batch's (detached) mean row-norm, i.e. pure norm-variance reduction that lets the BPMF prior set the scale; >0 pins the radius to a fixed value.",
    ),
    ctx_shift_coef: float = typer.Option(
        0.0, "--ctx-shift-coef",
        help="Context-ranking auxiliary strength: add coef * aux that rewards the context prediction for ranking the screen's hits high (a bounded, scale-invariant signal -- supplies the reordering the bpmf probit objective lacks). 0 = off. Transformer/embed encoders only (not modernbert_text).",
    ),
    ctx_shift_mode: str = typer.Option(
        "directional", "--ctx-shift-mode",
        help="[--ctx-shift-coef] 'directional' (default): point-biserial correlation corr(s_ctx, label) -- bounded [-1,1], scale-free. 'rank': soft-AUC, mean sigmoid(s_ctx[hit]-s_ctx[non-hit]) -- bounded [0,1]. 'raw': literal mean|Δscore|/std(cold) using an empty-context encode -- a direction-blind control that reproduces the high-shift/low-reorder (quiet-sweep-89) failure mode. NB: directional/rank reference only s_ctx (an earlier cov/margin form was unbounded and blew the logits up).",
    ),
    cold_anchor_coef: float = typer.Option(
        0.0, "--cold-anchor-coef",
        help="Cold-start anchor strength: add coef * BCE(empty-context prediction, marginal hit rate), pinning the cold-start to a sane context-free prior. Makes the cold->ctx gap (context reordering) well-posed -- pair with --ctx-shift-coef so the aux improves ranking *with* context instead of gaming a free-floating cold-start. 0 = off. Transformer/embed encoders only.",
    ),
    em_period: int = typer.Option(
        0, "--em-period",
        help="EM / coordinate-ascent alternation period (epochs). >0 alternates "
             "training the transformer (inference net) and the gene factors V every "
             "N epochs (epoch 0..N-1 = transformer phase on the fixed init V, then "
             "N..2N-1 = V phase, ...), i.e. amortized ALS. Requires "
             "--no-freeze-gene-factors so V is a live block. 0 = joint training "
             "(default). Teacher-free route to BPMF-quality embeddings.",
    ),
    warm_start_glm_train: Optional[str] = typer.Option(
        None, "--warm-start-glm-train",
        help="GLM-5.1 handoff: path to the train traces jsonl (output/datasets/GLM-5-1_train.jsonl). When set with --warm-start-prob>0, a fraction of supervised contexts are GLM's first-n acquisitions (a realistic handoff context) so the model learns to continue from GLM-quality observations.",
    ),
    warm_start_prob: float = typer.Option(
        0.0, "--warm-start-prob",
        help="Probability a sampled training context is a GLM warm-start prefix (vs the usual random subset). ~0.5 is a good mix. 0 disables.",
    ),
    warm_start_n: str = typer.Option(
        "random", "--warm-start-n",
        help="GLM rounds to use for warm-start contexts: 'random' (n~U{1..rounds}) or a fixed int.",
    ),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers. Use >0 (e.g. 4-8) to overlap batch preparation with the GPU step."),
    dagger_rounds: int = typer.Option(1, "--dagger-rounds", help="On-policy DAgger rounds after round 0 (0 = random only)."),
    dagger_screens: int = typer.Option(64, help="# train screens rolled out per DAgger round."),
    val_al_screens: int = typer.Option(20, help="# val screens for the in-training AL metric (0 = use the ENTIRE validation set)."),
    eval_every: int = typer.Option(1, "--eval-every", help="Run validation (MSE + AL metric) every N epochs. Raise this when evaluating the full val set."),
    al_batch_size: int = typer.Option(100, help="AL batch size for val/DAgger rollouts."),
    al_n_steps: int = typer.Option(10, help="AL steps for val/DAgger rollouts."),
    text_backend: str = typer.Option("auto", help="Text embedder: auto|openai|azure|local. auto serves the shipped cache without a key."),
    wandb_project: str = typer.Option("assayloop-amortized-ranker", "--wandb-project"),
    wandb_entity: Optional[str] = typer.Option(None, "--wandb-entity", help="W&B entity. Default: your account default, or $WANDB_ENTITY."),
    wandb_mode: str = typer.Option("online", "--wandb-mode", help="online|offline|disabled."),
    wandb_group: Optional[str] = typer.Option(None, "--wandb-group", help="Optional W&B group name."),
    wandb_tags: Optional[str] = typer.Option(None, "--wandb-tags", help="Comma-sep W&B tags."),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto", help="auto|cpu|cuda."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip the post-training AL evaluation sweep."),
    eval_screen_set: str = typer.Option("public", "--eval-screen-set", help="Screen set for the auto-eval sweep."),
    eval_parallel: int = typer.Option(1, "--eval-parallel", help="Parallel screens in the eval sweep."),
):
    """Train the amortized gene ranker on BioGRID public-train, select
    checkpoints on the curated public validation set by default, then evaluate
    in the AL loop on the public test set, persisting the result as a sweep.
    Validation metrics are logged to wandb.
    """
    from .amortized.train import run_training

    encoder_type = encoder.replace("-", "_")

    summary = run_training(
        out_dir=out_dir, run_name=run_name,
        train_screen_set=train_screen_set, train_size=train_size,
        val_screen_set=val_screen_set, val_size=val_size,
        epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        d_model=d_model, d_gene=d_gene, d_hit=d_hit,
        num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout,
        encoder_type=encoder_type, bert_model=bert_model, bert_lr=bert_lr,
        freeze_bert=freeze_bert, bert_pool=bert_pool, text_max_tokens=text_max_tokens,
        use_description=use_description,
        max_context=max_context,
        max_targets=max_targets, hit_context_frac=hit_context_frac, leave_out=leave_out,
        init_gene_factors=init_gene_factors, freeze_gene_factors=freeze_gene_factors,
        objective=objective,
        bpmf_sigma_u=bpmf_sigma_u, bpmf_sigma_v=bpmf_sigma_v,
        bpmf_fit_sigma=bpmf_fit_sigma, bpmf_variational=bpmf_variational,
        disable_gene_bias=disable_gene_bias,
        gene_norm_reg=gene_norm_reg, gene_norm_target=gene_norm_target,
        ctx_shift_coef=ctx_shift_coef, ctx_shift_mode=ctx_shift_mode,
        cold_anchor_coef=cold_anchor_coef,
        em_period=em_period,
        num_workers=num_workers,
        warm_start_glm_train=warm_start_glm_train, warm_start_prob=warm_start_prob,
        warm_start_n=warm_start_n,
        dagger_rounds=dagger_rounds, dagger_screens=dagger_screens,
        val_al_screens=val_al_screens, eval_every=eval_every, al_batch_size=al_batch_size,
        al_n_steps=al_n_steps, text_backend=text_backend, wandb_project=wandb_project,
        wandb_entity=wandb_entity, wandb_mode=wandb_mode, wandb_group=wandb_group,
        wandb_tags=_csv(wandb_tags), seed=seed, device=device,
    )
    rprint(f"[green]Trained[/] ranker -> {summary['out_dir']}")
    rprint(f"  best val_n_hits_vs_random: {summary.get('best_val_n_hits_vs_random')}")
    rprint(f"  vocab={summary['vocab_size']} val_cov={summary['val_coverage']:.2%} test_cov={summary['test_coverage']:.2%}")

    if not no_eval:
        rprint(f"[cyan]Evaluating[/] ranker on '{eval_screen_set}' (AL loop)...")
        sweep = _eval_ranker_sweep(
            checkpoint=summary["out_dir"], screen_set=eval_screen_set,
            batch_size=al_batch_size, n_steps=al_n_steps, parallel=eval_parallel,
            tag=summary["run_name"], seed=seed, device=device,
        )
        agg = sweep.aggregate or {}
        rprint(f"[green]Eval sweep[/] {sweep.sweep_id}: "
               f"mean_n_hits_vs_random={agg.get('mean_n_hits_vs_random')}, "
               f"mean_hits_auc={agg.get('mean_hits_auc')}")


@app.command("warm-text-cache")
def warm_text_cache(
    screen_sets: str = typer.Option(
        "public_train,public_validation,public_val",
        "--screen-sets",
        help="CSV of screen sets whose descriptions to embed into the on-disk cache. Defaults to every set train-ranker embeds (train + both validation folds). Missing sets are skipped.",
    ),
    text_backend: str = typer.Option("auto", "--text-backend", help="Text embedder: auto|openai|azure|local. openai and azure are the same model and share the cache; local is a different vector space."),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="Override the embedding model id (else the backend default)."),
):
    """Pre-populate the description-embedding cache for every screen set, once.

    train-ranker embeds the train/val descriptions via Azure at startup and
    caches them to disk (keyed by text+model), so DAgger rounds and re-runs are
    free. Running several train-ranker jobs in parallel cold-cache makes them all
    hit Azure at the same time (slow / can hang on a flaky endpoint). Run this
    once first, then launch the training jobs serially against the warm cache.
    """
    from .amortized import text_embed as te
    from .tasks import load_screens

    embedder = te.get_text_embedder(
        text_backend, **({"model": text_model} if text_model else {})
    )
    rprint(f"[cyan]Warming text cache[/] via {embedder.name}")
    screens: list = []
    seen: set[str] = set()
    for s in _csv(screen_sets):
        try:
            loaded = load_screens(target_set=s)
        except Exception as e:  # noqa: BLE001 - missing/unknown set: skip with a note
            rprint(f"  [yellow]skip[/] {s}: {e}")
            continue
        n_new = 0
        for sc in loaded:
            if sc.dataset_name not in seen:
                seen.add(sc.dataset_name)
                screens.append(sc)
                n_new += 1
        rprint(f"  {s}: {len(loaded)} screens ({n_new} new)")
    if not screens:
        rprint("[red]No screens loaded; nothing to cache.[/]")
        raise typer.Exit(1)
    te.embed_screens(screens, embedder)  # de-dupes by text, batches, saves cache
    rprint(f"[green]Done[/]: cache warmed for {len(screens)} unique screens.")


@app.command("train-ranker-rl")
def train_ranker_rl(
    init_checkpoint: Optional[str] = typer.Option(
        None, "--init-checkpoint",
        help="Warm-start ranker dir (a supervised checkpoint, e.g. output/rankers/<name>). Omit with --from-scratch to RL-train a fresh bilinear ranker.",
    ),
    init_ckpt_file: str = typer.Option(
        "model.pt", "--init-ckpt-file",
        help="Which checkpoint file to load from --init-checkpoint dir (model.pt or model_last.pt).",
    ),
    from_scratch: bool = typer.Option(
        False, "--from-scratch",
        help="Cold-start RL: build a fresh RankerNet from the arch flags below instead of warm-starting. Pair with --disable-gene-bias (+ small --d-gene) and --aux-hit-coef to learn the bilinear ranker by RL alone (no teacher, no pretraining). KL anchor is forced off.",
    ),
    encoder: str = typer.Option("transformer", "--encoder", help="from-scratch encoder: transformer|modernbert_embed|modernbert_text."),
    d_model: int = typer.Option(384, "--d-model", help="from-scratch: encoder width."),
    d_gene: int = typer.Option(16, "--d-gene", help="from-scratch: gene-factor dim K (the bilinear bottleneck; keep small, e.g. 10-32)."),
    d_hit: int = typer.Option(64, "--d-hit", help="from-scratch: hit-embedding dim."),
    nhead: int = typer.Option(2, "--nhead", help="from-scratch: attention heads."),
    num_layers: int = typer.Option(3, "--num-layers", help="from-scratch: transformer layers."),
    dim_feedforward: int = typer.Option(1024, "--dim-feedforward", help="from-scratch: FFN width."),
    dropout: float = typer.Option(0.0, "--dropout", help="from-scratch: dropout."),
    bert_model: str = typer.Option("answerdotai/ModernBERT-base", "--bert-model", help="from-scratch: HF model id for the BERT encoders."),
    disable_gene_bias: bool = typer.Option(
        False, "--disable-gene-bias",
        help="Zero+freeze the per-gene bias so the ranking must come from û·V_g (BPMF-style bilinear bottleneck, V trainable). Recommended with --from-scratch. Inherited from the checkpoint config on warm-start runs.",
    ),
    aux_hit_coef: float = typer.Option(
        0.0, "--aux-hit-coef",
        help="Weight of the auxiliary online self-supervised in-context hit-prediction BCE (predict held-out genes' own hit labels from the observed context; reuses the policy forward). The dense signal that lets V be learned by RL from scratch. Try ~1.0 from scratch; ~0.5 as a regularizer on warm-start runs.",
    ),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Artifact dir (default output/rankers/<run-name>)."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name (wandb + artifact dir)."),
    train_screen_set: str = typer.Option("public_train", "--train-screen-set", help="Screens rolled out for RL."),
    train_size: Optional[int] = typer.Option(None, "--train-size", help="Subsample N train screens (default: all)."),
    eval_screen_set: str = typer.Option("public_validation", "--eval-screen-set", help="Held-out screens for the eval/early-stop metric."),
    eval_size: Optional[int] = typer.Option(None, "--eval-size", help="Subsample N eval screens (default: all)."),
    eval_screens: int = typer.Option(30, "--eval-screens", help="# held-out screens scored per eval round (0 = all)."),
    train_eval_screens: int = typer.Option(30, "--train-eval-screens", help="# train screens scored each eval round for the train-vs-test split (0 = all)."),
    test_eval_set: Optional[str] = typer.Option(None, "--test-eval-set", help="If set (e.g. 'public'), also evaluate the policy on this held-out *test* set every eval round and record it (history 'eval_test', wandb 'eval_test/*'). Lets you track val->test generalization across the whole RL trajectory, not just at the best-val checkpoint. Off by default."),
    epochs: int = typer.Option(3, help="Passes over the train screens."),
    screens_per_step: int = typer.Option(1, "--screens-per-step", help="Screens accumulated per optimizer step."),
    group_size: int = typer.Option(8, "--group-size", help="GRPO rollouts per screen (>=2)."),
    n_steps: int = typer.Option(10, "--n-steps", help="AL steps per rollout."),
    batch_size: int = typer.Option(100, "--batch-size", help="Genes acquired per AL step (k)."),
    gamma: float = typer.Option(1.0, help="Reward discount."),
    temperature: float = typer.Option(1.0, help="Policy softmax temperature."),
    kl_coef: float = typer.Option(0.1, "--kl-coef", help="KL anchor to the warm-start reference."),
    ent_coef: float = typer.Option(0.01, "--ent-coef", help="Entropy bonus coefficient."),
    terminal_coef: float = typer.Option(1.0, "--terminal-coef", help="Weight of the terminal n_hits_vs_random bonus (adjusted reward mode only)."),
    reward_mode: str = typer.Option(
        "telescope", "--reward-mode",
        help="telescope: per-step increments of cumulative n_hits_vs_random (return == deployed metric; default). "
             "context_delta: hits from the context-informed batch minus the frozen reference's static no-context top-k (forces context use). "
             "context_delta_reset: like context_delta but the no-context baseline is refreshed to the current policy every --ctx-reset-epochs (prevents the reward saturation / group-variance collapse that can cause a destructive update vs a stale reference). "
             "nvr_terminal: a single terminal reward = the final cumulative n_hits_vs_random (directly optimize the deployed metric, no per-step shaping; sparser than telescope). "
             "adjusted: per-step batch_random_adjusted_hit_rate + terminal bonus.",
    ),
    ctx_reset_epochs: int = typer.Option(
        25, "--ctx-reset-epochs",
        help="For --reward-mode context_delta_reset: refresh the no-context baseline to the current policy every N epochs.",
    ),
    adv_std_floor: float = typer.Option(
        1e-6, "--adv-std-floor",
        help="Floor on the per-group return std used to normalize advantages. The default (1e-6) lets advantages explode when a group's returns are near-identical (low-entropy policy on one screen) -> a destructive update with no KL trust region. Raise to ~0.1-0.5 to cap that blow-up.",
    ),
    adv_clip: float = typer.Option(
        0.0, "--adv-clip",
        help="If >0, clip normalized advantages to [-adv_clip, +adv_clip]. A direct guard against advantage spikes (try ~5-10). 0 disables.",
    ),
    warm_start_glm_train: Optional[str] = typer.Option(
        None, "--warm-start-glm-train",
        help="GLM-5.1 handoff training: path to the train traces jsonl (e.g. output/datasets/GLM-5-1_train.jsonl, from collect-dataset). Each rollout pre-observes GLM's first n acquired genes (true labels) and the transformer continues for n_steps-n rounds; terminal EF is over the full budget.",
    ),
    warm_start_n: str = typer.Option(
        "0", "--warm-start-n",
        help="How many GLM rounds to hand off before the transformer takes over. 'random' (V2) samples n~U{0..n_steps-1} per screen per epoch (pick the deployment n later via a validation sweep); an int trains a fixed handoff point. 0 = no warm start.",
    ),
    warm_start_eval_dir: Optional[str] = typer.Option(
        None, "--warm-start-eval-dir",
        help="Dir of GLM val run dirs (each with result.json) used to seed the in-training validation handoff eval. Defaults to $ASSAYLOOP_SHARED_PATH/runs.",
    ),
    warm_start_eval_prefix: Optional[str] = typer.Option(
        "sweep-ad6af08f-", "--warm-start-eval-prefix",
        help="Prefix of the val GLM run dirs under --warm-start-eval-dir.",
    ),
    eval_warm_n: int = typer.Option(
        0, "--eval-warm-n",
        help="GLM rounds to seed during the in-training validation eval (deployment-style handoff used for checkpoint selection). 0 = pure-transformer val eval. For a random-n (V2) run, set a representative n (e.g. 5).",
    ),
    handoff_select: str = typer.Option(
        "off", "--handoff-select",
        help="Save a separate model_handoff.pt at the best handoff-VAL epoch. 'gemini' evaluates a Gemini warm-start->ranker handoff on val (+ test) every eval round (adjusted EF); 'off' disables.",
    ),
    handoff_val_prefix: str = typer.Option("sweep-6cd2d623-", "--handoff-val-prefix", help="Run-dir prefix of the LLM's validation traces for the handoff eval."),
    handoff_test_prefix: str = typer.Option("sweep-a79fd5ce-", "--handoff-test-prefix", help="Run-dir prefix of the LLM's test traces for the handoff eval."),
    handoff_warm_n: int = typer.Option(3, "--handoff-warm-n", help="Warm-start rounds seeded from the LLM traces in the handoff eval."),
    handoff_trace_dir: list[str] = typer.Option(None, "--handoff-trace-dir", help="Dir(s) to search for handoff traces (repeatable). Default: output/runs plus the shared runs dir."),
    rl_max_context: int = typer.Option(1024, "--rl-max-context", help="Cap observed-gene context fed to the embedding encoders per step."),
    max_tokens: int = typer.Option(0, "--max-tokens", help="modernbert-text only: per-step token budget for RL rollouts (0 = use the checkpoint's). The 8192-token training cap is too slow for RL; try 1024-2048."),
    gpus: int = typer.Option(1, "--gpus", help="GPUs for data-parallel training (>1 launches torchrun; screens are sharded across ranks and gradients averaged)."),
    lr: float = typer.Option(1e-5, help="AdamW LR (head / from-scratch transformer)."),
    bert_lr: float = typer.Option(1e-6, "--bert-lr", help="ModernBERT backbone LR (modernbert-embed / modernbert-text warm start)."),
    weight_decay: float = typer.Option(0.0, "--weight-decay", help="AdamW weight decay."),
    grad_clip: float = typer.Option(1.0, "--grad-clip", help="Grad-norm clip."),
    eval_every: int = typer.Option(1, "--eval-every", help="Eval every N epochs."),
    split_half_eval: bool = typer.Option(
        False, "--split-half-eval",
        help="Split eval screens into two halves for cross-validated checkpoint selection. "
             "Best-on-A is scored on B and vice versa; the average is an unbiased metric "
             "with no validation leakage (useful for LOPO where eval = held-out test fold).",
    ),
    text_backend: str = typer.Option("auto", help="Text embedder: auto|openai|azure|local (auto reuses the checkpoint's)."),
    no_description: bool = typer.Option(
        False, "--no-description",
        help="Force the no-description ablation regardless of the checkpoint (normally the flag is inherited from the warm-start checkpoint).",
    ),
    wandb_project: str = typer.Option("assayloop-amortized-ranker", "--wandb-project"),
    wandb_entity: Optional[str] = typer.Option(None, "--wandb-entity", help="W&B entity. Default: your account default, or $WANDB_ENTITY."),
    wandb_mode: str = typer.Option("online", "--wandb-mode", help="online|offline|disabled."),
    wandb_group: Optional[str] = typer.Option(None, "--wandb-group", help="Optional W&B group name."),
    wandb_tags: Optional[str] = typer.Option(None, "--wandb-tags", help="Comma-sep W&B tags."),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto", help="auto|cpu|cuda."),
    full_genome: bool = typer.Option(False, "--full-genome", help="Expand candidates to the universe of genes (union across all screens, filtered to freq >= 2) during RL rollouts. Matches the open-vocab LLM setting."),
    save_every: int = typer.Option(0, "--save-every", help="Save a checkpoint every N epochs (model_epoch_N.pt). 0 = only best + last."),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip the post-training AL evaluation sweep."),
    eval_parallel: int = typer.Option(1, "--eval-parallel", help="Parallel screens in the post-training eval sweep."),
):
    """RL-fine-tune a warm-started amortized ranker to directly optimize the AL
    metrics (per-step batch_random_adjusted_hit_rate + terminal n_hits_vs_random)
    with GRPO-style group-normalized advantages and a KL anchor to the init.

    Logs explore/exploit diagnostics (vs random / kNN / diversified baselines, a
    hits-only-context ablation, and train-vs-test n_hits_vs_random) to wandb.
    Saves a checkpoint consumable by ``eval-ranker`` and
    ``eval-ranker-handoff``.

    Use --gpus N for data-parallel training across N GPUs (re-launches under
    torchrun; screens are sharded across ranks and gradients averaged).
    """
    from .amortized.rl import run_rl_training

    if not from_scratch and not init_checkpoint:
        rprint("[red]Provide --init-checkpoint, or use --from-scratch to RL-train a fresh model.[/]")
        raise typer.Exit(1)

    rl_kwargs = dict(
        init_checkpoint=init_checkpoint, init_ckpt_file=init_ckpt_file,
        from_scratch=from_scratch,
        encoder_type=encoder.replace("-", "_"),
        d_model=d_model, d_gene=d_gene, d_hit=d_hit, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout,
        bert_model=bert_model, disable_gene_bias=disable_gene_bias,
        aux_hit_coef=aux_hit_coef,
        out_dir=out_dir, run_name=run_name,
        train_screen_set=train_screen_set, train_size=train_size,
        eval_screen_set=eval_screen_set, eval_size=eval_size,
        eval_screens=eval_screens, train_eval_screens=train_eval_screens,
        test_eval_set=test_eval_set,
        epochs=epochs, screens_per_step=screens_per_step, group_size=group_size,
        n_steps=n_steps, batch_size=batch_size, gamma=gamma, temperature=temperature,
        kl_coef=kl_coef, ent_coef=ent_coef, terminal_coef=terminal_coef,
        reward_mode=reward_mode, ctx_reset_epochs=ctx_reset_epochs,
        adv_std_floor=adv_std_floor, adv_clip=adv_clip,
        warm_start_glm_train=warm_start_glm_train, warm_start_n=warm_start_n,
        warm_start_eval_dir=((warm_start_eval_dir or _HANDOFF_RUNS_DIR)
                            if eval_warm_n > 0 else None),
        warm_start_eval_prefix=warm_start_eval_prefix, eval_warm_n=eval_warm_n,
        handoff_select=handoff_select, handoff_val_prefix=handoff_val_prefix,
        handoff_test_prefix=handoff_test_prefix, handoff_warm_n=handoff_warm_n,
        handoff_trace_dirs=(list(handoff_trace_dir) if handoff_trace_dir else None),
        rl_max_context=rl_max_context, rl_max_tokens=(max_tokens or None),
        lr=lr, bert_lr=bert_lr,
        weight_decay=weight_decay, grad_clip=grad_clip, eval_every=eval_every,
        split_half_eval=split_half_eval,
        text_backend=text_backend, use_description=(False if no_description else None),
        wandb_project=wandb_project,
        wandb_entity=wandb_entity, wandb_mode=wandb_mode, wandb_group=wandb_group,
        wandb_tags=_csv(wandb_tags), seed=seed, device=device,
        full_genome=full_genome, save_every=save_every,
    )

    if gpus and gpus > 1:
        # Data-parallel: re-launch training under torchrun. The parent process
        # fixes run_name/out_dir so it can read the summary back, runs only the
        # distributed training (rank 0 writes artifacts), then does the post eval
        # sweep itself (single process) below.
        run_name = run_name or f"ranker-rl-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        resolved_out = Path(out_dir) if out_dir else (config.OUTPUT_PATH / "rankers" / run_name)
        resolved_out.mkdir(parents=True, exist_ok=True)
        rl_kwargs["run_name"] = run_name
        rl_kwargs["out_dir"] = str(resolved_out)
        kwargs_path = resolved_out / "_rl_ddp_kwargs.json"
        kwargs_path.write_text(json.dumps(rl_kwargs), encoding="utf-8")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={gpus}", "--standalone",
            "--module", "assayloop.amortized.rl_ddp",
            "--kwargs-json", str(kwargs_path),
        ]
        rprint(f"[cyan]Launching[/] distributed RL on {gpus} GPUs via torchrun "
               f"(run_name={run_name})...")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            rprint(f"[red]torchrun exited with code {proc.returncode}[/]")
            raise typer.Exit(proc.returncode)
        summary_path = resolved_out / "summary.json"
        if not summary_path.is_file():
            rprint(f"[red]No summary.json under {resolved_out} (training failed?).[/]")
            raise typer.Exit(1)
        summary = json.loads(summary_path.read_text())
    else:
        summary = run_rl_training(**rl_kwargs)

    rprint(f"[green]RL-trained[/] ranker -> {summary['out_dir']}")
    # nvr_adj is EF. The label is spelled the way the key is spelled in
    # summary.json / history.json, which the published checkpoints already
    # carry, so grepping the printout finds the field in the file.
    rprint(f"  init  eval nvr_adj: {summary.get('init_eval_nvr_adj')} "
           f"(raw={summary.get('init_eval_n_hits_vs_random')}, knn={summary.get('init_eval_knn')})")
    rprint(f"  best  eval nvr_adj: {summary.get('best_eval_nvr_adj')} "
           f"(beat init: {summary.get('best_beat_init')})")
    if summary.get("best_handoff_val_nvr_adj") is not None:
        rprint(f"  best  handoff-val nvr_adj: {summary.get('best_handoff_val_nvr_adj')} "
               f"(init={summary.get('init_handoff_val_nvr_adj')}) -> model_handoff.pt")
    if not summary.get("best_beat_init"):
        rprint("  [yellow]best did not beat init -> model.pt == INIT weights; "
               "eval model_last.pt for the updated policy.[/]")
    rprint("  model.pt = best-on-val (nvr_adj), model_last.pt = final-epoch"
           + (", model_handoff.pt = best handoff-val"
              if summary.get("best_handoff_val_nvr_adj") is not None else ""))

    if not no_eval:
        rprint("[cyan]Evaluating[/] RL ranker on 'public' (AL loop)...")
        sweep = _eval_ranker_sweep(
            checkpoint=summary["out_dir"], screen_set="public",
            batch_size=batch_size, n_steps=n_steps, parallel=eval_parallel,
            tag=summary["run_name"], seed=seed, device=device,
        )
        agg = sweep.aggregate or {}
        rprint(f"[green]Eval sweep[/] {sweep.sweep_id}: "
               f"mean_n_hits_vs_random={agg.get('mean_n_hits_vs_random')}, "
               f"mean_hits_auc={agg.get('mean_hits_auc')}")


@app.command("eval-ranker")
def eval_ranker(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained ranker dir (output/rankers/<name>)."),
    screen_set: str = typer.Option(
        "public",
        "--screen-set",
        help=(
            "public|public_validation|public_val|public_test|public_train|all."
        ),
    ),
    batch_size: int = typer.Option(100, help="AL batch size."),
    n_steps: int = typer.Option(10, help="AL steps."),
    parallel: int = typer.Option(1, "--parallel", "-j", help="Parallel screens."),
    tag: str = typer.Option("", help="Sweep tag (defaults to the checkpoint name)."),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto", help="auto|cpu|cuda."),
    ckpt_file: str = typer.Option(
        "model.pt", "--ckpt-file",
        help="Which weights file inside --checkpoint to load. model.pt is the "
             "best-on-validation epoch; model_last.pt is the final epoch. The "
             "paper's AssayLoop rows use model_last.pt (see full_genome_table.py).",
    ),
    ablate_context: bool = typer.Option(
        True, "--ablate-context/--no-ablate-context",
        help="Also run a no-context (static, description-only) AL sweep and report the context delta.",
    ),
    context_value: bool = typer.Option(
        True, "--context-value/--no-context-value",
        help="Compute the lightweight offline counterfactual context-value scalar.",
    ),
    cv_screens: int = typer.Option(60, "--cv-screens", help="# screens sampled for the offline context-value check."),
    cv_ctx_size: int = typer.Option(50, "--cv-ctx-size", help="Oracle context size for the offline context-value check."),
):
    """Evaluate a trained amortized ranker in the AL loop, persisting the
    result as a sweep exactly like other run types.

    With ``--ablate-context`` (default), a second no-context sweep is run so you
    can read off how much the AL context actually buys. ``--context-value``
    additionally computes a fast offline counterfactual lift (no AL loop)."""
    from pathlib import Path as _Path

    ckpt = _Path(checkpoint)
    if not (ckpt / ckpt_file).is_file():
        rprint(f"[red]error:[/red] no {ckpt_file} under {checkpoint!r}.")
        raise typer.Exit(1)
    tag = tag or ckpt.name
    sweep = _eval_ranker_sweep(
        checkpoint=str(ckpt), screen_set=screen_set, batch_size=batch_size,
        n_steps=n_steps, parallel=parallel, tag=tag, seed=seed, device=device,
        ckpt_file=ckpt_file,
    )
    agg = sweep.aggregate or {}
    ctx_ef = agg.get("mean_n_hits_vs_random")
    ctx_auc = agg.get("mean_hits_auc")
    rprint(f"[green]Eval sweep[/] {sweep.sweep_id} ({screen_set}): "
           f"mean_n_hits_vs_random={ctx_ef}, mean_hits_auc={ctx_auc}")

    if ablate_context:
        rprint("[cyan]Ablation[/] running no-context (static) AL sweep...")
        sweep_nc = _eval_ranker_sweep(
            checkpoint=str(ckpt), screen_set=screen_set, batch_size=batch_size,
            n_steps=n_steps, parallel=parallel, tag=tag, seed=seed, device=device,
            ckpt_file=ckpt_file, ignore_context=True,
        )
        agg_nc = sweep_nc.aggregate or {}
        nc_ef = agg_nc.get("mean_n_hits_vs_random")
        nc_auc = agg_nc.get("mean_hits_auc")
        rprint(f"[green]No-context sweep[/] {sweep_nc.sweep_id} ({screen_set}): "
               f"mean_n_hits_vs_random={nc_ef}, mean_hits_auc={nc_auc}")
        d_ef = _safe_delta(ctx_ef, nc_ef)
        d_auc = _safe_delta(ctx_auc, nc_auc)
        rprint(f"[bold]Context delta[/] (ctx - noctx): "
               f"n_hits_vs_random={d_ef}, hits_auc={d_auc}")
        if d_ef is not None and abs(d_ef) < 1e-6:
            rprint("[yellow]~zero delta: the model is not using the AL context.[/]")

    if context_value:
        from .amortized.context_value import context_value as _context_value

        rprint("[cyan]Offline context-value[/] (counterfactual lift, no AL loop)...")
        cv = _context_value(
            checkpoint=str(ckpt), screen_set=screen_set, device=device,
            n_screens=cv_screens, ctx_size=cv_ctx_size, batch_size=batch_size, seed=seed,
        )
        out_path = ckpt / "context_value.json"
        out_path.write_text(json.dumps(cv, indent=2), encoding="utf-8")
        rprint(f"  mean AP lift (oracle-ctx - empty-ctx): {cv.get('mean_ap_lift')}")
        rprint(f"  frac screens with positive lift: {cv.get('frac_positive')}  "
               f"(n={cv.get('n_screens')})")
        rprint(f"  mean hits@batch lift: {cv.get('mean_hits_at_batch_lift')}  "
               f"-> wrote {out_path}")


@app.command("eval-ranker-handoff")
def eval_ranker_handoff(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained ranker dir (output/rankers/<name>)."),
    screen_set: str = typer.Option("public", "--screen-set", help="public|public_validation|..."),
    n: int = typer.Option(
        -1, "--n",
        help="# GLM warm-start rounds for every screen (0..n_steps). "
             "-1 means 'must pass --n'.",
    ),
    batch_size: int = typer.Option(100, help="AL batch size (must match the GLM traces; default 100)."),
    n_steps: int = typer.Option(10, help="AL steps."),
    parallel: int = typer.Option(1, "--parallel", "-j", help="Parallel screens."),
    tag: str = typer.Option("", help="Sweep tag (defaults to a descriptive name)."),
    seed: int = typer.Option(0),
    device: str = typer.Option("auto", help="auto|cpu|cuda."),
    ckpt_file: str = typer.Option(
        "model.pt", "--ckpt-file",
        help="Which weights file inside --checkpoint to load. model.pt is the "
             "best-on-validation epoch; model_last.pt is the final epoch. The "
             "paper's AssayLoop rows use model_last.pt (see full_genome_table.py).",
    ),
    warm_dir: str = typer.Option(_HANDOFF_RUNS_DIR, "--warm-dir", help="Dir of GLM-5.1 run traces."),
    warm_prefix: str = typer.Option("", "--warm-prefix", help="GLM run-dir prefix (auto by screen-set)."),
):
    """Run the GLM-5.1 -> AssayFormer **handoff** and persist it as a sweep.

    Per screen, GLM-5.1 replays its recorded first ``n`` AL rounds (from the
    shared val/test traces, with true labels) and the trained ranker continues
    for the remaining ``n_steps - n`` rounds. The sweep is persisted exactly
    like ``eval-ranker``, so the recovery curve shows GLM's strong early
    rounds and then the transformer.

    ``--n`` is the handoff round; the paper's AssayLoop configuration is
    ``--n 3``."""
    from pathlib import Path as _Path

    ckpt = _Path(checkpoint)
    if not (ckpt / ckpt_file).is_file():
        rprint(f"[red]error:[/red] no {ckpt_file} under {checkpoint!r}.")
        raise typer.Exit(1)

    dev = device
    if dev == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = warm_prefix or _HANDOFF_PREFIX_BY_SET.get(screen_set, "")
    if not prefix:
        rprint(
            f"[red]error:[/red] no GLM trace prefix known for screen_set={screen_set!r}; "
            "pass --warm-prefix explicitly."
        )
        raise typer.Exit(1)

    default_n = max(0, n)
    if n < 0:
        rprint("[red]error:[/red] pass --n <int>.")
        raise typer.Exit(1)
    label = f"n{n}"

    tag = tag or f"{ckpt.name}-handoff-{label}"
    sweep = _eval_ranker_handoff_sweep(
        checkpoint=str(ckpt), screen_set=screen_set, batch_size=batch_size,
        n_steps=n_steps, parallel=parallel, tag=tag, seed=seed, device=dev,
        warm_dir=warm_dir, warm_prefix=prefix,
        n_by_screen=None, default_n=default_n, ckpt_file=ckpt_file,
    )
    agg = sweep.aggregate or {}
    rprint(
        f"[green]Handoff sweep[/] {sweep.sweep_id} ({screen_set}, {label}): "
        f"mean_n_hits_vs_random={agg.get('mean_n_hits_vs_random')}, "
        f"mean_hits_auc={agg.get('mean_hits_auc')}"
    )
    rprint("It is a normal sweep; scripts/full_genome_table.py and "
           "scripts/export_recovery_curves.py will pick it up.")


@app.command(
    "train-bpmf",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_bpmf(
    ctx: typer.Context,
    cpu: bool = typer.Option(
        False, "--cpu",
        help="Use the reference CPU Gibbs sampler instead of the GPU one.",
    ),
) -> None:
    """Fit the BPMF factorization that initialises AssayFormer's gene embeddings.

    Unknown options are forwarded verbatim to the underlying script, so the paper's
    factorisation (K=10, 2000 iterations, 1000 burn-in, thin 2) is::

        assayloop train-bpmf --target-set public_train --K 10

    Run ``python -m assayloop.scripts.train_bpmf_gpu --help`` for the full option
    list; ``--K`` accepts a comma list on the GPU path so one call can sweep K.
    """
    module = "assayloop.scripts.train_bpmf" if cpu else "assayloop.scripts.train_bpmf_gpu"
    cmd = [sys.executable, "-m", module, *ctx.args]
    rprint(f"[cyan]Running[/] {' '.join(cmd[1:])}")
    raise typer.Exit(subprocess.run(cmd).returncode)


# name -> (module, paper reference, what it draws). Prerequisite steps are listed
# under "needs" so `assayloop figure --list` is a runnable recipe, not just an index.
_PAPER_FIGURES: dict[str, dict[str, str]] = {
    "main-table": {
        "module": "assayloop.scripts.full_genome_table",
        "ref": "Table 2",
        "what": "Main results table: every method on the 20 test screens.",
        "needs": "one sweep per row (see the README's row -> command map)",
    },
    "recovery-curves": {
        "module": "assayloop.scripts.export_recovery_curves",
        "ref": "Table 2 (companion)",
        "what": "Per-step hit-recovery curves for every method in the main table.",
        "needs": "the same sweeps as main-table",
    },
    "label-ablation": {
        "module": "assayloop.scripts.plot_label_ablation",
        "ref": "Figure 4",
        "what": "With vs. without per-round hit labels, one dumbbell per method.",
        "needs": "the llm_single and llm_single_blind sweeps",
    },
    "llm-pathways": {
        "module": "assayloop.scripts.plot_llm_pathway_heatmap",
        "ref": "Figure 5",
        "what": "Reactome enrichment heatmap of what biology each LLM goes after.",
        "needs": "the per-LLM sweeps",
    },
    "pathway-sunburst": {
        "module": "assayloop.scripts.plot_pathway_sunburst",
        "ref": "Figure 6",
        "what": "Two-level pathway sunbursts for six representative methods.",
        "needs": "the corresponding sweeps",
    },
    "lopo": {
        "module": "assayloop.scripts.plot_lopo_results",
        "ref": "Figure 7",
        "what": "Leave-one-phenotype-out generalisation.",
        "needs": "a train-ranker run per LOPO split",
    },
    "scaling": {
        "module": "assayloop.scripts.scaling_law_plot",
        "ref": "Figure 8",
        "what": "EF@10 vs. training-set size and vs. model size.",
        "needs": "scaling_law_sweep then scaling_law_eval",
    },
    "embedding-init": {
        "module": "assayloop.scripts.plot_embedding_init_story",
        "ref": "Figure 9",
        "what": "Curated-network recovery vs. downstream task performance, by init.",
        "needs": "compute_embedding_matrix + eval_gene_networks per init",
    },
    "embedding-drift": {
        "module": "assayloop.scripts.plot_embedding_drift",
        "ref": "Figure 10",
        "what": "How far gene embeddings move during training (BPMF init).",
        "needs": "a checkpoint with saved gene embeddings",
    },
    "handoff-composition": {
        "module": "assayloop.scripts.paper_handoff_composition",
        "ref": "Figure 11",
        "what": "Per-round pathway/complex composition of the handoff's hits.",
        "needs": "an eval-ranker-handoff sweep",
    },
    "handoff-timeline": {
        "module": "assayloop.scripts.paper_handoff_timeline",
        "ref": "Figure 11",
        "what": "How acquisition behaviour changes at the LLM -> AssayFormer handoff.",
        "needs": "an eval-ranker-handoff sweep",
    },
    "influence": {
        "module": "assayloop.scripts.paper_influence_figure",
        "ref": "Figures 12-13",
        "what": "Context-conditional gene-influence heatmaps (boosted / suppressed).",
        "needs": "compute_influence_matrix",
    },
    "bpmf-organization": {
        "module": "assayloop.scripts.paper_bpmf_k10_organization",
        "ref": "Figure 14",
        "what": "How the BPMF K=10 space is organised: HDBSCAN / CORUM / Reactome, "
        "each under PCA and cosine UMAP.",
        "needs": "train-bpmf --K 10",
    },
    "embedding-drift-by-source": {
        "module": "assayloop.scripts.plot_embedding_drift_by_source",
        "ref": "Figure 15",
        "what": "Embedding drift during training, for every ablated init.",
        "needs": "one checkpoint per embedding init",
    },
    "diversity": {
        "module": "assayloop.scripts.diversity_analysis",
        "ref": "Table 2 (VS, PO columns)",
        "what": "Vendi score and pathway overlap from stored acquired batches.",
        "needs": "the sweeps being compared",
    },
}


@app.command(
    "figure",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def figure(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Figure name; omit with --list."),
    list_: bool = typer.Option(False, "--list", help="List the figures and exit."),
) -> None:
    """Regenerate a paper figure or table.

    Each entry is a thin wrapper over ``python -m assayloop.scripts.<module>``;
    unknown options are forwarded verbatim, and each script's own ``--help`` has
    the details. Figures read persisted results, so the runs a figure depends on
    (the "needs" column) have to exist first -- nothing is recomputed implicitly.
    """
    if list_ or not name:
        table = Table(title="Paper figures")
        table.add_column("name", style="cyan")
        table.add_column("paper")
        table.add_column("what it draws")
        table.add_column("needs", style="dim")
        for key, spec in _PAPER_FIGURES.items():
            table.add_row(key, spec["ref"], spec["what"], spec["needs"])
        console.print(table)
        if not name and not list_:
            raise typer.Exit(1)
        return

    spec = _PAPER_FIGURES.get(name)
    if spec is None:
        rprint(f"[red]error:[/red] unknown figure {name!r}. "
               f"Known: {', '.join(sorted(_PAPER_FIGURES))}")
        raise typer.Exit(1)

    cmd = [sys.executable, "-m", spec["module"], *ctx.args]
    rprint(f"[cyan]Running[/] {' '.join(cmd[1:])}  ({spec['ref']})")
    raise typer.Exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    app()
