"""Per-screen and per-sweep inner-loop runner.

Factories
---------

The CLI creates models and acquisitions by *name*, so a single
``assayloop run`` invocation can sweep across all of them without
having to import them. :func:`make_model` and :func:`make_acquisition`
below are the authoritative lists of what those names resolve to.

Each factory accepts an optional dict of overrides, so a sweep can vary
hyperparameters without redefining strings.

run_one_screen
--------------

Build a :class:`AssayBenchGeneBatchTask`, wire it into a
:class:`SequentialLoop`, run, persist, and return the
:class:`RunResult`.

run_sweep
---------

Drive ``run_one_screen`` across ``(screen, model, acquisition, seed)``
combinations, then aggregate per-screen HitsAUC into a single
mean-HitsAUC + per-screen-result blob.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from assaybench.core import SequentialLoop
from assaybench.core.types import RunResult

from .. import config
from ..tracing import run_trace_scope
from ..metrics import AnDCGAtK, BatchDiversity, BatchHits, HitsAUC
from ..tasks import (
    ScreenRecord,
    load_screens,
    make_task,
)

log = logging.getLogger("assayloop.experiment.runner")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_model(name: str, *, overrides: dict[str, Any] | None = None, seed: int = 0):
    """Build a model by name. ``seed`` becomes the default RNG seed if the
    user hasn't supplied one in ``overrides`` (so ``sweep --seeds 0,1,2``
    actually controls model randomness)."""
    o = dict(overrides or {})
    name = (name or "null").lower()

    if name == "null":
        from ..models.null_model import NullModel
        return NullModel()

    if name == "knn":
        from ..models.knn_gene_embedding import KNNGeneEmbedding
        return KNNGeneEmbedding(**o)

    if name == "rf":
        from ..models.rf_gene_embedding import RFGeneEmbedding
        o.setdefault("random_state", seed)
        return RFGeneEmbedding(**o)

    if name in ("bayesian_mlp", "bmlp"):
        from ..models.bayesian_mlp import BayesianMLPModel
        o.setdefault("random_state", seed)
        return BayesianMLPModel(**o)

    if name == "biobo":
        # BioBO (Li et al., ICLR 2026): Bayesian MLP surrogate over concatenated
        # PRESAGE gene-embedding sources. Pairs with the ``bio_ucb`` acquisition.
        from ..data.gene_embeddings.presage import presage_concat
        from ..models.bayesian_mlp import BayesianMLPModel
        sources = o.pop("sources", "genept,depmap")
        if isinstance(sources, str):
            sources = sources.split(",")
        o.setdefault("random_state", seed)
        return BayesianMLPModel(provider=presage_concat(sources), **o)

    if name in ("llm_ranker", "llm_incontext", "llm_rank"):
        from ..llm.client import LLMClientConfig
        from ..models.llm_incontext_ranker import LLMInContextRanker
        llm = o.pop("llm", None) or LLMClientConfig()
        o.setdefault("seed", seed)
        return LLMInContextRanker(llm=llm, **o)

    if name in ("screen_knn", "screenknn"):
        from ..models.screen_knn import ScreenKNNModel
        return ScreenKNNModel(**o)

    if name == "bpmf":
        from ..models.bpmf_model import BPMFModel
        o.setdefault("seed", seed)
        return BPMFModel(**o)
    if name in ("amortized_ranker", "ranker"):
        from ..models.amortized_ranker import AmortizedRankerModel
        return AmortizedRankerModel(**o)

    if name == "maml":
        from ..models.maml_ranker import MAMLRankerModel
        return MAMLRankerModel(**o)

    if name == "agent_ranker":
        # "Haiku-4.5 Agent": an LLM agent that analyses the public training
        # screens with real code execution, sandboxed in an Apptainer container
        # built by scripts/build_agent_sandbox.sh. Raises at construction if the
        # container is missing rather than silently becoming a plain ranker.
        from ..models.agent_ranker import AgentRankerModel
        o.setdefault("seed", seed)
        return AgentRankerModel(**o)

    if name in ("hypothesis_ranker", "hypothesis", "hyp_ranker"):
        # ICBR-EF: hypothesis-generation ranker (LLM proposes mechanisms, then
        # ranks genes against them).
        from ..llm.client import LLMClientConfig
        from ..models.hypothesis_ranker import HypothesisRankerModel
        llm = o.pop("llm", None) or LLMClientConfig(
            provider="anthropic", model="claude-sonnet-4-6",
            temperature=0.0, max_tokens=8192,
        )
        o.setdefault("seed", seed)
        return HypothesisRankerModel(llm=llm, **o)

    if name in ("llmnn", "llm_nn"):
        # LLMNN (Gupta et al., arXiv 2509.21403): the LLM proposes cluster
        # centers, nearest-neighbour expansion in a PRESAGE embedding space
        # fills the batch.
        from ..llm.client import LLMClientConfig
        from ..models.llm_nn import LLMNNModel
        llm = o.pop("llm", None) or LLMClientConfig(
            provider="anthropic", model="claude-sonnet-4-6",
            temperature=0.0, max_tokens=4096,
        )
        o.setdefault("seed", seed)
        return LLMNNModel(llm=llm, **o)

    raise ValueError(f"Unknown model name {name!r}")


def make_acquisition(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
    seed: int = 0,
):
    """Build an acquisition by name. ``seed`` becomes the default RNG seed
    if the user hasn't supplied one in ``overrides`` (so ``sweep --seeds
    0,1,2`` actually controls acquisition randomness too)."""
    o = dict(overrides or {})
    name = (name or "random").lower()

    if name == "random":
        from ..acquisitions.random_acq import RandomAcquisition
        o.setdefault("seed", seed)
        return RandomAcquisition(**o)

    if name == "greedy":
        from ..acquisitions.greedy_from_model import GreedyFromModel
        o.setdefault("seed", seed)
        return GreedyFromModel(**o)

    if name == "ucb":
        from ..acquisitions.ucb_from_model import UCBFromModel
        o.setdefault("seed", seed)
        return UCBFromModel(**o)

    if name in ("bio_ucb", "bioucb"):
        # BioBO's acquisition: UCB reweighted by a decaying piBO prior built
        # from hypergeometric pathway enrichment over the observed hits.
        from ..acquisitions.bio_ucb import BioUCBFromModel
        o.setdefault("seed", seed)
        return BioUCBFromModel(**o)

    # ``llm_single`` is the open-vocabulary acquisition (matches AssayBench:
    # no candidate list shown; the LLM picks from its own knowledge of the
    # genome; we filter to in-pool genes afterwards).
    #
    # ``llm_single_blind`` is the ablation variant that hides hit/miss
    # labels in the AL history block: the LLM still sees which genes
    # have already been sampled (so it doesn't waste batch picks on
    # them), but cannot use the labels to update its ranking. This
    # isolates the model's zero-shot prior from any active-learning
    # feedback. If ``llm_single`` and ``llm_single_blind`` perform
    # similarly, the AL loop is doing nothing useful for the LLM.
    if name in ("llm_single", "llm_single_blind", "llm"):
        from ..acquisitions.llm_single_acq import LLMSingleAcquisition
        from ..llm.client import LLMClientConfig
        llm = o.pop("llm", None) or LLMClientConfig()
        o.setdefault("seed", seed)
        if name == "llm_single_blind":
            o.setdefault("reveal_labels", False)
        return LLMSingleAcquisition(llm=llm, **o)

    raise ValueError(f"Unknown acquisition name {name!r}")


# ---------------------------------------------------------------------------
# Per-run config + result wrappers
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    task: str = "option2"            # for now, only option2 is wired
    screen_set: str = "public"       # "public" | "public_train" | ... | path-to-YAML
    dataset_names: list[str] | None = None
    model: str = "knn"
    acq: str = "greedy"
    batch_size: int = 100
    n_steps: Optional[int] = 10
    warm_start_size: int = 0
    seed: int = 0
    model_overrides: dict[str, Any] = field(default_factory=dict)
    acq_overrides: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=lambda: ["hits_auc", "andcg"])
    persist: bool = True
    tag: str = ""
    parallel: int = 1  # >1 => run screens concurrently on a thread pool
    # Maximum fraction of acquisition *shortfall* (requested slots
    # minus actually-acquired) the inner loop tolerates before
    # aborting. ``None`` disables. Default 0.5 catches degenerate
    # LLM / agent runs that under-supply or time out on most steps.
    # The loop NEVER pads under-supplied batches with random genes,
    # so this guard plus an honest shortfall_frac is the only signal
    # that a policy is broken.
    max_shortfall_frac: Optional[float] = 1.0
    max_shortfall_frac_warmup: int = 2
    universe_genes: list[str] | None = None
    # Trace collection for post-training (SFT/distillation) dataset export.
    # When ``collect_traces`` is True the run is kept OFF the dashboard
    # (no result.json persisted, no events published) and one SFT record
    # per LLM AL step is appended to ``trace_out``. ``trace_idx`` /
    # ``teacher_label`` are recorded in each example's metadata.
    collect_traces: bool = False
    trace_out: Optional[str] = None
    trace_idx: int = 0
    teacher_label: Optional[str] = None

    def short(self) -> str:
        return f"{self.task}|{self.model}|{self.acq}|bs{self.batch_size}|s{self.seed}"


@dataclass
class ScreenResult:
    screen_name: str
    run_id: str
    final_metrics: dict[str, float]
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_name": self.screen_name,
            "run_id": self.run_id,
            "final_metrics": self.final_metrics,
            "n_steps": self.n_steps,
        }


@dataclass
class SweepResult:
    sweep_id: str
    config: dict[str, Any]
    per_screen: list[ScreenResult]
    aggregate: dict[str, float]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "config": self.config,
            "per_screen": [s.to_dict() for s in self.per_screen],
            "aggregate": self.aggregate,
            "elapsed_s": self.elapsed_s,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metric(name: str):
    n = name.lower()
    if n in ("hits_auc", "auc"):
        return HitsAUC()
    if n in ("andcg", "andcg@k", "andcg_at_k"):
        return AnDCGAtK()
    if n in ("batch_hits",):
        return BatchHits()
    if n in ("batch_diversity",):
        return BatchDiversity()
    raise ValueError(f"Unknown metric {name!r}")


# Keys each batch metric writes; used to sync a step's recomputed values
# into ``final_metrics`` and to detect changes during backfill.
BATCH_HITS_KEYS = (
    "batch_n_hits",
    "batch_size_actual",
    "batch_hit_rate",
    "batch_random_expected_n_hits",
    "batch_random_expected_hit_rate",
    "batch_random_adjusted_hit_rate",
)
BATCH_DIVERSITY_KEYS = (
    "batch_diversity",
    "batch_diversity_vs_random",
    "batch_vendi",
    "batch_vendi_ratio",
    "batch_diversity_n",
    "batch_pathway_diversity",
    "batch_pathway_coverage",
    "batch_pathway_overlap_vs_random",
    "batch_pathway_n",
)


def backfill_batch_metrics(
    result: dict[str, Any],
    *,
    batch_hits: "BatchHits | None" = None,
    batch_diversity: "BatchDiversity | None" = None,
) -> bool:
    """Recompute per-step batch metrics on a persisted ``result.json`` dict.

    Reconstructs the per-step ``Observation`` lists from the stored
    ``acquired_batch`` + ``hits`` arrays and replays them through the
    *real* ``BatchHits`` / ``BatchDiversity`` metric objects, so backfilled
    values match what a fresh run would have produced. The random-baseline
    inputs ``BatchHits`` needs (``total_hits`` and the remaining-pool size)
    are read from each step's already-persisted ``hits_auc`` family, so no
    screen reload is required; ``BatchDiversity`` only needs the gene names.

    Mutates ``result`` in place (per-step ``metrics`` and the run's
    ``final_metrics``). Returns ``True`` if any value changed.

    Warm-start step 0 is intentionally skipped (it has no acquisition
    batch), matching the inner loop, but its observations still flow into
    the cumulative history used by later steps.
    """
    from assaybench.core.types import Observation

    if batch_hits is None and batch_diversity is None:
        return False

    steps = result.get("steps") or []
    if not steps:
        return False

    cumulative_obs: list[Observation] = []
    changed = False
    last_batch_keys_present: set[str] = set()

    for step in steps:
        genes = list(step.get("acquired_batch") or [])
        hits = list(step.get("hits") or [])
        new_obs = [
            Observation(candidate=g, label={"hit": bool(h)})
            for g, h in zip(genes, hits)
        ]
        cumulative_obs.extend(new_obs)

        # Step 0 is the warm-start synthetic record: the inner loop computes
        # no batch metrics there (no acquisition happened), so we skip it but
        # keep its observations in the cumulative history above.
        if step.get("step") == 0:
            continue

        m = step.get("metrics")
        if not isinstance(m, dict):
            continue

        if batch_hits is not None:
            total_hits = m.get("total_hits")
            n_remaining = m.get("n_remaining")
            if (
                isinstance(total_hits, (int, float))
                and isinstance(n_remaining, (int, float))
                and total_hits > 0
            ):
                from types import SimpleNamespace

                gt = SimpleNamespace(hits=[True] * int(round(total_hits)))
                candidates_remaining = [None] * int(round(n_remaining))
                out = batch_hits.score(
                    cumulative_obs,
                    None,
                    gt,
                    candidates_remaining,
                    new_observations=new_obs,
                    acquired_batch=genes,
                ) or {}
                for k, v in out.items():
                    if abs(float(m.get(k) or 0.0) - float(v)) > 1e-9:
                        changed = True
                    m[k] = v
                    last_batch_keys_present.add(k)

        if batch_diversity is not None:
            out = batch_diversity.score(
                cumulative_obs,
                None,
                None,
                [],
                new_observations=new_obs,
                acquired_batch=genes,
            ) or {}
            for k, v in out.items():
                if abs(float(m.get(k) or 0.0) - float(v)) > 1e-9:
                    changed = True
                m[k] = v
                last_batch_keys_present.add(k)

    # Sync the last (non-warm) step's batch keys into final_metrics, matching
    # the inner loop where final_metrics is the last step's metrics dict.
    if last_batch_keys_present:
        fm = result.get("final_metrics")
        if not isinstance(fm, dict):
            fm = {}
        last = steps[-1].get("metrics") or {}
        for k in last_batch_keys_present:
            if k in last:
                if abs(float(fm.get(k) or 0.0) - float(last[k])) > 1e-9:
                    changed = True
                fm[k] = last[k]
        result["final_metrics"] = fm

    return changed


def _serialise_run_result(run: RunResult) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "task_id": run.task_id,
        "config": run.config,
        "final_metrics": run.final_metrics,
        "steps": [
            {
                "step": s.step,
                "n_batch": len(s.acquired_batch),
                "acquired_batch": [str(c) for c in s.acquired_batch],
                "hits": [
                    bool(o.label.get("hit"))
                    if isinstance(o.label, dict)
                    else False
                    for o in s.new_observations
                ],
                "relevance_scores": [
                    float(o.label.get("relevance_score", float("nan")))
                    if isinstance(o.label, dict)
                    else float("nan")
                    for o in s.new_observations
                ],
                "metrics": s.metrics,
                "acquisition_trace": _strip_trace(s.acquisition_trace),
            }
            for s in run.history
        ],
    }


def _strip_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Drop very large fields from the trace so result.json stays small."""
    if not trace:
        return {}
    out = {}
    for k, v in trace.items():
        if isinstance(v, str) and len(v) > 1000:
            out[k] = v[:1000] + "...<truncated>"
        else:
            out[k] = v
    return out


def _ensure_run_dir(run_id: str) -> Path:
    d = config.OUTPUT_PATH / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Run a single screen
# ---------------------------------------------------------------------------


def run_one_screen(
    screen: ScreenRecord,
    cfg: RunConfig,
    *,
    run_id: str | None = None,
    sweep_id: str | None = None,
    verbose: bool = True,
    on_step: Any = None,
    model_obj: Any = None,
    acq_obj: Any = None,
) -> RunResult:
    """Build task/model/acq/metrics, run the inner loop, persist.

    ``on_step`` is an optional per-AL-step progress callback forwarded to the
    inner loop (see :class:`SequentialLoop`); used to surface in-loop progress
    (e.g. during GEPA optimization) without the noisy ``verbose`` print path.

    ``model_obj`` lets a caller inject an already-built Model instance instead
    of resolving ``cfg.model`` via :func:`make_model` (used by the amortized-
    ranker trainer to evaluate the live in-memory model on validation screens).

    ``acq_obj`` lets a caller inject an already-built AcquisitionFunction
    instead of resolving ``cfg.acq`` via :func:`make_acquisition` (used by the
    GLM->transformer handoff, where each screen needs its own recorded GLM
    rounds wired into the acquisition).
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
    task = make_task(screen, warm_start_size=cfg.warm_start_size, seed=cfg.seed,
                     universe_genes=cfg.universe_genes)
    model = model_obj if model_obj is not None else make_model(
        cfg.model, overrides=cfg.model_overrides, seed=cfg.seed
    )
    all_candidates = task.candidates()
    cov = model.coverage(all_candidates)
    if cov is not None:
        log.info("[%s] embedding coverage: %.1f%% (%d/%d genes)", run_id,
                 100 * cov, int(cov * len(all_candidates)), len(all_candidates))
    acq = acq_obj if acq_obj is not None else make_acquisition(
        cfg.acq, overrides=cfg.acq_overrides, seed=cfg.seed,
    )
    metrics = [_make_metric(m) for m in cfg.metrics]

    loop = SequentialLoop(
        task=task,
        model=model,
        acquisition=acq,
        metrics=metrics,
        batch_size=cfg.batch_size,
        run_id=run_id,
        sweep_id=sweep_id,
        max_shortfall_frac=cfg.max_shortfall_frac,
        max_shortfall_frac_warmup=cfg.max_shortfall_frac_warmup,
        on_step=on_step,
        # The loop itself does no logging. This is what associates every LLM
        # call the model or acquisition makes underneath with this run, so it
        # lands in ``output/runs/<run_id>/llm_calls.jsonl``.
        trace_scope=run_trace_scope,
    )
    result = loop.run(n_steps=cfg.n_steps, verbose=verbose)

    # Trace-collection runs are never persisted to ``output/runs`` (no
    # result.json) so the dashboard's directory scan never picks them up.
    if cfg.persist and not cfg.collect_traces:
        run_dir = _ensure_run_dir(run_id)
        (run_dir / "result.json").write_text(
            json.dumps(_serialise_run_result(result), indent=2, default=str)
        )
        (run_dir / "config.json").write_text(
            json.dumps({
                "screen_name": screen.dataset_name,
                "config": cfg.__dict__,
            }, indent=2, default=str)
        )

    if cfg.collect_traces and cfg.trace_out:
        from ..data.sft_export import write_sft_records

        teacher = cfg.teacher_label or cfg.acq
        n = write_sft_records(
            result,
            screen,
            cfg.trace_out,
            seed=cfg.seed,
            trace_idx=cfg.trace_idx,
            teacher_label=teacher,
        )
        if verbose:
            log.info(
                "[%s] wrote %d SFT record(s) for screen %s (trace %d) -> %s",
                run_id, n, screen.dataset_name, cfg.trace_idx, cfg.trace_out,
            )

    return result


# ---------------------------------------------------------------------------
# Run a sweep
# ---------------------------------------------------------------------------


def _aggregate(per_screen: list[ScreenResult]) -> dict[str, float]:
    """Aggregate per-screen final metrics into mean/min/max.

    Failed screens (those whose ``final_metrics`` contains an ``"error"``
    key) are excluded from the metric means but counted in ``n_failed`` so
    partial sweeps cannot quietly report optimistic means.
    """
    if not per_screen:
        return {}
    failed = [s for s in per_screen if "error" in (s.final_metrics or {})]
    ok = [s for s in per_screen if "error" not in (s.final_metrics or {})]

    keys = set()
    for s in ok:
        keys.update(s.final_metrics.keys())
    agg: dict[str, float] = {}
    for k in sorted(keys):
        vals = [
            s.final_metrics.get(k)
            for s in ok
            if isinstance(s.final_metrics.get(k), (int, float))
        ]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        agg[f"mean_{k}"] = float(sum(vals) / len(vals))
        agg[f"min_{k}"] = float(min(vals))
        agg[f"max_{k}"] = float(max(vals))
    agg["n_screens"] = float(len(per_screen))
    agg["n_ok"] = float(len(ok))
    agg["n_failed"] = float(len(failed))
    if failed:
        agg["failed_screens"] = [s.screen_name for s in failed]  # type: ignore[assignment]
    return agg


def _load_cached_screen_result(run_dir: Path, screen_name: str, run_id: str) -> ScreenResult | None:
    rj = run_dir / "result.json"
    if not rj.exists():
        return None
    try:
        d = json.loads(rj.read_text())
        fm = d.get("final_metrics") or {}
        n_steps = len(d.get("history", []))
        return ScreenResult(
            screen_name=screen_name, run_id=run_id,
            final_metrics=fm, n_steps=n_steps,
        )
    except Exception:
        return None


def run_sweep(
    cfg: RunConfig,
    *,
    sweep_id: str | None = None,
    verbose: bool = True,
    acq_factory: Any = None,
    resume: bool = False,
) -> SweepResult:
    """Run ``cfg`` across a screen set and persist a dashboard sweep.

    ``acq_factory`` is an optional ``callable(ScreenRecord) -> AcquisitionFunction``.
    When provided, each screen is run with its own acquisition instance instead
    of ``cfg.acq`` (used by the GLM->transformer handoff to inject per-screen
    GLM rounds). Returning ``None`` from the factory falls back to ``cfg.acq``.
    """
    sweep_id = sweep_id or f"sweep-{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    screens = load_screens(
        dataset_names=cfg.dataset_names,
        target_set=cfg.screen_set,
    )
    if not screens:
        raise RuntimeError(
            f"No screens resolved for screen_set={cfg.screen_set!r} / "
            f"dataset_names={cfg.dataset_names!r}"
        )
    if verbose:
        log.info("[%s] %d screens resolved", sweep_id, len(screens))

    parallel = max(1, int(getattr(cfg, "parallel", 1) or 1))
    n_total = len(screens)
    # Worker verbose prints would interleave under parallelism; emit a
    # single completion line per screen from the parent thread instead.
    worker_verbose = verbose and parallel == 1
    print_lock = threading.Lock()

    def _per_run_id(i: int, screen: ScreenRecord) -> str:
        return f"{sweep_id}-{i:02d}-{screen.dataset_name[:40].replace('/', '_')}"

    def _do(i: int, screen: ScreenRecord) -> tuple[int, ScreenResult]:
        per_run_id = _per_run_id(i, screen)
        if resume:
            cached = _load_cached_screen_result(
                config.OUTPUT_PATH / "runs" / per_run_id,
                screen.dataset_name, per_run_id,
            )
            if cached is not None:
                if verbose:
                    log.info("[%s] resuming — skipped %s (already done)",
                             sweep_id, screen.dataset_name)
                return i, cached
        try:
            acq_obj = acq_factory(screen) if acq_factory is not None else None
            res = run_one_screen(
                screen, cfg, run_id=per_run_id, sweep_id=sweep_id,
                verbose=worker_verbose, acq_obj=acq_obj,
            )
            return i, ScreenResult(
                screen_name=screen.dataset_name,
                run_id=per_run_id,
                final_metrics=res.final_metrics,
                n_steps=len(res.history),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Run failed for screen %s", screen.dataset_name)
            return i, ScreenResult(
                screen_name=screen.dataset_name,
                run_id=per_run_id,
                final_metrics={"error": f"{type(e).__name__}: {e}"},
                n_steps=0,
            )

    results: list[tuple[int, ScreenResult]] = []
    if parallel <= 1:
        for i, screen in enumerate(screens):
            if verbose:
                print(
                    f"\n=== [{sweep_id}] screen {i + 1}/{n_total}: "
                    f"{screen.dataset_name} (#genes={len(screen.genes)}, "
                    f"hits={screen.total_hits}) ==="
                )
            results.append(_do(i, screen))
    else:
        if verbose:
            print(
                f"\n=== [{sweep_id}] {n_total} screens, parallel={parallel} "
                f"({cfg.model}/{cfg.acq}) ==="
            )
            sys.stdout.flush()
        done = 0
        with ThreadPoolExecutor(
            max_workers=parallel, thread_name_prefix="assayloop-screen"
        ) as ex:
            futures = {ex.submit(_do, i, s): (i, s) for i, s in enumerate(screens)}
            for fut in as_completed(futures):
                pair = fut.result()
                results.append(pair)
                done += 1
                if verbose:
                    idx, sr = pair
                    s = futures[fut][1]
                    fm = sr.final_metrics or {}
                    norm = fm.get("hits_auc_normalized")
                    auc = fm.get("hits_auc")
                    err = fm.get("error")
                    if err:
                        suffix = f"  ERROR: {err}"
                    else:
                        bits = []
                        if isinstance(norm, (int, float)):
                            bits.append(f"hits_auc_norm={norm:.3f}")
                        if isinstance(auc, (int, float)):
                            bits.append(f"hits_auc={auc:.4f}")
                        suffix = "  " + " ".join(bits) if bits else ""
                    with print_lock:
                        print(
                            f"  [{sweep_id}] {done}/{n_total} done "
                            f"(#{idx:02d} {s.dataset_name}){suffix}"
                        )
                        sys.stdout.flush()

    results.sort(key=lambda p: p[0])
    per_screen = [r for _, r in results]

    aggregate = _aggregate(per_screen)
    elapsed = time.time() - t0

    sweep = SweepResult(
        sweep_id=sweep_id,
        config=cfg.__dict__,
        per_screen=per_screen,
        aggregate=aggregate,
        elapsed_s=elapsed,
    )

    if cfg.persist:
        sweep_dir = config.OUTPUT_PATH / "sweeps" / sweep_id
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "sweep.json").write_text(
            json.dumps(sweep.to_dict(), indent=2, default=str)
        )

    if verbose:
        print(f"\n=== Sweep {sweep_id} done in {elapsed:.1f}s ===")
        for k, v in aggregate.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return sweep


__all__ = [
    "RunConfig",
    "ScreenResult",
    "SweepResult",
    "make_model",
    "make_acquisition",
    "run_one_screen",
    "run_sweep",
    "backfill_batch_metrics",
]
