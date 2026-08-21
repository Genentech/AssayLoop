"""Re-evaluate baselines with full-genome candidate pool for fair LLM comparison.

Classical models (kNN, RF, BPMF, transformer) normally select from the
screen's gene library (~18k genes).  LLMs select from the whole genome
(~20k HGNC symbols) and pay a shortfall penalty for out-of-library picks.
This script re-runs each baseline with candidates = union of all genes
across all screens, so every method faces the same pool.

Usage::

    # Full universe (22k genes, includes pseudogenes etc.)
    uv run python -m assayloop.scripts.full_genome_table

    # Filtered: only genes in >= 2 test screens (~21k, removes rare junk)
    uv run python -m assayloop.scripts.full_genome_table --min-screen-freq 2
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from assaybench.benchmark.sequential import load_common_essentials
from assayloop import config
from assayloop.experiment.runner import RunConfig, make_model, run_one_screen
from assayloop.metrics.effective_pathways import (
    M_BATCH, M_DATASET, M_SCREEN, RETENTION, effective_pathways)
from assayloop.tasks import load_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("full_genome_table")

ROOT = Path(__file__).resolve().parents[3]
RANKERS_DIR = config.OUTPUT_PATH / "rankers"
OUTPUT_FILE = ROOT / "output" / "tables" / "full_genome_baselines.tex"

# Where finished runs are read from, in search order. ASSAYLOOP_RESULTS
# defaults to this repo's output/ (exactly where `assayloop run` writes);
# ASSAYLOOP_SHARED_PATH is an optional second location for runs produced by a
# teammate; ASSAYLOOP_PUBLISHED holds the downloaded bundle of LLM and
# external-baseline sweeps (scripts/fetch_sweeps.sh). Read through
# ``_run_result`` / ``_sweep_run_paths`` rather than indexing RUNS_DIR
# directly -- a lookup that only checks the first root silently ignores both
# the shared directory and the downloaded bundle.
RUNS_DIR = config.RESULTS_PATH / "runs"           # where new runs are written
SHARED_RUNS_DIR = config.SHARED_PATH / "runs"
RUN_DIRS = [RUNS_DIR, SHARED_RUNS_DIR, config.PUBLISHED_PATH / "runs"]


def _run_result(run_id: str) -> Path | None:
    """First ``<root>/<run_id>/result.json`` that exists, else None."""
    for rd in RUN_DIRS:
        fp = rd / run_id / "result.json"
        if fp.is_file():
            return fp
    return None


def _sweep_run_paths(sweep_id: str) -> list[Path]:
    """Every ``result.json`` belonging to *sweep_id*, across all roots.

    De-duplicated on run_id, first root wins, so a run you re-ran locally
    takes precedence over the shared or published copy of the same run.
    """
    seen: dict[str, Path] = {}
    for rd in RUN_DIRS:
        if not rd.is_dir():
            continue
        for fp in sorted(rd.glob(f"{sweep_id}-*/result.json")):
            seen.setdefault(fp.parent.name, fp)
    return [seen[k] for k in sorted(seen)]


def _require_runs(sweep_id: str, label: str, n_expected: int) -> list[Path]:
    """``_sweep_run_paths`` with a hard error instead of a degraded row.

    Every row backed by a published sweep has an obvious failure mode: with
    no runs found, the mean of an empty list becomes None and the row prints
    as ``-``. A dash in a results table reads as "we measured nothing here",
    not "you did not download the data", so this raises instead.
    """
    paths = _sweep_run_paths(sweep_id)
    if len(paths) < n_expected:
        raise config.MissingConfiguredPath(
            f"{label}: found {len(paths)} of {n_expected} runs for sweep "
            f"{sweep_id}.\nSearched: "
            + ", ".join(str(d) for d in RUN_DIRS)
            + f"\n{config.PUBLISHED_HINT}"
        )
    return paths


def _load_traces(trace_prefix: str, label: str) -> dict:
    """Warm-start traces for a handoff row, across every run root.

    Raises rather than returning an empty dict: the caller passes
    ``n_warm=n_warm if rounds else 0``, so no traces silently turns the
    handoff row into a plain AssayFormer row that still gets published under
    the handoff's name.
    """
    from assayloop.amortized.warmstart import load_run_traces

    traces: dict = {}
    for rd in RUN_DIRS:
        if rd.is_dir():
            traces.update(load_run_traces(rd, trace_prefix))
    if not traces:
        raise config.MissingConfiguredPath(
            f"{label}: no warm-start traces matching {trace_prefix}* in any of "
            + ", ".join(str(d) for d in RUN_DIRS)
            + f".\nWithout them this row would silently become a cold-start "
              f"AssayFormer run.\n{config.PUBLISHED_HINT}"
        )
    return traces

# (label, model_name, acq, model_overrides, ranker_checkpoint, sweep_id_suffix)
HANDOFF_RANKER = "gf-bpmf-train-hits-rl-fg-s19"
HANDOFF_CKPT_FILE = "model_last.pt"
# RL runs read from model_last.pt (the final GRPO epoch) rather than model.pt
# (the best-on-validation epoch), for two different reasons:
#
#   HANDOFF_RANKER -- deliberate. The paper reports the final policy, so every
#     table row and figure that uses this ranker loads epoch 99. Its model.pt is
#     epoch 92 (val NVR 6.544 vs 6.479 at epoch 99) and is a genuinely different
#     set of weights; loading it would silently shift every AssayLoop number.
#   gf-random-train-d10-rl -- forced. Its model.pt is a byte-for-byte copy of
#     the supervised init (verified by hash), because no epoch beat the init on
#     validation, so loading it would reproduce the supervised run instead of
#     the GRPO one. model_last.pt holds the actual RL model.
#
# Other RL ablations carry RL weights in model.pt and are left alone.
_RL_MODEL_LAST = {HANDOFF_RANKER, "gf-random-train-d10-rl"}

METHODS = [
    ("Prior hit baseline", "screen_knn", "greedy", {"prior_only": True}, None, "fg-prior-hit"),
    ("kNN baseline", "screen_knn", "greedy", {}, None, "fg-knn"),
    ("RF (greedy)", "rf", "greedy", {}, None, "fg-rf-greedy"),
    ("RF + UCB", "rf", "ucb", {}, None, "fg-rf-ucb"),
    (r"BPMF~\cite{}", "bpmf", "greedy", {}, None, "fg-bpmf"),
    ("Transformer", None, "greedy", {}, "gf-random-train-d10", "fg-transformer"),
    (r"\quad + BPMF Embeddings", None, "greedy", {}, "gf-bpmf-train-hits", "fg-transformer-bpmf"),
    (r"\quad + GRPO (= AssayLoop)", None, "greedy", {}, HANDOFF_RANKER, "fg-assayloop-s19"),
    (r"Transformer + DAgger~\cite{}", None, "greedy", {}, "ranker-dagger", "fg-dagger"),
    (r"MAML (+BPMF)~\cite{}", None, "greedy", {}, "maml-bpmf-d10", "fg-maml"),
    # --- Ablation: supervised ---
    ("Transformer (Random)", None, "greedy", {}, "gf-random-train-d10", "fg-ablation-random"),
    ("Transformer (BPMF)", None, "greedy", {}, "gf-bpmf-train-hits", "fg-ablation-bpmf"),
    (r"\quad - screen desc.", None, "greedy", {}, "scaling-L-k10-bpmf-d1349-s0", "fg-ablation-nodesc"),
    ("Transformer (SVD)", None, "greedy", {}, "gf-svd-raw-train-d10", "fg-ablation-svd-raw"),
    ("Transformer (SVD-Sphere)", None, "greedy", {}, "gf-svd-train-d10", "fg-ablation-svd-sphere"),
    ("Transformer (MF)", None, "greedy", {}, "gf-mf-raw-train-d10", "fg-ablation-mf-raw"),
    ("Transformer (MF-Sphere)", None, "greedy", {}, "gf-mf-train-d10", "fg-ablation-mf-sphere"),
    ("Transformer (GenePT-PCA)", None, "greedy", {}, "gf-genept-train-d10", "fg-ablation-genept"),
    ("Transformer (K562-PCA)", None, "greedy", {}, "gf-k562-train-d10", "fg-ablation-k562"),
    # --- Ablation: + GRPO ---
    (r"Random\hfill + GRPO", None, "greedy", {}, "gf-random-train-d10-rl", "fg-ablation-random-rl-last"),
    (r"BPMF\hfill + GRPO", None, "greedy", {}, HANDOFF_RANKER, "fg-ablation-bpmf-rl"),
    (r"BPMF (no desc.)\hfill + GRPO", None, "greedy", {}, "scaling-L-k10-bpmf-d1349-s0-rl", "fg-ablation-nodesc-rl"),
    (r"BPMF\hfill + GRPO (NVR$_{\text{terminal}}$)", None, "greedy", {}, "gf-bpmf-train-hits-nvr_terminal-rl", "fg-ablation-nvr-terminal"),
    (r"SVD\hfill + GRPO", None, "greedy", {}, "gf-svd-raw-train-d10-rl", "fg-ablation-svd-raw-rl"),
    (r"SVD-Sphere\hfill + GRPO", None, "greedy", {}, "gf-svd-train-d10-rl", "fg-ablation-svd-sphere-rl"),
    (r"MF\hfill + GRPO", None, "greedy", {}, "gf-mf-raw-train-d10-rl", "fg-ablation-mf-raw-rl"),
    (r"MF-Sphere\hfill + GRPO", None, "greedy", {}, "gf-mf-train-d10-rl", "fg-ablation-mf-sphere-rl"),
    (r"GenePT-PCA\hfill + GRPO", None, "greedy", {}, "gf-genept-train-d10-rl", "fg-ablation-genept-rl"),
    (r"K562-PCA\hfill + GRPO", None, "greedy", {}, "gf-k562-train-d10-rl", "fg-ablation-k562-rl"),
]

# Per-method warm-start rounds. Gemini and GPT-5.5 use n=3 (new -n3 sweep_ids,
# so they recompute); GLM, AssayLLM and Joint GRPO stay at n=2 with their
# original sweep_ids (reuse existing cached runs).
# (label, trace_sweep_prefix, ranker_checkpoint, n_warm, sweep_id_suffix)
HANDOFF_METHODS = [
    ("GLM-5.1 - AssayLoop Handoff", "sweep-e0dd3203-", HANDOFF_RANKER, 2, "fg-handoff-glm-s19"),
    ("Gemini-3.1-Pro - AssayLoop Handoff", "sweep-a79fd5ce-", HANDOFF_RANKER, 3, "fg-handoff-gemini-s19-n3"),
    ("GPT-5.5 - AssayLoop Handoff", "sweep-3aa3df2d-", HANDOFF_RANKER, 3, "fg-handoff-gpt55-s19-n3"),
    ("GPT-5.6 Sol - AssayLoop Handoff", "sweep-07e318e9-", HANDOFF_RANKER, 3, "fg-handoff-gpt56sol-s19-n3"),
]

# Handoff from an LLM warm start to a raw BPMF model (instead of the AssayLoop
# transformer ranker). Currently unused; kept for optional re-enabling.
# (label, trace_sweep_prefix, bpmf_root, K, n_warm, sweep_id_suffix)
BPMF_HANDOFF_METHODS = []

# Fine-tuned-LLM prediction dumps; set ASSAYLOOP_LLM_PREDICTIONS.
QWEN_REPARSE = config.LLM_PREDICTIONS_PATH

# Handoff from JSONL traces (finetuned LLM predictions -> transformer)
# (label, jsonl_path, ranker_checkpoint, n_warm, sweep_id_suffix)
JSONL_HANDOFF_METHODS = [
    ("AssayLLM - AssayLoop Handoff",
     QWEN_REPARSE / "sft_grpo_test_predictions.jsonl",
     HANDOFF_RANKER, 2, "fg-handoff-assayllm-s19"),
    ("AssayLLM-AssayLoop Handoff GRPO",
     QWEN_REPARSE / "handoff_best_handoff_k2_gf-bpmf-train-hits-rl_predictions.jsonl",
     "gf-bpmf-train-hits-rl", 2, "fg-handoff-joint-grpo"),
]

# LLM rows: recalculate NVR from existing sweep results using the universe denominator.
LLM_METHODS = [
    ("GLM-5.1", ("tag+acq", "GLM-5.1", "llm_single")),
    (r"\quad - hit labels", ("tag+acq", "GLM-5.1", "llm_single_blind")),
    ("GLM-5.2", ("tag+acq", "GLM-5.2", "llm_single")),
    ("Kimi-K2.6", ("tag+acq", "Kimi-K2.6", "llm_single")),
    (r"\quad - hit labels [Kimi]", ("sweep_id", "sweep-f132664e")),
    ("Claude Opus-4.8", ("tag+acq", "claude-opus-4.8", "llm_single")),
    (r"\quad - hit labels [Opus]", ("sweep_id", "sweep-29fd32fe")),
    ("Claude Sonnet-4.6", ("tag+acq", "claude-sonnet-4-6", "llm_single")),
    ("Claude Haiku-4.5", ("tag+acq", "claude-haiku-4.5-20251001", "llm_single")),
    ("Qwen3.6-27B", ("tag+acq", "qwen3.6-27b", "llm_single")),
    (r"\quad - hit labels [Qwen]", ("sweep_id", "sweep-6b384ce9")),
    ("Gemini-3.1-pro", ("sweep_id", "sweep-a79fd5ce")),
    (r"\quad - hit labels [Gemini]", ("sweep_id", "sweep-cd32999e")),
    ("GPT-5.5", ("sweep_id", "sweep-3aa3df2d")),
    ("GPT-5.6 Sol", ("sweep_id", "sweep-07e318e9")),
    (r"ICBR-EF~\cite{}", ("sweep_id", "sweep-210a554b")),
    (r"LLMNN~\cite{}", ("sweep_id", "sweep-f2fe727f")),
    (r"Haiku-4.5 Agent", ("sweep_id", "sweep-04b370cb")),
]

# Baselines supplied as raw sweep IDs whose runs acquire from the f2 universe
# (n2 > 0), so they use the SAME domain-adjusted metrics as AssayLoop (forgiving
# out-of-library-but-in-universe picks) rather than the full-budget LLM formula.
# (label, sweep_id) -- result.json read from output/runs or the shared runs dir.
RAW_SWEEP_METHODS = [
    (r"BioBO~\cite{}", "sweep-f3234ddc"),
    (r"Haystacks~\cite{}", "sweep-f9ae7231"),
]

# (label, path, format)  format: "llm" or "handoff"
JSONL_METHODS = [
    ("Qwen3.6-27B (base)", QWEN_REPARSE / "base_model_test_predictions.jsonl", "llm"),
    (r"\quad + SFT (GLM-5.1 traces)", QWEN_REPARSE / "sft_test_predictions.jsonl", "llm"),
    (r"\quad + SFT + GRPO (= AssayLLM)", QWEN_REPARSE / "sft_grpo_test_predictions.jsonl", "llm"),
]

# Rebrand: rendered label per internal key (keys/filters stay as-is above).
# Transformer model -> AssayFormer; the handoff framework -> AssayLoop.
# External baselines (Transformer + DAgger, generic LLM sections) unchanged.
_DISPLAY = {
    # Citation keys live here, not in the METHODS labels above: those strings are
    # the `results` dict keys and are matched literally by _BASE_LLM_SKIP and the
    # section emitters, so editing them there would silently break row lookup.
    r"BPMF~\cite{}": r"BPMF~\cite{Salakhutdinov2008-ad}",
    r"Transformer + DAgger~\cite{}": r"Transformer + DAgger~\cite{Ross2011-wr}",
    r"MAML (+BPMF)~\cite{}":
        r"MAML (w/ BPMF embs.)~\cite{finn2017model,Salakhutdinov2008-ad}",
    r"BioBO~\cite{}": r"BioBO~\cite{li2025biobo}",
    r"Haystacks~\cite{}": r"Haystacks~\cite{Rubbi2026-oo}",
    r"LLMNN~\cite{}": r"LLMNN~\cite{gupta2025llms}",
    r"ICBR-EF~\cite{}": r"ICBR-EF~\cite{Wainrib2026-rs}",
    r"BPMF\hfill + GRPO (NVR$_{\text{terminal}}$)":
        r"BPMF\hfill + GRPO (EF$_{\text{terminal}}$)",
    "Transformer": "AssayFormer (random embs.)",
    r"\quad + GRPO (= AssayLoop)": r"\quad + GRPO (= AssayFormer)",
    "GLM-5.1 - AssayLoop Handoff": r"GLM-5.1 $\rightarrow$ AssayFormer",
    "Gemini-3.1-Pro - AssayLoop Handoff": r"Gemini-3.1-Pro $\rightarrow$ AssayFormer",
    "GPT-5.5 - AssayLoop Handoff": r"GPT-5.5 $\rightarrow$ AssayFormer",
    "GPT-5.6 Sol - AssayLoop Handoff": r"GPT-5.6 Sol $\rightarrow$ AssayFormer",
    "AssayLLM - AssayLoop Handoff": r"AssayLLM $\rightarrow$ AssayFormer",
    "AssayLLM-AssayLoop Handoff GRPO": r"Joint AssayLLM-AssayFormer GRPO",
    "Transformer (Random)": "AssayFormer (Random)",
    "Transformer (BPMF)": "AssayFormer (BPMF)",
    "Transformer (SVD)": "AssayFormer (SVD)",
    "Transformer (SVD-Sphere)": "AssayFormer (SVD-Sphere)",
    "Transformer (MF)": "AssayFormer (MF)",
    "Transformer (MF-Sphere)": "AssayFormer (MF-Sphere)",
    "Transformer (GenePT-PCA)": "AssayFormer (GenePT-PCA)",
    "Transformer (K562-PCA)": "AssayFormer (K562-PCA)",
}


def _load_essentials():
    """DepMap common essentials, from the copy bundled with assaybench.

    This used to read a CSV at the repo root and return an empty set when the
    file was absent. An empty set does not fail: it makes the non-essential
    columns below silently identical to the plain ones, which is a wrong table
    rather than a missing one. The packaged loader raises instead.
    """
    return set(load_common_essentials())


def _noness_nvr_from_run(result_path, screen_genes, essentials_set,
                         total_hits_noness, domain_size_noness):
    """NVR excluding essential genes (option B: essentials don't exist).

    Ignores essential genes entirely: they don't count as picks, hits,
    or candidates. Measures ability to find screen-specific hits.
    """
    rd = json.loads(result_path.read_text())
    hits_noness = 0
    picks_noness = 0
    for step in rd.get("steps", []):
        for g, h in zip(step.get("acquired_batch", []),
                        step.get("hits", [])):
            if g in essentials_set:
                continue
            if g in screen_genes:
                picks_noness += 1
                if h:
                    hits_noness += 1
    hits_ess = 0
    rd = json.loads(result_path.read_text())
    for step in rd.get("steps", []):
        for g, h in zip(step.get("acquired_batch", []),
                        step.get("hits", [])):
            if g in essentials_set and g in screen_genes and h:
                hits_ess += 1
    total_found = hits_noness + hits_ess
    pct_ess = hits_ess / total_found if total_found > 0 else 0

    if total_hits_noness <= 0 or picks_noness <= 0:
        return 0.0, 0.0, float(pct_ess)
    rand_exp = picks_noness * total_hits_noness / domain_size_noness
    nvr = hits_noness / rand_exp if rand_exp > 0 else 0
    frac = hits_noness / total_hits_noness
    return float(nvr), float(frac), float(pct_ess)


def _noness_nvr_runs(sweep_id, screens, essentials_set, noness_by_screen):
    """Mean non-essential NVR, frac_hits, and pct_essential across screens."""
    nvrs, fracs, pcts = [], [], []
    for i, screen in enumerate(screens):
        run_id = "%s-%02d-%s" % (sweep_id, i, screen.dataset_name)
        fp = _run_result(run_id)
        if fp is None:
            continue
        ne = noness_by_screen.get(screen.dataset_name, {})
        total_hits_ne = ne.get("total_hits", 0)
        domain_size_ne = ne.get("domain_size", 1)
        nvr, frac, pct = _noness_nvr_from_run(
            fp, set(screen.genes), essentials_set,
            total_hits_ne, domain_size_ne)
        nvrs.append(nvr)
        fracs.append(frac)
        pcts.append(pct)
    if not nvrs:
        return 0.0, 0.0, 0.0
    return float(np.mean(nvrs)), float(np.mean(fracs)), float(np.mean(pcts))


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_ranker_model(run_name: str, ckpt_file: str = "model.pt"):
    ckpt_dir = RANKERS_DIR / run_name
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    if "arch" not in cfg:
        from assayloop.models.maml_ranker import MAMLRankerModel
        return MAMLRankerModel(checkpoint=str(ckpt_dir), device=_default_device())
    from assayloop.models.amortized_ranker import AmortizedRankerModel
    return AmortizedRankerModel(checkpoint=str(ckpt_dir),
                                ckpt_file=ckpt_file, device=_default_device())


def latex_val(m, d=2):
    if m is None:
        return "-"
    return f"${m:.{d}f}$"


def pct_val(m, d=1):
    """Format a [0, 1] metric as a percentage number (bare; the '%' lives in the
    column header). Returns '-' if None."""
    if m is None:
        return "-"
    return f"${m * 100:.{d}f}$"


def _adj_nvr_from_run(result_path, screen_genes, universe_set,
                      total_hits, domain_size, budget=1000):
    """Domain-adjusted NVR from a result.json.

    Classifies each pick:
      n1 = in-domain (gene in screen library)
      n2 = out-of-domain, in-universe (forgiven)
      n3 = out-of-universe or unfilled (penalized)

    NVR = hits / ((n1 + n3) * H / D)
    """
    from assayloop.metrics.hits_auc import adjusted_nvr_value
    rd = json.loads(result_path.read_text())
    n1 = n2 = hits = 0
    for step in rd.get("steps", []):
        for g, h in zip(step.get("acquired_batch", []),
                        step.get("hits", [])):
            if g in screen_genes:
                n1 += 1
                if h:
                    hits += 1
            elif g in universe_set:
                n2 += 1
    # single source of truth for the NVR arithmetic (shared with the RL evaluator)
    return adjusted_nvr_value(hits, n1, n2, domain_size, total_hits, budget)


def _adj_nvr_runs(sweep_id, screens, universe_set, budget=1000):
    """Compute mean domain-adjusted NVR across all screens for a sweep."""
    nvrs = []
    for i, screen in enumerate(screens):
        run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
        fp = _run_result(run_id)
        if fp is None:
            continue
        lib = set(screen.genes)
        total_hits = sum(screen.hits)
        nvrs.append(_adj_nvr_from_run(fp, lib, universe_set, total_hits,
                                       len(lib), budget))
    return float(np.mean(nvrs)) if nvrs else 0.0


def _adj_nauc_from_run(result_path, screen_genes, universe_set, total_hits,
                       domain_size, budget=1000):
    """Domain-adjusted normalized AUC — the trajectory analogue of
    ``_adj_nvr_from_run``.

    The cumulative in-library-hit curve is built on the *effective* budget axis
    with the screen library as the domain (matching EF):
      - in-library picks advance the budget by one (and add a hit if a hit),
      - out-of-library-but-in-universe picks are *forgiven* (skipped: no budget,
        no hit),
      - out-of-universe (hallucinated) picks are charged (advance the budget,
        no hit),
      - unfilled budget is NOT charged: the curve stops at the last acquired
        point, so nAUC scores ordering quality over what was committed. The
        under-supply penalty lives in EF and the Shortfall column instead.

    PER-BATCH: the curve is sampled once per acquisition batch (not per gene), so
    it is INVARIANT to the order of genes within a batch. This is the right
    granularity because the policies (esp. LLMs) are not asked to rank genes
    within a batch. Returned value is ``auc / best`` where ``best`` is the optimal
    curve integrated on the SAME per-batch x-grid.
    """
    if total_hits <= 0 or domain_size <= 0:
        return 0.0
    rd = json.loads(result_path.read_text())
    eff = 0      # charged picks so far (n1 + out-of-universe)
    cum = 0      # in-library hits so far
    n2 = 0       # forgiven picks
    xs = [0.0]
    ys = [0.0]
    for step in rd.get("steps", []):
        for g, h in zip(step.get("acquired_batch", []), step.get("hits", [])):
            if g in screen_genes:
                eff += 1
                if h:
                    cum += 1
            elif g in universe_set:
                n2 += 1                       # forgiven: no budget, no hit
            else:
                eff += 1                      # out-of-universe: charged, no hit
        # one curve point per batch -> intra-batch gene order does not matter
        xs.append(eff / domain_size)
        ys.append(cum / total_hits)
    # nAUC measures ordering quality over the CHARGED-ACQUIRED domain only
    # (x ends at eff/|L| = (N_L + N_Ū)/|L|). Unfilled budget is NOT charged
    # here -- that under-supply penalty already lives in EF (via its eff_budget)
    # and in the explicit Shortfall column, so charging it a third time in nAUC
    # would just move all three columns together on one deficiency. `budget` is
    # kept in the signature for call-site compatibility but no longer used.
    auc = float(np.trapezoid(ys, xs))
    # Batch-matched oracle: integrate the optimal curve on the SAME per-batch
    # x-grid (linear interpolation within a batch), so numerator and denominator
    # use an identical granularity/convention rather than mixing a per-batch
    # numerator with a continuous closed-form oracle.
    ys_best = [min(x * domain_size / total_hits, 1.0) for x in xs]
    best = float(np.trapezoid(ys_best, xs))
    return float(auc / best) if best > 0 else 0.0


def _adj_nauc_runs(sweep_id, screens, universe_set, budget=1000):
    """Mean domain-adjusted normalized AUC across all screens for a sweep."""
    naucs = []
    for i, screen in enumerate(screens):
        run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
        fp = _run_result(run_id)
        if fp is None:
            continue
        lib = set(screen.genes)
        naucs.append(_adj_nauc_from_run(fp, lib, universe_set, sum(screen.hits),
                                        len(lib), budget))
    return float(np.mean(naucs)) if naucs else 0.0


def _round_submitted(r):
    """Genes submitted/acquired in one JSONL round (finetuned-LLM uses
    ``submitted``; the handoff format uses ``acquired``)."""
    if isinstance(r, dict):
        return r.get("submitted") or r.get("acquired") or []
    return r if isinstance(r, list) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-screen-freq", type=int, default=0,
                    help="Only include genes appearing in >= N test screens. "
                         "0 = no filter (full 22k universe). "
                         "2 = removes pseudogenes/antisense (~21k).")
    args = ap.parse_args()

    screens = load_screens(target_set="public")
    all_genes = {g for s in screens for g in s.genes}

    if args.min_screen_freq > 0:
        gene_freq = Counter()
        for s in screens:
            for g in set(s.genes):
                gene_freq[g] += 1
        universe = sorted(g for g in all_genes if gene_freq[g] >= args.min_screen_freq)
        n_dropped = len(all_genes) - len(universe)
        log.info("Filtered universe: %d genes (dropped %d with freq < %d), %d screens",
                 len(universe), n_dropped, args.min_screen_freq, len(screens))
        sweep_tag = f"fg-f{args.min_screen_freq}"
        out_file = ROOT / "output" / "tables" / f"full_genome_baselines_f{args.min_screen_freq}.tex"
    else:
        universe = sorted(all_genes)
        log.info("Full genome: %d genes, %d screens", len(universe), len(screens))
        sweep_tag = "fg"
        out_file = OUTPUT_FILE

    universe_set = set(universe)
    BUDGET = 1000
    essentials = _load_essentials()
    log.info("Essential genes: %d loaded", len(essentials))

    # Per-screen non-essential stats
    noness_by_screen = {}
    for s in screens:
        lib = set(s.genes)
        lib_noness = lib - essentials
        hits_noness = sum(1 for g, h in zip(s.genes, s.hits) if h and g not in essentials)
        noness_by_screen[s.dataset_name] = {
            "total_hits": hits_noness,
            "domain_size": len(lib_noness),
        }

    base_cfg = RunConfig(
        screen_set="public",
        model="null",
        acq="greedy",
        batch_size=100,
        n_steps=10,
        persist=True,
        metrics=["hits_auc"],
        max_shortfall_frac=1.0,
        universe_genes=universe,
    )

    lib_by_name = {s.dataset_name: set(s.genes) for s in screens}

    from assayloop.metrics.batch_diversity import BatchDiversity
    log.info("Loading BatchDiversity scorer...")
    scorer = BatchDiversity()
    scorer._ensure_provider()
    scorer._ensure_gene_sets()
    log.info("  done.")

    def _compute_diversity_from_runs(run_paths):
        vendi_per_run, pathway_per_run = [], []
        screen_batches = []
        for fp in run_paths:
            try:
                r = json.loads(fp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            v_steps, p_steps, batches = [], [], []
            for step in r.get("steps", []):
                batch = step.get("acquired_batch")
                if not batch or len(batch) < 2:
                    continue
                batches.append(batch)
                scores = scorer.score([], None, None, [], acquired_batch=batch)
                if "batch_vendi_ratio" in scores:
                    v_steps.append(scores["batch_vendi_ratio"])
                if "batch_pathway_overlap_vs_random" in scores:
                    p_steps.append(scores["batch_pathway_overlap_vs_random"])
            if batches:
                screen_batches.append(batches)
            if v_steps:
                vendi_per_run.append(float(np.mean(v_steps)))
            if p_steps:
                pathway_per_run.append(float(np.mean(p_steps)))
        out = {}
        if vendi_per_run:
            out["vendi"] = float(np.mean(vendi_per_run))
        if pathway_per_run:
            out["pathway"] = float(np.mean(pathway_per_run))
        if screen_batches:
            out.update(effective_pathways(screen_batches))
        return out

    def _oob_frac_from_runs(run_paths):
        fracs = []
        for fp in run_paths:
            try:
                r = json.loads(fp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            lib = None
            for name, genes in lib_by_name.items():
                if name in fp.parent.name:
                    lib = genes
                    break
            if lib is None:
                continue
            total_oob, total_picks = 0, 0
            for step in r.get("steps", []):
                batch = step.get("acquired_batch", [])
                total_picks += len(batch)
                total_oob += sum(1 for g in batch if g not in lib)
            if total_picks > 0:
                fracs.append(total_oob / total_picks)
        return float(np.mean(fracs)) if fracs else 0.0

    results = {}

    for label, model_name, acq, model_overrides, ranker_name, sweep_suffix in METHODS:
        log.info("=== %s ===", label)
        sweep_id = f"sweep-{sweep_tag}-{sweep_suffix}" if sweep_tag != "fg" else f"sweep-{sweep_suffix}"
        nvrs, naucs, fracs, shortfalls = [], [], [], []

        if ranker_name is not None:
            ckpt = HANDOFF_CKPT_FILE if ranker_name in _RL_MODEL_LAST else "model.pt"
            ckpt_path = RANKERS_DIR / ranker_name / ckpt
            if not ckpt_path.exists():
                log.warning("  checkpoint not found: %s (skipping)", ckpt_path)
                results[label] = {"nvr": None, "nauc": None, "frac": None,
                                  "shortfall": None, "vendi": None, "pathway": None}
                continue
            model_obj = _load_ranker_model(ranker_name, ckpt_file=ckpt)
        elif model_name:
            model_obj = make_model(model_name, overrides=model_overrides)
        else:
            model_obj = None

        cfg = replace(
            base_cfg,
            model=model_name or "null",
            acq=acq,
            model_overrides=model_overrides,
        )

        for i, screen in enumerate(screens):
            run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
            cached = _run_result(run_id)
            if cached is not None:
                log.info("  [%d/%d] %s (cached)", i + 1, len(screens), screen.dataset_name)
                rd = json.loads(cached.read_text())
                fm = rd.get("final_metrics", {})
            else:
                log.info("  [%d/%d] %s", i + 1, len(screens), screen.dataset_name)
                res = run_one_screen(
                    screen, cfg, model_obj=model_obj, verbose=False,
                    run_id=run_id, sweep_id=sweep_id,
                )
                fm = res.final_metrics or {}

            nvrs.append(fm.get("n_hits_vs_random", 0))
            fracs.append(fm.get("frac_hits", 0))
            shortfalls.append(fm.get("shortfall_frac", 0))
            ha = fm.get("hits_auc", 0)
            hab = fm.get("hits_auc_best", 1)
            naucs.append(ha / hab if hab > 0 else 0)

        # Diversity + OOB shortfall + adjusted NVR from cached runs
        run_paths = _sweep_run_paths(sweep_id)
        div = _compute_diversity_from_runs(run_paths)
        oob = _oob_frac_from_runs(run_paths)
        nvr_adj = _adj_nvr_runs(sweep_id, screens, universe_set, BUDGET)
        nauc_adj = _adj_nauc_runs(sweep_id, screens, universe_set, BUDGET)
        nvr_ne, frac_ne, pct_ess = _noness_nvr_runs(sweep_id, screens, essentials, noness_by_screen)

        results[label] = {
            "nvr": nvr_adj,
            "nauc": nauc_adj,
            "frac": float(np.mean(fracs)),
            "shortfall": oob,
            "vendi": div.get("vendi"),
            "pathway": div.get("pathway"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": nvr_ne,
            "frac_ne": frac_ne,
            "pct_ess": pct_ess,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f  vendi=%s  pathway=%s",
                 results[label]["nvr"], results[label]["nauc"],
                 results[label]["frac"], results[label]["shortfall"],
                 f'{div.get("vendi", 0):.2f}', f'{div.get("pathway", 0):.2f}')

        if model_obj is not None:
            del model_obj

    # --- Baselines given as raw sweep IDs (universe pool -> domain-adjusted) ---
    def _raw_sweep_row(sweep_id, label):
        _require_runs(sweep_id, label, len(screens))
        run_paths, nvrs, naucs, fracs, pcts = [], [], [], [], []
        for i, screen in enumerate(screens):
            fp = _run_result(f"{sweep_id}-{i:02d}-{screen.dataset_name}")
            if fp is None:
                continue
            run_paths.append(fp)
            lib = set(screen.genes)
            H, D = sum(screen.hits), len(screen.genes)
            nvrs.append(_adj_nvr_from_run(fp, lib, universe_set, H, D, BUDGET))
            naucs.append(_adj_nauc_from_run(fp, lib, universe_set, H, D, BUDGET))
            fm = json.loads(fp.read_text()).get("final_metrics", {})
            if "frac_hits" in fm:
                fracs.append(fm["frac_hits"])
            ne = noness_by_screen.get(screen.dataset_name, {})
            if ne.get("total_hits", 0) > 0:
                _, _, pct = _noness_nvr_from_run(fp, lib, essentials,
                                                 ne["total_hits"], ne["domain_size"])
                pcts.append(pct)
        if not run_paths:
            return {"nvr": None, "nauc": None, "frac": None, "shortfall": None,
                    "vendi": None, "pathway": None, "pct_ess": None}
        div = _compute_diversity_from_runs(run_paths)
        return {
            "nvr": float(np.mean(nvrs)) if nvrs else None,
            "nauc": float(np.mean(naucs)) if naucs else None,
            "frac": float(np.mean(fracs)) if fracs else None,
            "shortfall": _oob_frac_from_runs(run_paths),
            "vendi": div.get("vendi"), "pathway": div.get("pathway"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "pct_ess": float(np.mean(pcts)) if pcts else None,
        }

    for label, sweep_id in RAW_SWEEP_METHODS:
        log.info("=== %s (raw sweep %s) ===", label, sweep_id)
        results[label] = _raw_sweep_row(sweep_id, label)
        r = results[label]
        log.info("  NVR=%s  nAUC=%s  frac=%s  shortfall=%s",
                 r["nvr"], r["nauc"], r["frac"], r["shortfall"])

    # --- Handoff methods ---
    from assayloop.acquisitions.glm_handoff_acq import GlmHandoffAcquisition
    from assayloop.acquisitions.greedy_from_model import GreedyFromModel

    for label, trace_prefix, ranker_name, n_warm, sweep_suffix in HANDOFF_METHODS:
        log.info("=== %s ===", label)
        sweep_id = f"sweep-{sweep_tag}-{sweep_suffix}" if sweep_tag != "fg" else f"sweep-{sweep_suffix}"
        nvrs, naucs, fracs, shortfalls = [], [], [], []

        # Load traces from LLM sweep runs
        traces = _load_traces(trace_prefix, label)

        model_obj = _load_ranker_model(ranker_name, ckpt_file=HANDOFF_CKPT_FILE)
        cfg = replace(base_cfg, model="null", acq="greedy")

        for i, screen in enumerate(screens):
            run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
            cached = _run_result(run_id)
            if cached is not None:
                log.info("  [%d/%d] %s (cached)", i + 1, len(screens), screen.dataset_name)
                rd_data = json.loads(cached.read_text())
                fm = rd_data.get("final_metrics", {})
            else:
                log.info("  [%d/%d] %s", i + 1, len(screens), screen.dataset_name)
                screen_traces = traces.get(screen.dataset_name, [])
                rounds = screen_traces[0] if screen_traces else []
                acq_obj = GlmHandoffAcquisition(
                    rounds=rounds,
                    n_warm=n_warm if rounds else 0,
                    base=GreedyFromModel(seed=0),
                )
                res = run_one_screen(
                    screen, cfg, model_obj=model_obj, acq_obj=acq_obj,
                    verbose=False, run_id=run_id, sweep_id=sweep_id,
                )
                fm = res.final_metrics or {}

            nvrs.append(fm.get("n_hits_vs_random", 0))
            fracs.append(fm.get("frac_hits", 0))
            shortfalls.append(fm.get("shortfall_frac", 0))
            ha = fm.get("hits_auc", 0)
            hab = fm.get("hits_auc_best", 1)
            naucs.append(ha / hab if hab > 0 else 0)

        run_paths = _sweep_run_paths(sweep_id)
        div = _compute_diversity_from_runs(run_paths)
        oob = _oob_frac_from_runs(run_paths)
        nvr_adj = _adj_nvr_runs(sweep_id, screens, universe_set, BUDGET)
        nauc_adj = _adj_nauc_runs(sweep_id, screens, universe_set, BUDGET)
        nvr_ne, frac_ne, pct_ess = _noness_nvr_runs(sweep_id, screens, essentials, noness_by_screen)

        results[label] = {
            "nvr": nvr_adj,
            "nauc": nauc_adj,
            "frac": float(np.mean(fracs)),
            "shortfall": oob,
            "vendi": div.get("vendi"),
            "pathway": div.get("pathway"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": nvr_ne,
            "frac_ne": frac_ne,
            "pct_ess": pct_ess,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f",
                 results[label]["nvr"], results[label]["nauc"],
                 results[label]["frac"], results[label]["shortfall"])

        del model_obj

    # --- LLM -> raw-BPMF handoff methods ---
    import glob as _glob
    for label, trace_prefix, bpmf_root, bpmf_k, n_warm, sweep_suffix in BPMF_HANDOFF_METHODS:
        log.info("=== %s ===", label)
        sweep_id = f"sweep-{sweep_tag}-{sweep_suffix}" if sweep_tag != "fg" else f"sweep-{sweep_suffix}"
        pkls = sorted(_glob.glob(
            f"{bpmf_root}/bpmf_public_train_K{bpmf_k}_su1_sv1_*/bpmf_result.pkl"))
        if not pkls:
            log.warning("  no BPMF K%d pkl in %s; skipping", bpmf_k, bpmf_root)
            results[label] = {kk: None for kk in ("nvr", "nauc", "frac", "shortfall",
                                                  "vendi", "pathway", "nvr_ne",
                                                  "frac_ne", "pct_ess")}
            continue
        traces = _load_traces(trace_prefix, label)
        model_obj = make_model("bpmf", overrides={"checkpoint_path": pkls[-1]})
        cfg = replace(base_cfg, model="null", acq="greedy")

        naucs, fracs = [], []
        for i, screen in enumerate(screens):
            run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
            cached = _run_result(run_id)
            if cached is not None:
                log.info("  [%d/%d] %s (cached)", i + 1, len(screens), screen.dataset_name)
                fm = json.loads(cached.read_text()).get("final_metrics", {})
            else:
                log.info("  [%d/%d] %s", i + 1, len(screens), screen.dataset_name)
                screen_traces = traces.get(screen.dataset_name, [])
                rounds = screen_traces[0] if screen_traces else []
                acq_obj = GlmHandoffAcquisition(
                    rounds=rounds, n_warm=n_warm if rounds else 0,
                    base=GreedyFromModel(seed=0))
                res = run_one_screen(screen, cfg, model_obj=model_obj, acq_obj=acq_obj,
                                     verbose=False, run_id=run_id, sweep_id=sweep_id)
                fm = res.final_metrics or {}
            fracs.append(fm.get("frac_hits", 0))
            ha = fm.get("hits_auc", 0); hab = fm.get("hits_auc_best", 1)
            naucs.append(ha / hab if hab > 0 else 0)

        run_paths = _sweep_run_paths(sweep_id)
        div = _compute_diversity_from_runs(run_paths)
        oob = _oob_frac_from_runs(run_paths)
        nvr_adj = _adj_nvr_runs(sweep_id, screens, universe_set, BUDGET)
        nauc_adj = _adj_nauc_runs(sweep_id, screens, universe_set, BUDGET)
        nvr_ne, frac_ne, pct_ess = _noness_nvr_runs(sweep_id, screens, essentials, noness_by_screen)
        results[label] = {
            "nvr": nvr_adj, "nauc": nauc_adj, "frac": float(np.mean(fracs)),
            "shortfall": oob, "vendi": div.get("vendi"), "pathway": div.get("pathway"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": nvr_ne, "frac_ne": frac_ne, "pct_ess": pct_ess,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f", nvr_adj,
                 results[label]["nauc"], results[label]["frac"], oob)
        del model_obj

    # --- JSONL-based handoff methods ---
    for label, jsonl_path, ranker_name, n_warm, sweep_suffix in JSONL_HANDOFF_METHODS:
        log.info("=== %s ===", label)
        sweep_id = f"sweep-{sweep_tag}-{sweep_suffix}" if sweep_tag != "fg" else f"sweep-{sweep_suffix}"
        nvrs, naucs, fracs, shortfalls = [], [], [], []

        with open(jsonl_path) as fh:
            jsonl_screens = [json.loads(line) for line in fh]
        jsonl_by_name = {s["dataset_name"]: s for s in jsonl_screens}

        model_obj = _load_ranker_model(ranker_name, ckpt_file=HANDOFF_CKPT_FILE)
        cfg = replace(base_cfg, model="null", acq="greedy")

        for i, screen in enumerate(screens):
            run_id = f"{sweep_id}-{i:02d}-{screen.dataset_name}"
            cached = _run_result(run_id)
            if cached is not None:
                log.info("  [%d/%d] %s (cached)", i + 1, len(screens), screen.dataset_name)
                rd_data = json.loads(cached.read_text())
                fm = rd_data.get("final_metrics", {})
            else:
                log.info("  [%d/%d] %s", i + 1, len(screens), screen.dataset_name)
                js = jsonl_by_name.get(screen.dataset_name, {})
                rg = js.get("round_genes", [])
                rounds = []
                for r in rg:
                    if isinstance(r, dict):
                        genes = r.get("new_hits", []) + r.get("new_misses", [])
                        if not genes:
                            genes = r.get("hits", []) + r.get("misses", [])
                        if not genes:
                            genes = r.get("acquired", [])
                        if not genes:
                            genes = r.get("submitted", [])
                        rounds.append(genes)
                acq_obj = GlmHandoffAcquisition(
                    rounds=rounds,
                    n_warm=n_warm if rounds else 0,
                    base=GreedyFromModel(seed=0),
                )
                res = run_one_screen(
                    screen, cfg, model_obj=model_obj, acq_obj=acq_obj,
                    verbose=False, run_id=run_id, sweep_id=sweep_id,
                )
                fm = res.final_metrics or {}

            nvrs.append(fm.get("n_hits_vs_random", 0))
            fracs.append(fm.get("frac_hits", 0))
            shortfalls.append(fm.get("shortfall_frac", 0))
            ha = fm.get("hits_auc", 0)
            hab = fm.get("hits_auc_best", 1)
            naucs.append(ha / hab if hab > 0 else 0)

        run_paths = _sweep_run_paths(sweep_id)
        div = _compute_diversity_from_runs(run_paths)
        oob = _oob_frac_from_runs(run_paths)
        nvr_adj = _adj_nvr_runs(sweep_id, screens, universe_set, BUDGET)
        nauc_adj = _adj_nauc_runs(sweep_id, screens, universe_set, BUDGET)
        nvr_ne, frac_ne, pct_ess = _noness_nvr_runs(sweep_id, screens, essentials, noness_by_screen)

        results[label] = {
            "nvr": nvr_adj,
            "nauc": nauc_adj,
            "frac": float(np.mean(fracs)),
            "shortfall": oob,
            "vendi": div.get("vendi"),
            "pathway": div.get("pathway"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": nvr_ne,
            "frac_ne": frac_ne,
            "pct_ess": pct_ess,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f",
                 results[label]["nvr"], results[label]["nauc"],
                 results[label]["frac"], results[label]["shortfall"])

        del model_obj

    # --- LLM methods: recalculate NVR from existing sweep results ---
    from assayloop.scripts import results_index
    from assayloop.scripts.results_index import _load_all_sweeps, load_sweep_index, resolve_row

    all_sweeps = _load_all_sweeps()
    index = load_sweep_index(all_sweeps)

    # Build screen name -> domain_size lookup
    domain_size_by_name = {s.dataset_name: len(s.genes) for s in screens}
    hits_by_name = {s.dataset_name: sum(s.hits) for s in screens}

    for label, lookup_key in LLM_METHODS:
        log.info("=== %s (recalc) ===", label)
        hit = resolve_row(lookup_key, index)
        if hit is None:
            # Not a warning-and-dash: every one of these rows is a published
            # number, and an unresolved lookup means the sweep bundle is not
            # in place, not that the method scored nothing.
            raise config.MissingConfiguredPath(
                f"{label}: no sweep matches {lookup_key!r}.\nSearched: "
                + ", ".join(str(d) for d in results_index.SWEEP_DIRS)
                + f"\n{config.PUBLISHED_HINT}"
            )

        _, _, sd = hit
        nvrs, naucs, fracs, shortfalls = [], [], [], []
        for ps in sd.get("per_screen", []):
            fm = ps.get("final_metrics") or {}
            n_hits = fm.get("n_hits", 0)
            total_hits = fm.get("total_hits", 1)
            screen_name = ps.get("screen_name", "")
            domain_size = domain_size_by_name.get(screen_name, 19000)
            # LLMs: all shortfall = n3 (out-of-universe), so eff_budget = BUDGET
            # NVR = h / (BUDGET * H / D)
            rand_expected = BUDGET * total_hits / domain_size
            nvr = n_hits / rand_expected if rand_expected > 0 else 0
            nvrs.append(nvr)
            fracs.append(fm.get("frac_hits", 0))
            shortfalls.append(fm.get("shortfall_frac", 0))
            ha = fm.get("hits_auc", 0)
            hab = fm.get("hits_auc_best", 1)
            naucs.append(ha / hab if hab > 0 else 0)

        # Diversity from original sweep runs
        from assayloop.scripts.results_index import get_run_ids, compute_diversity, _find_run_result
        rids = get_run_ids(sd)
        div = compute_diversity(rids, scorer) if rids else {}

        # Non-essential NVR + domain-adjusted nAUC from per-screen result.json.
        # (For library-pool LLMs n2=0 so this matches the raw nAUC; computing it
        # the same way keeps the whole nAUC column on one consistent basis.)
        nvrs_ne, fracs_ne, pcts_ess = [], [], []
        naucs_adj = []
        for ps in sd.get("per_screen", []):
            rid = ps.get("run_id", "")
            screen_name = ps.get("screen_name", "")
            fp = _find_run_result(rid) if rid else None
            ne = noness_by_screen.get(screen_name, {})
            if fp:
                lib = lib_by_name.get(screen_name, set())
                H = hits_by_name.get(screen_name, 0)
                D = domain_size_by_name.get(screen_name, len(lib) or 1)
                if H > 0 and D > 0:
                    naucs_adj.append(_adj_nauc_from_run(
                        fp, lib, universe_set, H, D, BUDGET))
                if ne.get("total_hits", 0) > 0:
                    nvr_n, frac_n, pct_n = _noness_nvr_from_run(
                        fp, lib, essentials, ne["total_hits"], ne["domain_size"])
                    nvrs_ne.append(nvr_n)
                    fracs_ne.append(frac_n)
                    pcts_ess.append(pct_n)

        results[label] = {
            "nvr": float(np.mean(nvrs)) if nvrs else None,
            "nauc": (float(np.mean(naucs_adj)) if naucs_adj
                     else (float(np.mean(naucs)) if naucs else None)),
            "frac": float(np.mean(fracs)) if fracs else None,
            "shortfall": float(np.mean(shortfalls)) if shortfalls else None,
            "vendi": div.get("vendi_mean"),
            "pathway": div.get("pathway_mean"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": float(np.mean(nvrs_ne)) if nvrs_ne else None,
            "frac_ne": float(np.mean(fracs_ne)) if fracs_ne else None,
            "pct_ess": float(np.mean(pcts_ess)) if pcts_ess else None,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f",
                 results[label]["nvr"] or 0, results[label]["nauc"] or 0,
                 results[label]["frac"] or 0, results[label]["shortfall"] or 0)

    # --- JSONL-based methods (finetuned LLMs, AssayLLM handoff) ---
    for label, jsonl_path, fmt in JSONL_METHODS:
        log.info("=== %s (jsonl) ===", label)
        if not jsonl_path.is_file():
            log.warning("  NOT FOUND: %s", jsonl_path)
            results[label] = {"nvr": None, "nauc": None, "frac": None, "shortfall": None, "nvr_ne": None, "frac_ne": None, "pct_ess": None}
            continue

        with open(jsonl_path) as fh:
            screens_data = [json.loads(line) for line in fh]

        nvrs, naucs, fracs, shortfalls = [], [], [], []
        for s in screens_data:
            total_hits = s.get("total_hits", 1)
            if fmt == "handoff":
                steps = s.get("per_step", [])
            else:
                steps = s.get("per_round", [])
            if not steps:
                continue
            cum_hits = steps[-1].get("cum_hits", 0)
            frac_hits = cum_hits / total_hits if total_hits > 0 else 0
            fracs.append(frac_hits)

            # Domain-adjusted NVR + nAUC: classify each submitted gene, forgiving
            # in-universe/out-of-library picks (n2) in BOTH the EF denominator and
            # the nAUC budget axis, matching _adj_nvr_from_run / _adj_nauc_from_run.
            screen_name = s.get("dataset_name", "")
            domain_size = domain_size_by_name.get(screen_name, s.get("library_size", 19000))
            screen_lib = lib_by_name.get(screen_name, set())
            rg = s.get("round_genes", [])
            n2_per_round = []
            n1 = n2 = 0
            for r in rg:
                r_n2 = 0
                for g in _round_submitted(r):
                    if g in screen_lib:
                        n1 += 1
                    elif g in universe_set:
                        n2 += 1
                        r_n2 += 1
                n2_per_round.append(r_n2)
            n3 = BUDGET - n1 - n2
            eff = n1 + n3
            rand_expected = eff * total_hits / domain_size if domain_size > 0 else 0
            nvr = cum_hits / rand_expected if rand_expected > 0 else 0
            nvrs.append(nvr)

            # nAUC on the effective (cumulative genes minus forgiven n2),
            # library domain, over the CHARGED-ACQUIRED domain only -- unfilled
            # budget is NOT charged here (that penalty lives in EF and the
            # Shortfall column), matching _adj_nauc_from_run.
            if total_hits > 0 and domain_size > 0:
                cum_n2 = 0
                xs = [0.0]
                ys = [0.0]
                for j, r in enumerate(steps):
                    cum_n2 += n2_per_round[j] if j < len(n2_per_round) else 0
                    eff_genes = max(0.0, r.get("cum_genes", 0) - cum_n2)
                    xs.append(eff_genes / domain_size)
                    ys.append(r.get("cum_hits", 0) / total_hits)
                auc = float(np.trapezoid(ys, xs))
                # batch-matched oracle on the same x-grid (see _adj_nauc_from_run)
                ys_best = [min(x * domain_size / total_hits, 1.0) for x in xs]
                best = float(np.trapezoid(ys_best, xs))
                naucs.append(auc / best if best > 0 else 0)
            else:
                naucs.append(0)

            # Shortfall: always against 100-gene budget per round
            sf_fracs = []
            for j, r in enumerate(steps):
                acquired = r.get("new_genes", 100)
                sf_fracs.append(max(0, 100 - acquired) / 100)
            shortfalls.append(float(np.mean(sf_fracs)) if sf_fracs else 0)

        # Diversity from JSONL batches
        from assayloop.scripts.results_index import compute_diversity_from_batches
        jsonl_batches = []
        for s in screens_data:
            rg = s.get("round_genes", [])
            if rg and isinstance(rg[0], dict):
                batches = []
                for r in rg:
                    in_lib = (r.get("new_hits", []) + r.get("new_misses", [])
                              or r.get("hits", []) + r.get("misses", []))
                    fallback = r.get("submitted", []) or r.get("acquired", [])
                    batches.append(in_lib if in_lib else fallback)
                jsonl_batches.append(batches)
            else:
                jsonl_batches.append([])
        div = compute_diversity_from_batches(jsonl_batches, scorer) if any(b for b in jsonl_batches) else {}

        # Non-essential NVR from JSONL per-round gene lists
        nvrs_ne, fracs_ne, pcts_ess = [], [], []
        for s in screens_data:
            screen_name = s.get("dataset_name", "")
            ne = noness_by_screen.get(screen_name, {})
            total_hits_ne = ne.get("total_hits", 0)
            domain_size_ne = ne.get("domain_size", 1)
            if total_hits_ne <= 0:
                continue
            hits_found_ne = 0
            hits_found_ess = 0
            picks_ne = 0
            rg = s.get("round_genes", [])
            for r in rg:
                if isinstance(r, dict):
                    for g in (r.get("new_hits", []) or r.get("hits", [])):
                        if isinstance(g, str):
                            if g in essentials:
                                hits_found_ess += 1
                            else:
                                hits_found_ne += 1
                                picks_ne += 1
                    for g in (r.get("new_misses", []) or r.get("misses", [])):
                        if isinstance(g, str) and g not in essentials:
                            picks_ne += 1
            total_found = hits_found_ne + hits_found_ess
            pct_e = hits_found_ess / total_found if total_found > 0 else 0
            pcts_ess.append(pct_e)
            if picks_ne > 0 and total_hits_ne > 0:
                rand_exp = picks_ne * total_hits_ne / domain_size_ne
                nvrs_ne.append(hits_found_ne / rand_exp if rand_exp > 0 else 0)
                fracs_ne.append(hits_found_ne / total_hits_ne)

        results[label] = {
            "nvr": float(np.mean(nvrs)) if nvrs else None,
            "nauc": float(np.mean(naucs)) if naucs else None,
            "frac": float(np.mean(fracs)) if fracs else None,
            "shortfall": float(np.mean(shortfalls)) if shortfalls else None,
            "vendi": div.get("vendi_mean"),
            "pathway": div.get("pathway_mean"),
            "ep_b": div.get("ep_batch"),
            "ep_s": div.get("ep_screen"),
            "ep_d": div.get("ep_dataset"),
            "ep_ann": div.get("ep_n_annotated"),
            "ep_drop": div.get("ep_n_dropped"),
            "ep_bmin": div.get("ep_batch_ann_min"),
            "ep_bp5": div.get("ep_batch_ann_p5"),
            "ep_pmed": div.get("ep_batch_pick_med"),
            "ep_pp5": div.get("ep_batch_pick_p5"),
            "ep_fmin": div.get("ep_batch_full_min"),
            "ep_fp5": div.get("ep_batch_full_p5"),
            "ep_nfull": div.get("ep_batch_n_full"),
            "ep_bret": div.get("ep_batch_retention"),
            "ep_sret": div.get("ep_screen_retention"),
            "ep_smin": div.get("ep_screen_ann_min"),
            "ep_nb": div.get("ep_n_batches"),
            "nvr_ne": float(np.mean(nvrs_ne)) if nvrs_ne else None,
            "frac_ne": float(np.mean(fracs_ne)) if fracs_ne else None,
            "pct_ess": float(np.mean(pcts_ess)) if pcts_ess else None,
        }
        log.info("  NVR=%.2f  nAUC=%.4f  frac=%.3f  shortfall=%.3f",
                 results[label]["nvr"] or 0, results[label]["nauc"] or 0,
                 results[label]["frac"] or 0, results[label]["shortfall"] or 0)

    # The Base-LLMs "Qwen3.6-27B" row and the finetuning-progression
    # "Qwen3.6-27B (base)" row are the same base model; source them both from
    # lix's canonical reparse (base_model_test_predictions.jsonl) so readers
    # don't see two slightly different numbers for one model. Overrides the
    # older sweep-based values computed in the LLM_METHODS loop above.
    if results.get("Qwen3.6-27B (base)", {}).get("nvr") is not None:
        results["Qwen3.6-27B"] = dict(results["Qwen3.6-27B (base)"])

    # Rarefaction diagnostics: which methods could not supply the fixed
    # reference counts, and so lost units (or a whole EP-D cell) to the
    # >= M_* filter. Read this before trusting a "-" in an EP column.
    def _pct(x):
        return "-" if x is None else f"{100 * x:.0f}%"

    def _ok(x):
        return "-" if x is None else f"{x:.0f}"

    _ep_rows = [(lab, r) for lab, r in sorted(results.items())
                if r and r.get("ep_ann") is not None]
    _ep_bad = [(lab, r) for lab, r in _ep_rows
               if r.get("ep_b") is None or r.get("ep_s") is None
               or r.get("ep_d") is None]
    log.info("EP rarefaction (M_batch=%d, M_screen=%d, M_dataset=%d, "
             "retention>=%.0f%%): %d methods, %d with a dashed EP cell",
             M_BATCH, M_SCREEN, M_DATASET, 100 * RETENTION,
             len(_ep_rows), len(_ep_bad))
    # Every method that lost a cell, plus why: low retention at batch/screen
    # scope, or too few pooled annotated genes for M_DATASET.
    for lab, r in sorted(_ep_bad, key=lambda kv: kv[1]["ep_ann"]):
        log.info("  %-40s pooled=%7d full_p5=%6s screen_min=%6s "
                 "ret(B/S)=%s/%s -> EP-B=%s EP-S=%s EP-D=%s",
                 lab.split(" [")[0][:40], r["ep_ann"], r.get("ep_fp5"),
                 r.get("ep_smin"),
                 _pct(r.get("ep_bret")), _pct(r.get("ep_sret")),
                 _ok(r.get("ep_b")), _ok(r.get("ep_s")), _ok(r.get("ep_d")))

    # Method, NVR, nAUC, Frac, Shortfall, %Ess, Vendi, Path.Ov., EP-B, EP-S, EP-D
    NC = 11
    DASHES = " & ".join(["-"] * (NC - 1)) + r" \\"

    def _row(label):
        r = results.get(label)
        disp = _DISPLAY.get(label, label.split(" [")[0])
        if r is None or r.get("nvr") is None:
            return f"{disp} & {DASHES}"
        return (
            f"{disp} & "
            f"{latex_val(r['nvr'])} & "                 # EF: ratio (not 0-1)
            f"{pct_val(r['nauc'])} & "
            f"{pct_val(r['frac'])} & "
            f"{pct_val(r['shortfall'])} & "
            f"{pct_val(r.get('pct_ess'))} & "
            f"{pct_val(r.get('vendi'))} & "
            f"{latex_val(r.get('pathway'))} & "        # Path. Ov.: vs-random ratio
            # Effective pathways, rarefied to a fixed annotated-gene count per
            # scope; the three are NOT comparable to each other.
            f"{latex_val(r.get('ep_b'), 1)} & "
            f"{latex_val(r.get('ep_s'), 0)} & "
            f"{latex_val(r.get('ep_d'), 0)} \\\\"
        )

    def _empty_row(label):
        return f"{label} & {DASHES}"

    def _section(title, first=False):
        pfx = "" if first else r"\midrule" + "\n"
        return (
            pfx
            + rf"\multicolumn{{{NC}}}{{@{{}}l}}"
            + rf"{{\textit{{\textbf{{{title}}}}}}}"
            + r" \\ \addlinespace"
        )

    # Build LaTeX table — identical structure to baselines.tex
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full performance comparison of various baselines and ablations. "
        r"Metrics evaluate enrichment factor (EF, hit rate relative to random), "
        r"normalized Area Under the Curve (nAUC), fraction of hits found, mean "
        r"shortfall, Vendi Diversity, and batch pathway overlap versus random. "
        r"EP-B/EP-S/EP-D are the effective number of Reactome pathways covered, at "
        r"batch, screen and dataset scope: each gene is assigned to one of its "
        r"pathways at random, and the scope is subsampled to a fixed annotated-gene "
        rf"count ({M_BATCH}, {M_SCREEN} and {M_DATASET} respectively), so the three "
        r"are bounded by their reference count and not comparable to one another. "
        r"A dash marks a method that cannot supply that count.}"
    )
    lines.append(r"\label{tab:baselines_results}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    # This table has ~63 rows: too tall for one page at \footnotesize/\scriptsize
    # (\scriptsize still ran into the footer / page number), and it is narrower
    # than \textwidth, so any \resizebox scales it UP (bigger font, width
    # overflow). Fix by shrinking the font only -- \tiny keeps width well within
    # \textwidth and fits the row count on a full-page [p] float with footer
    # clearance.
    lines.append(r"\tiny")
    lines.append(r"\begin{tabular}{@{}lcccccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{EF} & \textbf{nAUC (\%)} & "
        r"\textbf{Frac.\ hits (\%)} & \textbf{Shortfall (\%)} & "
        r"\textbf{Ess.\ (\%)} & "
        r"\textbf{Vendi (\%)} & \textbf{Path.\ Ov.} & "
        r"\textbf{EP-B} & \textbf{EP-S} & \textbf{EP-D} \\ \midrule"
    )

    # --- Base LLMs ---
    # ICBR-EF and LLMNN are computed via LLM_METHODS but rendered under
    # "Agent, Tuned & Meta-Learning Baselines" below, so skip them here to
    # avoid duplicate rows.
    _BASE_LLM_SKIP = {r"ICBR-EF~\cite{}", r"LLMNN~\cite{}", r"Haiku-4.5 Agent",
                      r"BioBO~\cite{}"}
    lines.append(_section("Base LLMs", first=True))
    for label, _ in LLM_METHODS:
        if label in _BASE_LLM_SKIP:
            continue
        lines.append(_row(label))

    # --- AssayLLM (Ours) ---
    lines.append(_section("AssayLLM (Ours)"))
    for label, _, _ in JSONL_METHODS:
        if "Handoff" in label or "Joint" in label:
            continue
        lines.append(_row(label))

    # --- Adaptive Experimental Design Methods ---
    lines.append(_section(r"Adaptive Experimental Design Methods"))
    lines.append(_row("Prior hit baseline"))
    lines.append(_row("kNN baseline"))
    lines.append(_row("RF (greedy)"))
    lines.append(_row("RF + UCB"))
    lines.append(_row(r"BPMF~\cite{}"))
    lines.append(_row(r"Transformer + DAgger~\cite{}"))
    lines.append(_row(r"MAML (+BPMF)~\cite{}"))
    lines.append(_row(r"BioBO~\cite{}"))
    lines.append(_row(r"Haystacks~\cite{}"))

    # --- Agent Harnesses ---
    lines.append(_section(r"Agent Harnesses"))
    lines.append(_row(r"Haiku-4.5 Agent"))
    lines.append(_row(r"LLMNN~\cite{}"))
    lines.append(_row(r"ICBR-EF~\cite{}"))

    # --- AssayFormer (Ours) ---
    lines.append(_section("AssayFormer (Ours)"))
    lines.append(_row("Transformer"))
    lines.append(_row(r"\quad + BPMF Embeddings"))
    lines.append(_row(r"\quad + GRPO (= AssayLoop)"))

    # --- AssayLoop (handoff framework) ---
    lines.append(_section("AssayLoop (Ours)"))
    lines.append(_row("GLM-5.1 - AssayLoop Handoff"))
    lines.append(_row("Gemini-3.1-Pro - AssayLoop Handoff"))
    lines.append(_row("GPT-5.5 - AssayLoop Handoff"))
    lines.append(_row("GPT-5.6 Sol - AssayLoop Handoff"))
    lines.append(_row("AssayLLM - AssayLoop Handoff"))
    lines.append(_row("AssayLLM-AssayLoop Handoff GRPO"))

    # --- Ablations ---
    lines.append(_section("Ablations"))
    lines.append(_row("Transformer (Random)"))
    lines.append(_row("Transformer (BPMF)"))
    lines.append(_row(r"\quad - screen desc."))
    lines.append(_row("Transformer (SVD)"))
    lines.append(_row("Transformer (MF)"))
    lines.append(_row("Transformer (MF-Sphere)"))
    lines.append(_row("Transformer (GenePT-PCA)"))
    lines.append(_row("Transformer (K562-PCA)"))
    lines.append(r"\addlinespace")
    lines.append(_row(r"Random\hfill + GRPO"))
    lines.append(_row(r"BPMF\hfill + GRPO"))
    lines.append(_row(r"BPMF (no desc.)\hfill + GRPO"))
    lines.append(_row(r"BPMF\hfill + GRPO (NVR$_{\text{terminal}}$)"))
    lines.append(_row(r"SVD\hfill + GRPO"))
    lines.append(_row(r"MF\hfill + GRPO"))
    lines.append(_row(r"MF-Sphere\hfill + GRPO"))
    lines.append(_row(r"GenePT-PCA\hfill + GRPO"))
    lines.append(_row(r"K562-PCA\hfill + GRPO"))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", out_file)

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
