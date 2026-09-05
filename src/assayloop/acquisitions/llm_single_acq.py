"""LLM single-call acquisition (open-vocabulary, AssayBench-style).

This is the "no candidate list" acquisition: the LLM is given ONLY
the screen description (and, in later AL steps, a short summary of
what's already been validated) and asked to produce ``batch_size``
gene symbols **from its own knowledge of the genome**.

This matches the AssayBench paper's standard ranking prompt
(see ``assaybench/data/prompts/objective_prompts.yaml`` ::
``biogrid_ranking_prompt`` and ``internal_biogrid_ranking_prompt``)
— the model is asked for 100 HGNC symbols based on the experimental
context alone; no in-prompt candidate pool is supplied.

Why this matters
----------------

Showing the LLM a candidate list constrains it to those genes — useful
when the dataset has an obscure library, but it also hides the model's
own prior over the genome behind a "pick from this list" framing.
Open-vocabulary lets us measure the model's actual biology knowledge,
which is what the AssayBench leaderboard scores.

Post-LLM filtering
------------------

The LLM's output is parsed, deduped, upper-cased, and then filtered to
genes still in the unrevealed acquisition pool. By default that pool is
the shared f2 gene universe used in the paper; ``--screen-library``
restricts it to the current screen's measured library instead.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from assaybench.core.acquisition import AcquisitionFunction
from assaybench.core.types import ModelPrediction, StepRecord
from assaybench.llm.history_format import format_history_by_round
from ..llm.client import LLMClientConfig, complete_ex
from ..llm.parse_genes import extract_gene_list, organism_suffix

log = logging.getLogger("assayloop.acquisitions.llm_single")


_SYSTEM_PROMPT = (
    "You are a computational biologist. Given a CRISPR screen "
    "description, produce a ranked list of {batch_size} gene symbols "
    "({convention} nomenclature for {organism}) that you believe are "
    "MOST LIKELY to be hits in this screen, ordered strongest first. "
    "Use only your knowledge of biology — no chain-of-thought in the "
    "output. Return ONLY a comma-separated list of {batch_size} "
    "symbols, no prose, no numbering."
)


# Brace-free static rewrite of ``_SYSTEM_PROMPT`` used as the seed candidate
# for prompt optimization (e.g. GEPA). The dynamic batch_size / convention /
# organism specifics are NOT included here because they are already restated
# in the user message (the final instruction + ``organism_suffix``), so an
# optimizer can freely rewrite this text without breaking ``str.format``
# placeholders or losing any required output-format guidance.
DEFAULT_LLM_SYSTEM_PROMPT = (
    "You are a computational biologist. Given a CRISPR screen description, "
    "produce a ranked list of gene symbols that you believe are MOST LIKELY "
    "to be hits in this screen, ordered strongest first. Use only your "
    "knowledge of biology. Return ONLY the requested comma-separated list of "
    "gene symbols, no prose, no numbering."
)


# Match the literal "100" in the AssayBench prompt's
# "list of 100 genes" / "list of exactly 100 HGNC gene symbols"
# wording, so we can swap it for an arbitrary batch_size.
_RX_LIST_OF_100 = re.compile(
    r"\b(list of)\s+(?:exactly\s+)?100\s+([A-Za-z]+(?:\s+gene)?\s+symbols?|"
    r"genes?)\b",
    re.IGNORECASE,
)
_RX_FORMAT_GENE100 = re.compile(r"GENE100\b")




def _build_screen_prompt(
    ctx: dict[str, Any],
    batch_size: int,
) -> str:
    """Construct the screen-description prompt body.

    Prefers the dataset's pre-rendered ``question`` field (which is
    AssayBench's official template, populated by InternalAssayBench /
    AssayBench loaders). Falls back to a manually composed AssayBench-
    style block when ``question`` is empty (e.g. public-split rows
    that don't carry a rendered prompt).
    """
    q = (ctx.get("question") or "").strip()
    if q:
        # The AssayBench template hardcodes "list of 100 ... genes" and
        # "GENE1, GENE2, ..., GENE100". Substitute batch_size in both
        # places so the LLM is asked for the right count, then return.
        q = _RX_LIST_OF_100.sub(rf"\1 exactly {batch_size} \2", q)
        q = _RX_FORMAT_GENE100.sub(f"GENE{batch_size}", q)
        return q

    # No pre-rendered question — build an AssayBench-style description
    # from the structured screen fields we have.
    cell_line = ctx.get("cell_line") or "the indicated"
    cell_type = ctx.get("cell_type") or ""
    library_methodology = ctx.get("library_methodology") or "perturbation"
    library_type = ctx.get("library_type") or "genome-wide"
    condition_clause = ctx.get("condition_clause") or ""
    phenotype = ctx.get("phenotype") or "produces the screen phenotype"
    organism = ctx.get("organism") or "Homo sapiens"
    convention = ctx.get("gene_symbol_convention") or (
        "HGNC" if "sapiens" in str(organism).lower() else "MGI"
    )
    cell_type_clause = f" ({cell_type})" if cell_type else ""
    return (
        "## Goal\n\n"
        f"You are tasked with ranking genes from a genetic perturbation "
        f"screen. Based on the experimental context and hit criteria "
        f"provided below, provide a list of exactly {batch_size} genes "
        f"that are hits in this screen, ranked from strongest to "
        f"weakest according to the criteria defined below.\n\n"
        "## Experimental Context\n\n"
        f"This screen was performed in {cell_line} cells{cell_type_clause}. "
        f"Researchers used a {library_type} library "
        f"({library_methodology}) to systematically perturb gene "
        f"function{condition_clause}.\n\n"
        "## Screen Objective\n\n"
        f"The primary objective of this screen was to identify a set of "
        f"hit genes, each of which {phenotype}\n\n"
        "## Required Output Format\n\n"
        f"Provide your response as an ordered list of exactly "
        f"{batch_size} {convention} gene symbols (standard nomenclature "
        f"for {organism}).\n\n"
        "Format:\n"
        f"GENE1, GENE2, GENE3, ..., GENE{batch_size}\n"
    )


class LLMSingleAcquisition(AcquisitionFunction):
    """Open-vocabulary single-call LLM acquisition.

    The LLM is NOT shown a candidate list — it picks from its own
    knowledge of the genome (matching AssayBench's standard ranking
    prompt). The acquisition then filters the LLM's output to genes
    that are still in the unrevealed acquisition pool (the shared f2
    universe by default, or the screen library when explicitly requested).

    Args:
        llm: LLMClientConfig. Defaults to the env-resolved vLLM/GLM-5
            with thinking on, ``max_tokens=32000``, ``temperature=1.0``.
        include_history_summary: include the running hit/non-hit table
            (with a "do not re-suggest" instruction) in the prompt.
            Default ``True``.
        reveal_labels: when ``True`` (default) the history block tells
            the LLM which previously-sampled genes were hits and which
            were not. When ``False`` it shows only the sampled-gene
            list per round, with "Do NOT re-suggest" but no labels —
            this is the ``llm_single_blind`` ablation that isolates the
            model's zero-shot prior from any active-learning feedback.
            If ``include_history_summary=False`` this flag is moot
            (no history block is emitted at all).
        oversample_factor: ask the LLM for ``ceil(batch_size *
            oversample_factor)`` symbols so that after filtering for
            in-library + not-yet-revealed we still have enough to fill
            the batch. Default ``1.0`` — matches AssayBench's prompt
            verbatim (asks for exactly ``batch_size`` symbols). Bump
            this above 1.0 only if you observe a real shortfall on
            later AL steps (the LLM might re-suggest already-revealed
            genes even when told not to), in which case e.g. ``1.3``
            keeps the head of the ranking faithful while leaving some
            slack. Values >1 spend extra LLM time generating ranks
            past ``batch_size`` that are usually discarded, and dilute
            the model's focus on its top picks.
        seed: RNG seed (used by the per-acquisition Python rng, e.g.
            for any future tie-breaking).
        system_prompt: optional override for the system message. When set,
            it is used VERBATIM (no ``str.format``); when ``None`` (default)
            the built-in ``_SYSTEM_PROMPT`` template is rendered with the
            batch size / nomenclature / organism. Used by prompt optimizers
            (e.g. GEPA) to inject a candidate system prompt. The dynamic
            count / convention / organism are already restated in the user
            message, so a static override loses no required guidance.
    """

    def __init__(
        self,
        *,
        llm: LLMClientConfig | None = None,
        include_history_summary: bool = True,
        reveal_labels: bool = True,
        oversample_factor: float = 1.0,
        seed: int = 42,
        system_prompt: str | None = None,
    ):
        self.llm = llm or LLMClientConfig()
        self.include_history_summary = include_history_summary
        self.reveal_labels = bool(reveal_labels)
        self.oversample_factor = float(oversample_factor)
        self.seed = int(seed)
        self.system_prompt = system_prompt or None
        self._rng = random.Random(seed)
        self._last_trace: dict[str, Any] = {}

    def name(self) -> str:
        cfg = self.llm.resolved()
        suffix = "" if self.reveal_labels else "_blind"
        return f"llm_single{suffix}/{cfg.provider}:{cfg.model}"

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_trace = {}

    def last_trace(self) -> dict[str, Any]:
        return self._last_trace

    def _build_prompt(
        self,
        history: list[StepRecord],
        batch_size: int,
        task_context: dict[str, Any] | None,
    ) -> tuple[str, str, int]:
        ctx = task_context or {}

        n_requested = max(
            batch_size,
            int(round(batch_size * self.oversample_factor)),
        )

        organism = ctx.get("organism") or "Homo sapiens"
        convention = ctx.get("gene_symbol_convention") or (
            "HGNC" if "sapiens" in str(organism).lower() else "MGI"
        )
        if self.system_prompt:
            # Optimizer-supplied prompt: use verbatim (no .format). The
            # count / convention / organism are carried by the user message.
            system = self.system_prompt
        else:
            system = _SYSTEM_PROMPT.format(
                batch_size=n_requested,
                convention=convention,
                organism=organism,
            )

        screen_block = _build_screen_prompt(ctx, n_requested)
        history_block = (
            format_history_by_round(history, include_labels=self.reveal_labels)
            if self.include_history_summary else ""
        )
        suffix = organism_suffix(ctx)
        user = (
            f"{screen_block}{history_block}\n\n"
            f"=== Final instruction ===\n"
            f"Return ONLY a comma-separated list of {n_requested} gene "
            f"symbols (no prose, no numbers, no markdown). {suffix}"
        )
        return system, user, n_requested

    def suggest(
        self,
        history: list[StepRecord],
        candidates: list[Any],
        batch_size: int,
        model_prediction: ModelPrediction | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not candidates:
            return []

        system, user, n_requested = self._build_prompt(
            history, batch_size, task_context
        )

        try:
            completion = complete_ex(self.llm, prompt=user, system=system)
            text = completion.text
            reasoning = completion.reasoning or ""
        except Exception as e:
            log.warning(
                "LLM single-call failed (%s: %s); returning empty batch "
                "(no padding).", type(e).__name__, e,
            )
            self._last_trace = {
                "acquisition": "llm_single_blind" if not self.reveal_labels else "llm_single",
                "reveal_labels": self.reveal_labels,
                "error": f"{type(e).__name__}: {e}",
                "n_llm_picked": 0,
                "shortfall": batch_size,
                "raw_response_preview": "",
                "n_requested_from_llm": n_requested,
            }
            return []

        picks_raw = extract_gene_list(text)
        # Build an upper-case index of unrevealed candidates (the
        # acquisition is open-vocab, but we must still only acquire
        # genes in the configured pool that have not already been revealed).
        cand_upper = {str(c).upper(): c for c in candidates}
        n_matched = 0
        n_already_revealed = 0
        n_unknown = 0
        # The inner loop passes only unrevealed candidates. A repeated pick
        # and a symbol outside the configured universe therefore both miss
        # this index and are reported together as out-of-pool shortfall.
        picked: list[Any] = []
        seen: set[Any] = set()
        for g in picks_raw:
            c = cand_upper.get(g)
            if c is None:
                n_unknown += 1
                continue
            n_matched += 1
            if c in seen:
                continue
            picked.append(c)
            seen.add(c)
            if len(picked) >= batch_size:
                break

        ctx = task_context or {}
        screen = ctx.get("screen_name") or ctx.get("dataset_name") or "?"
        step = len(history) + 1

        shortfall = max(0, batch_size - len(picked))
        if shortfall > 0:
            log.warning(
                "[%s step %d] under-supplied: %d of %d requested "
                "(raw picks=%d, matched-in-pool=%d, out-of-pool=%d). "
                "NOT padding.",
                screen, step, len(picked), batch_size, len(picks_raw),
                n_matched, n_unknown,
            )

        self._last_trace = {
            "acquisition": "llm_single_blind" if not self.reveal_labels else "llm_single",
            "reveal_labels": self.reveal_labels,
            "llm": self.llm.resolved().__dict__,
            "n_candidates_in_prompt": 0,  # open-vocab: no list shown
            "n_requested_from_llm": n_requested,
            "raw_response_preview": text[:600],
            "n_raw_picks": len(picks_raw),
            "n_matched_in_pool": n_matched,
            "n_out_of_pool": n_unknown,
            "n_llm_picked": len(picked),
            "n_already_revealed": n_already_revealed,
            "shortfall": shortfall,
            # Full prompt / reasoning / answer for trace-collection
            # (SFT-dataset export). These are large strings; the runner's
            # result.json serialiser truncates them via _strip_trace, but
            # the in-memory StepRecord keeps them intact for the SFT writer.
            "prompt_system": system,
            "prompt_user": user,
            "reasoning": reasoning,
            "response_text": text,
            "picked_genes": [str(c) for c in picked],
        }
        return picked


__all__ = ["LLMSingleAcquisition", "DEFAULT_LLM_SYSTEM_PROMPT"]
