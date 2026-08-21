"""Hypothesis-register LLM ranker (paper baseline).

Implements the ICL-EF strategy from "Can AI Scientist Agents Learn from
Lab-in-the-Loop Feedback?" (Wainrib et al., 2025).  No tools — the LLM
reasons over hit/miss feedback and maintains a JSON hypothesis register
that persists across active-learning iterations.

Candidate handling follows the LLMInContextRanker strategy: sample a
subset when the pool is too large, unsampled candidates get median-rank
scores.
"""

from __future__ import annotations

import json
import logging
import math
import random
from typing import Any

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ..llm.client import LLMClientConfig, complete
from assaybench.llm.parse_genes import extract_gene_list, organism_suffix

log = logging.getLogger("assayloop.models.hypothesis_ranker")

_SYSTEM_PROMPT = (
    "You are a computational biologist solving a sequential gene screen "
    "design problem. At each step you receive the screen description, the "
    "genes revealed so far (hits and non-hits), and your current hypothesis "
    "register — a JSON object summarising your evolving beliefs about which "
    "biological mechanisms, pathways, and gene families are most likely to "
    "yield hits in this screen.\n\n"
    "Your job:\n"
    "1. Analyse the new feedback (which genes were hits vs non-hits since "
    "the last round).\n"
    "2. Update your hypothesis register: refine, add, or discard hypotheses "
    "based on the evidence. Look for patterns — gene-family prefixes that "
    "are enriched among hits, pathway membership, functional similarity.\n"
    "3. Predict the genes most likely to be hits.\n\n"
    "Return your answer as a JSON object with exactly two keys:\n"
    '  "hypotheses_register": a list of hypothesis objects, each with '
    '"hypothesis" (string), "confidence" (high/medium/low), and '
    '"supporting_genes" (list of gene symbols that support it).\n'
    '  "genes": a list of gene symbols ranked from most likely hit to '
    "least likely.\n\n"
    "Return ONLY the JSON object, no other text."
)


def _format_observations(observations: list[Observation]) -> str:
    if not observations:
        return "  (none acquired yet)"
    hits = [str(o.candidate) for o in observations
            if isinstance(o.label, dict) and o.label.get("hit")]
    misses = [str(o.candidate) for o in observations
              if isinstance(o.label, dict) and not o.label.get("hit")]
    return (
        f"  Confirmed HITS ({len(hits)}):\n    "
        f"{', '.join(hits) or '(none)'}\n"
        f"  Confirmed NON-hits ({len(misses)}):\n    "
        f"{', '.join(misses) or '(none)'}"
    )


def _trim_question(q: str) -> str:
    start = q.find("## Experimental Context")
    if start != -1:
        q = q[start:]
    end = q.find("## Required Output Format")
    if end != -1:
        q = q[:end]
    return q.strip()


def _screen_block(ctx: dict[str, Any]) -> str:
    q = (ctx.get("question") or "").strip()
    if q:
        return _trim_question(q)
    parts = []
    for label, key in [
        ("Phenotype", "phenotype"),
        ("Cell line", "cell_line"),
        ("Cell type", "cell_type"),
        ("Organism", "organism"),
        ("Library methodology", "library_methodology"),
        ("Condition", "condition_clause"),
    ]:
        v = ctx.get(key)
        if v:
            parts.append(f"  {label}: {v}")
    return "\n".join(parts) or "  (no metadata)"


class HypothesisRankerModel(Model):
    """Tool-free LLM ranker with a persistent hypothesis register.

    At each AL step the LLM receives screen context, cumulative hit/miss
    feedback, and its previous hypothesis register. It returns an updated
    register and a ranked gene list.

    Args:
        llm: LLMClientConfig (default: anthropic / claude-sonnet-4-6).
        candidate_sample_size: max candidates shown in the prompt per call.
        batch_size: how many gene predictions to request.
        seed: RNG seed.
        score_temperature: softmax temperature for rank-to-score conversion.
        system_prompt: override for the system prompt.
    """

    def __init__(
        self,
        *,
        llm: LLMClientConfig | None = None,
        candidate_sample_size: int = 1000,
        batch_size: int = 100,
        seed: int = 0,
        score_temperature: float = 1.0,
        system_prompt: str | None = None,
    ):
        self._llm = llm or LLMClientConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=8192,
        )
        self._candidate_sample_size = int(candidate_sample_size)
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._score_temperature = float(score_temperature)
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._rng = random.Random(seed)
        self._hypothesis_register: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._hypothesis_register = []

    def name(self) -> str:
        cfg = self._llm.resolved()
        return f"hypothesis_ranker/{cfg.provider}:{cfg.model}"

    def _rank_to_scores(
        self, ranked: list[Any], all_candidates: list[Any],
    ) -> dict[Any, float]:
        if not ranked:
            return {}
        N = max(len(all_candidates), 1)
        T = max(1e-3, self._score_temperature * N)
        scores: dict[Any, float] = {}
        for i, g in enumerate(ranked):
            scores[g] = math.exp(-float(i) / T)
        m = max(scores.values()) or 1.0
        return {c: s / m for c, s in scores.items()}

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        if not candidates:
            return ModelPrediction(scores={}, metadata={"name": self.name()})

        ctx = task_context or {}
        suffix = organism_suffix(ctx)

        candidates_for_prompt = list(candidates)
        if len(candidates_for_prompt) > self._candidate_sample_size:
            candidates_for_prompt = self._rng.sample(
                candidates_for_prompt, self._candidate_sample_size,
            )

        cand_lines = []
        for i in range(0, len(candidates_for_prompt), 10):
            cand_lines.append(
                "  " + ", ".join(str(c) for c in candidates_for_prompt[i:i + 10])
            )

        register_json = json.dumps(self._hypothesis_register, indent=2)

        user_prompt = (
            f"=== Screen Context ===\n{_screen_block(ctx)}\n\n"
            f"=== Observations So Far ===\n{_format_observations(observations)}\n\n"
            f"=== Current Hypothesis Register ===\n{register_json}\n\n"
            f"=== Candidate genes ({len(candidates_for_prompt)} of {len(candidates)}) ===\n"
            f"{chr(10).join(cand_lines)}\n\n"
            f"=== Task ===\n"
            f"Update your hypothesis register based on the observations, then "
            f"predict the top {self._batch_size} genes most likely to be hits. "
            f"Return a JSON object with keys \"hypotheses_register\" and \"genes\". "
            f"{suffix}"
        )

        try:
            text = complete(self._llm, prompt=user_prompt, system=self._system_prompt)
        except Exception as e:
            log.warning("Hypothesis ranker failed: %s", e)
            return ModelPrediction(
                scores={},
                uncertainty=None,
                metadata={"name": self.name(), "error": str(e)},
            )

        new_register = self._hypothesis_register
        ranked_raw: list[str] = []
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                if "hypotheses_register" in obj and isinstance(obj["hypotheses_register"], list):
                    new_register = obj["hypotheses_register"]
                if "genes" in obj and isinstance(obj["genes"], list):
                    ranked_raw = [str(g).strip().upper() for g in obj["genes"]]
        except (json.JSONDecodeError, ValueError):
            log.warning("Failed to parse JSON from hypothesis ranker; falling back to gene extraction.")

        if not ranked_raw:
            ranked_raw = extract_gene_list(text)

        self._hypothesis_register = new_register
        log.info(
            "Hypothesis register updated: %d hypotheses.",
            len(self._hypothesis_register),
        )

        cand_upper = {str(c).upper(): c for c in candidates}
        ranked_in_pool: list[Any] = []
        seen: set[Any] = set()
        for g in ranked_raw:
            c = cand_upper.get(g)
            if c is not None and c not in seen:
                ranked_in_pool.append(c)
                seen.add(c)

        scores = self._rank_to_scores(ranked_in_pool, candidates)
        unc = {c: (0.1 if c in {str(x).upper() for x in candidates_for_prompt} else 0.5)
               for c in candidates}

        log.info(
            "Hypothesis ranker returned %d genes, %d matched candidate pool (of %d).",
            len(ranked_raw), len(ranked_in_pool), len(candidates),
        )

        return ModelPrediction(
            scores=scores,
            uncertainty=unc,
            metadata={
                "name": self.name(),
                "n_ranked_returned": len(ranked_raw),
                "n_matched_pool": len(ranked_in_pool),
                "n_hypotheses": len(self._hypothesis_register),
                "hypotheses_preview": json.dumps(self._hypothesis_register[:3]),
                "response_preview": text[:600],
            },
        )


__all__ = ["HypothesisRankerModel"]
