"""LLM in-context ranker model.

The Model variant: an LLM ranks a SAMPLE of unacquired candidates
given the acquired observations, and we softmax the ranking into
scores. Acquisition functions (greedy, UCB, ...) can then consume
these scores.

For libraries too large to fit in one prompt, we sample a subset per
call. Genes that aren't sampled this step get the mean score (i.e.
uniform belief).
"""

from __future__ import annotations

import math
import random
from typing import Any

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ..llm.client import LLMClientConfig, complete
from assaybench.llm.parse_genes import extract_gene_list, organism_suffix


_SYSTEM_PROMPT = (
    "You are a computational biologist scoring candidate genes for an "
    "active-learning CRISPR-screen prediction loop. Given the screen "
    "description and a small set of already-validated (hit / non-hit) "
    "examples, RANK the candidate genes from MOST LIKELY to LEAST LIKELY "
    "to be a hit in this screen. Return only the ranked gene-symbol "
    "list (one per row or comma-separated, highest-confidence FIRST)."
)


def _format_examples(
    observations: list[Observation],
    max_each: int | None = None,
) -> str:
    """Flat hit/non-hit list (no round structure — the Model interface
    receives flat observations, not the per-round history).

    No truncation by default (``max_each=None``) so the LLM sees every
    labelled example. Pass an integer only if you hit a hard prompt-
    budget constraint.
    """
    if not observations:
        return "  (none acquired yet)"
    hits = [str(o.candidate) for o in observations
            if isinstance(o.label, dict) and o.label.get("hit")]
    misses = [str(o.candidate) for o in observations
              if isinstance(o.label, dict) and not o.label.get("hit")]
    if max_each is not None:
        h_show, m_show = hits[:max_each], misses[:max_each]
        h_more = (f" (+{len(hits) - len(h_show)} more)"
                  if len(hits) > len(h_show) else "")
        m_more = (f" (+{len(misses) - len(m_show)} more)"
                  if len(misses) > len(m_show) else "")
    else:
        h_show, m_show = hits, misses
        h_more = m_more = ""
    return (
        f"  Confirmed HITS ({len(hits)}):\n    "
        f"{', '.join(h_show) or '(none)'}{h_more}\n"
        f"  Confirmed NON-hits ({len(misses)}):\n    "
        f"{', '.join(m_show) or '(none)'}{m_more}"
    )


def _screen_block(ctx: dict[str, Any]) -> str:
    parts = []
    for label, key in [
        ("Phenotype", "phenotype"),
        ("Cell line", "cell_line"),
        ("Cell type", "cell_type"),
        ("Organism", "organism"),
        ("Library methodology", "library_methodology"),
        ("Condition", "condition_clause"),
        ("Total hits expected", "total_hits"),
        ("Library size", "num_genes"),
    ]:
        v = ctx.get(key)
        if v:
            parts.append(f"  {label}: {v}")
    return "\n".join(parts) or "  (no metadata)"


class LLMInContextRanker(Model):
    """One LLM call per AL step, returning per-gene scores.

    Args:
        llm: LLMClientConfig (default: vLLM/GLM-5 from .env).
        candidate_sample_size: max candidates passed to the model per
            call (default 1000).
        seed: RNG seed.
        score_temperature: smaller -> sharper softmax. Default 1.0.
    """

    def __init__(
        self,
        *,
        llm: LLMClientConfig | None = None,
        candidate_sample_size: int = 1000,
        seed: int = 0,
        score_temperature: float = 1.0,
    ):
        self.llm = llm or LLMClientConfig()
        self.candidate_sample_size = int(candidate_sample_size)
        self.seed = int(seed)
        self.score_temperature = float(score_temperature)
        self._rng = random.Random(seed)
        self._last_meta: dict[str, Any] = {}

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._last_meta = {}

    def name(self) -> str:
        cfg = self.llm.resolved()
        return f"llm_incontext_ranker/{cfg.provider}:{cfg.model}"

    def _rank_to_scores(self, ranked: list[Any], all_candidates: list[Any]) -> dict[Any, float]:
        """Convert a (partial) ranking into a per-candidate score in [0, 1].

        Higher rank (closer to front) -> higher score. We use a softmax
        of negative rank position so the head is sharply favoured.
        """
        N = max(len(all_candidates), 1)
        # Rank position: ranked[0] -> 0; missing genes get the median rank.
        rank_pos: dict[Any, float] = {}
        for i, g in enumerate(ranked):
            rank_pos[g] = float(i)
        median = N / 2.0
        for c in all_candidates:
            rank_pos.setdefault(c, median)
        # Convert to softmax over -rank/T.
        T = max(1e-3, self.score_temperature * N)
        scores: dict[Any, float] = {}
        for c in all_candidates:
            scores[c] = math.exp(-rank_pos[c] / T)
        # Normalise to [0, 1] (max == 1).
        m = max(scores.values()) or 1.0
        return {c: s / m for c, s in scores.items()}

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        if not candidates:
            return ModelPrediction(scores={}, uncertainty={}, metadata={"name": self.name()})

        ctx = task_context or {}
        # Sample candidates if the full pool is too large.
        candidates_for_prompt = list(candidates)
        if len(candidates_for_prompt) > self.candidate_sample_size:
            candidates_for_prompt = self._rng.sample(candidates_for_prompt, self.candidate_sample_size)

        suffix = organism_suffix(ctx)
        cand_lines = []
        for i in range(0, len(candidates_for_prompt), 10):
            cand_lines.append("  " + ", ".join(str(c) for c in candidates_for_prompt[i:i + 10]))
        user = (
            f"=== Screen ===\n{_screen_block(ctx)}\n\n"
            f"=== Already validated examples ===\n{_format_examples(observations)}\n\n"
            f"=== Candidate genes to rank ({len(candidates_for_prompt)} of {len(candidates)}) ===\n"
            f"{chr(10).join(cand_lines)}\n\n"
            f"=== Task ===\n"
            f"Rank ALL {len(candidates_for_prompt)} candidate genes above from MOST to "
            f"LEAST likely to be a hit in this screen. Return a comma-separated list of "
            f"ALL candidate symbols in your ranked order, best first. No prose, no numbers. "
            f"{suffix}"
        )

        try:
            text = complete(self.llm, prompt=user, system=_SYSTEM_PROMPT)
        except Exception as e:
            # Fall back to uniform.
            base = 0.0
            return ModelPrediction(
                scores={c: base for c in candidates},
                uncertainty={c: 1.0 for c in candidates},
                metadata={"name": self.name(), "error": f"{type(e).__name__}: {e}"},
            )

        ranked = extract_gene_list(text)
        # Map back to candidate set.
        cand_upper = {str(c).upper(): c for c in candidates_for_prompt}
        ranked_in_set: list[Any] = []
        seen: set[Any] = set()
        for g in ranked:
            c = cand_upper.get(g)
            if c is not None and c not in seen:
                ranked_in_set.append(c)
                seen.add(c)

        scores = self._rank_to_scores(ranked_in_set, candidates)
        # High uncertainty for un-prompted candidates.
        unc = {c: (0.1 if c in cand_upper.values() else 0.5) for c in candidates}
        self._last_meta = {
            "n_ranked_returned": len(ranked_in_set),
            "n_prompt_candidates": len(candidates_for_prompt),
            "raw_response_preview": text[:600],
        }
        return ModelPrediction(
            scores=scores,
            uncertainty=unc,
            metadata={"name": self.name(), **self._last_meta},
        )


__all__ = ["LLMInContextRanker"]
