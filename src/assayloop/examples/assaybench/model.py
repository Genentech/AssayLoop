"""Option 1 model: in-context LLM hit-gene predictor for a target screen.

The model is run after each AL step; it produces a ranking of all
genes (candidates_for_score) for the target screen using the acquired
training screens as few-shot context. The ranking is exposed as
``ModelPrediction.scores`` where each gene's score is ``-rank`` (so
greedy / top-K consumers work naturally).
"""

from __future__ import annotations

from typing import Any

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from ...llm.client import LLMClientConfig, complete
from assaybench.llm.parse_genes import extract_gene_list, organism_suffix
from ...tasks import ScreenRecord


def _format_training_block(observations: list[Observation], top_k: int) -> str:
    if not observations:
        return "  (no training screens selected yet)"
    parts: list[str] = []
    for o in observations:
        s = o.candidate
        if not isinstance(s, ScreenRecord):
            continue
        # Top-K by relevance.
        idx = sorted(range(len(s.genes)), key=lambda i: -float(s.relevance_scores[i]))[:top_k]
        top_genes = [s.genes[i] for i in idx]
        parts.append(
            f"[{s.dataset_name}] phenotype={s.phenotype!r} | "
            f"cell_line={s.cell_line!r} | organism={s.organism!r}\n"
            f"  top {len(top_genes)} hit-like genes: {', '.join(top_genes)}"
        )
    return "\n\n".join(parts)


class Option1LLMRanker(Model):
    """In-context LLM ranker for Option 1.

    Args:
        target: the target ScreenRecord whose hit genes we are
            predicting.
        llm: LLM client config (default vLLM/GLM-5).
        n_predictions: number of gene symbols to ask for.
        few_shot_top_k: per-training-screen, how many top genes to
            show the LLM as exemplars.
    """

    def __init__(
        self,
        target: ScreenRecord,
        *,
        llm: LLMClientConfig | None = None,
        n_predictions: int = 100,
        few_shot_top_k: int = 30,
    ):
        self.target = target
        self.llm = llm or LLMClientConfig()
        self.n_predictions = int(n_predictions)
        self.few_shot_top_k = int(few_shot_top_k)
        self._last_ranking: list[str] = []

    def name(self) -> str:
        cfg = self.llm.resolved()
        return f"option1_llm_ranker/{cfg.provider}:{cfg.model}"

    def reset(self) -> None:
        self._last_ranking = []

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        training = _format_training_block(observations, self.few_shot_top_k)
        target_block = (
            f"phenotype: {self.target.phenotype}\n"
            f"  cell_line: {self.target.cell_line}\n"
            f"  organism: {self.target.organism}\n"
            f"  library: {self.target.library_type} ({self.target.library_methodology})\n"
            f"  condition: {self.target.condition_clause}"
        )
        suffix = organism_suffix({
            "organism": self.target.organism,
            "gene_symbol_convention": self.target.gene_symbol_convention,
        })
        user = (
            f"=== Training screens (already selected) ===\n{training}\n\n"
            f"=== Target screen ===\n  {target_block}\n\n"
            f"Predict the top {self.n_predictions} hit genes for the TARGET screen, "
            f"ranked from strongest to weakest evidence. Use only the "
            f"{self.target.gene_symbol_convention or 'HGNC'} symbol "
            f"namespace. Respond with ONLY the comma-separated gene list "
            f"(no numbers, no prose). {suffix}"
        )
        system = (
            "You are a computational biologist ranking candidate hit genes "
            "for a CRISPR perturbation screen given context from related "
            "training screens. Output strictly the requested list."
        )

        try:
            text = complete(self.llm, prompt=user, system=system)
        except Exception as e:  # noqa: BLE001
            return ModelPrediction(
                scores={},
                uncertainty=None,
                metadata={"error": f"{type(e).__name__}: {e}"},
            )

        genes = extract_gene_list(text)[: self.n_predictions]
        self._last_ranking = genes

        # Score = -rank so the metric machinery (and greedy acquisition,
        # if any consumer wants to use it) treats earlier ranks as higher.
        scores: dict[str, float] = {}
        for rank, g in enumerate(genes):
            if g not in scores:
                scores[g] = float(-rank)
        return ModelPrediction(
            scores=scores,
            uncertainty=None,
            metadata={"ranked_genes": genes, "n_returned": len(genes)},
        )


__all__ = ["Option1LLMRanker"]
