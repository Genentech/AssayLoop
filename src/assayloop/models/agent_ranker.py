"""Agent-based gene ranker model ("Haiku-4.5 Agent" in the results tables).

An LLM agent with code execution runs inside a sandboxed Apptainer container
that has access to the training screen dataset at /data/screens.json. At each
AL step it receives the screen context and observed hits/non-hits, can analyze
the training data programmatically, and returns a ranked list of gene
predictions (open-vocabulary — no candidate list provided).

The container is not shipped; it is built locally from the public training
screens in two steps, and the model refuses to run until it exists::

    uv run python -m assayloop.scripts.export_screen_dataset
    bash scripts/build_agent_sandbox.sh

Override its location with ``ASSAYLOOP_AGENT_SANDBOX_SIF``.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import anthropic

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation
from assaybench.llm.parse_genes import extract_gene_list, organism_suffix
from ..tracing import log_completion_call

log = logging.getLogger("assayloop.models.agent_ranker")


def _default_sif_path() -> Path:
    """Where ``scripts/build_agent_sandbox.sh`` writes the sandbox image."""
    env = os.getenv("ASSAYLOOP_AGENT_SANDBOX_SIF", "").strip()
    if env:
        return Path(env).expanduser()
    from .. import config

    return config.OUTPUT_PATH / "containers" / "agent_sandbox.sif"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a computational biologist solving a sequential gene screen design "
    "problem. At each step you receive the screen description and the genes "
    "revealed so far (hits and non-hits). Your goal is to predict which genes "
    "are most likely to be hits.\n\n"
    "You have access to a training dataset of 1349 screens at /data/screens.json. "
    "Each entry has: dataset_name, question (experimental context), "
    "cleaned_phenotype, hits (list of hit genes), non_hits (list of non-hit genes).\n\n"
    "Use the run_python tool to analyze this dataset — find screens with similar "
    "phenotypes or overlapping hits, and use that information to inform your "
    "predictions. Then return your final ranked gene list.\n\n"
    "The sandbox has NO internet access. Do not attempt to download packages, "
    "fetch URLs, or access any external resources. Available libraries: "
    "Python 3.11 standard library (json, collections, re, math, statistics, "
    "itertools, etc.), numpy, scipy, pandas, and scikit-learn. "
    "Tool output is truncated at 50k chars — never print entire datasets; "
    "compute summaries and aggregates instead.\n\n"
    "IMPORTANT: Your final message MUST end with a JSON object containing your "
    "ranked gene list. Use this exact format:\n"
    '{"genes": ["GENE1", "GENE2", "GENE3", ...]}\n'
    "Rank from most likely hit to least likely. Use HGNC nomenclature. "
    "No prose after the JSON object."
)


def _build_client() -> anthropic.Anthropic:
    """A plain public-API Anthropic client.

    The Tool Runner is a stock SDK feature, so nothing here needs a bespoke
    endpoint: a normal ``ANTHROPIC_API_KEY`` reproduces the agent runs. Shared
    with the LLM acquisitions so both authenticate identically and both raise
    :class:`~assayloop.llm.client.MissingAnthropicKey` when the key is unset.
    """
    from ..llm.client import make_anthropic_client

    return make_anthropic_client()


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


def _screen_context_block(ctx: dict[str, Any]) -> str:
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
    return "\n".join(parts) or "(no metadata)"


class AgentRankerModel(Model):
    """Agent-based gene ranker using sandboxed code execution.

    Args:
        system_prompt: Override for the agent's system prompt. None uses the
            built-in default.
        sif_path: Path to the Apptainer .sif sandbox image. Defaults to
            ``ASSAYLOOP_AGENT_SANDBOX_SIF``, else ``output/containers/
            agent_sandbox.sif`` — where ``scripts/build_agent_sandbox.sh``
            puts it.
        model_name: Anthropic model ID.
        batch_size: How many genes to ask the agent to return.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        sif_path: str | Path | None = None,
        model_name: str | None = None,
        batch_size: int = 100,
        seed: int = 0,
    ):
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._sif_path = Path(sif_path).expanduser() if sif_path else _default_sif_path()
        self._model_name = model_name or os.getenv("ANTHROPIC_MODEL_NAME", "claude-haiku-4-5")
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._client: anthropic.Anthropic | None = None
        self._check_sandbox()

    def _check_sandbox(self) -> None:
        """Fail before step 1 rather than on the first tool call.

        A missing sandbox does not degrade the agent into a plain LLM ranker --
        that would be a different baseline reported under this one's name -- so
        this raises, and names what to build.
        """
        if shutil.which("apptainer") is None:
            raise RuntimeError(
                "agent_ranker executes the model's code inside an Apptainer "
                "container, and 'apptainer' is not on PATH. Install Apptainer "
                "(https://apptainer.org), or run a baseline that needs no "
                "sandbox (--model hypothesis_ranker)."
            )
        if not self._sif_path.is_file():
            raise FileNotFoundError(
                f"Agent sandbox image not found at {self._sif_path}. Build it "
                "from the public training screens:\n"
                "  uv run python -m assayloop.scripts.export_screen_dataset\n"
                "  bash scripts/build_agent_sandbox.sh\n"
                "Or point ASSAYLOOP_AGENT_SANDBOX_SIF at an existing image."
            )

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return f"agent_ranker/{self._model_name}"

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = _build_client()
        return self._client

    def _run_python_tool(self, code: str) -> str:
        result = subprocess.run(
            [
                "apptainer", "exec",
                "--contain",
                "--network", "none",
                str(self._sif_path),
                "python3", "-c", code,
            ],
            capture_output=True, text=True, timeout=300,
        )
        output = result.stdout
        stderr = "\n".join(
            l for l in result.stderr.splitlines()
            if not l.startswith("INFO:")
        ).strip()
        if stderr:
            output += f"\nSTDERR:\n{stderr}"
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        output = output.strip() or "(no output)"
        max_len = 50_000
        if len(output) > max_len:
            output = output[:max_len] + f"\n\n[OUTPUT TRUNCATED — {len(output)} chars total, showing first {max_len}]"
        return output

    def _rank_to_scores(self, ranked: list[Any], all_candidates: list[Any]) -> dict[Any, float]:
        if not ranked:
            return {}
        N = max(len(all_candidates), 1)
        T = max(1e-3, 1.0 * N)
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

        user_prompt = (
            f"=== Screen Context ===\n{_screen_context_block(ctx)}\n\n"
            f"=== Observations So Far ===\n{_format_observations(observations)}\n\n"
            f"=== Task ===\n"
            f"Predict the top {self._batch_size} genes most likely to be hits in this "
            f"screen. You can use run_python to analyze the training data at "
            f"/data/screens.json to find similar screens and inform your predictions.\n\n"
            f"Return exactly {self._batch_size} gene symbols as a JSON object: "
            f'{{"genes": ["GENE1", "GENE2", ...]}} '
            f"ranked from most likely to least likely hit. {suffix}"
        )

        tool_calls: list[dict[str, Any]] = []

        @anthropic.beta_tool
        def run_python(code: str) -> str:
            """Execute Python code in a sandboxed container with NO internet access. Available: Python 3.11 stdlib, numpy, scipy, pandas, scikit-learn. Dataset at /data/screens.json."""
            output = self._run_python_tool(code)
            tool_calls.append({
                "tool": "run_python",
                "code": code[:2000],
                "output_preview": output[:1000],
            })
            return output

        client = self._get_client()
        agent_text = ""
        agent_messages: list[dict[str, str]] = []
        try:
            runner = client.beta.messages.tool_runner(
                model=self._model_name,
                max_tokens=4096,
                system=self._system_prompt,
                tools=[run_python],
                messages=[{"role": "user", "content": user_prompt}],
            )
            msg_t0 = time.time()
            for message in runner:
                msg_latency = time.time() - msg_t0
                text_parts: list[str] = []
                tool_use_extra: list[dict[str, Any]] = []
                for block in message.content:
                    if block.type == "text":
                        agent_text = block.text
                        text_parts.append(block.text)
                        agent_messages.append({"type": "text", "content": block.text[:1000]})
                        log.info("[agent] %s", block.text[:200])
                    elif block.type == "tool_use":
                        tool_use_extra.append({"tool": block.name, "input_preview": str(block.input)[:2000]})
                        log.info("[agent] tool_call: %s", block.name)
                usage = message.usage
                log_completion_call(
                    provider="anthropic",
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": self._system_prompt[:2000]},
                        {"role": "user", "content": user_prompt[:4000]},
                    ],
                    response_text="\n".join(text_parts),
                    finish_reason=message.stop_reason,
                    usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens} if usage else None,
                    latency_s=msg_latency,
                    extra={"tool_calls": tool_use_extra} if tool_use_extra else None,
                )
                msg_t0 = time.time()
        except Exception as e:
            tb = traceback.format_exc()
            log.error(
                "Agent ranker failed at predict(): %s\n%s",
                e, tb,
            )
            return ModelPrediction(
                scores={},
                uncertainty=None,
                metadata={
                    "name": self.name(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": tb,
                    "n_observations": len(observations),
                    "n_candidates": len(candidates),
                    "tool_calls": tool_calls,
                    "agent_messages": agent_messages,
                    "agent_text_before_error": agent_text[:600],
                },
            )

        ranked_raw = extract_gene_list(agent_text)
        cand_upper = {str(c).upper(): c for c in candidates}
        ranked_in_pool: list[Any] = []
        seen: set[Any] = set()
        for g in ranked_raw:
            c = cand_upper.get(g)
            if c is not None and c not in seen:
                ranked_in_pool.append(c)
                seen.add(c)

        scores = self._rank_to_scores(ranked_in_pool, candidates)
        log.info(
            "Agent returned %d genes, %d matched candidate pool (of %d).",
            len(ranked_raw), len(ranked_in_pool), len(candidates),
        )

        return ModelPrediction(
            scores=scores,
            uncertainty={c: 0.5 for c in candidates},
            metadata={
                "name": self.name(),
                "n_agent_returned": len(ranked_raw),
                "n_matched_pool": len(ranked_in_pool),
                "agent_response_preview": agent_text[:600],
                "tool_calls": tool_calls,
                "agent_messages": agent_messages,
            },
        )


__all__ = ["AgentRankerModel"]
