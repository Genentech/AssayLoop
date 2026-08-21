"""Post-training (SFT / distillation) dataset export.

Converts an inner-loop :class:`RunResult` into chat-messages JSONL records,
one per active-learning step, for finetuning a smaller student model on a
teacher LLM's traces.

Each record is a single supervised example::

    {
      "messages": [
        {"role": "system",    "content": "<acquisition system prompt>"},
        {"role": "user",      "content": "<screen + AL-history prompt>"},
        {"role": "assistant", "content": "<think>\n<reasoning>\n</think>\n\n<gene list>"}
      ],
      "metadata": {... screen / step / seed / metrics / hits tags ...}
    }

The teacher's *verbatim* answer is kept as the assistant target (that is what
we distill); the post-filter picks, hit outcomes, and per-batch metrics are
recorded in ``metadata`` so the dataset can be reward-filtered downstream
(the chosen "keep everything, tag with metrics" policy).

Only steps produced by an LLM acquisition (``llm_single`` / ``llm_single_blind``)
carry the full ``prompt_system`` / ``prompt_user`` / ``reasoning`` /
``response_text`` fields in their ``acquisition_trace`` (see
``acquisitions/llm_single_acq.py``); steps without them (warm-start, errored,
or non-LLM acquisitions) are skipped.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from assaybench.core.types import Observation, RunResult, StepRecord

# Serialised appends are guarded so concurrent screen runs (the runner's
# thread pool) cannot interleave partial JSONL lines into one file.
_WRITE_LOCK = threading.Lock()


def _obs_hit(o: Observation) -> bool:
    return bool(o.label.get("hit")) if isinstance(o.label, dict) else False


def scan_sft_records(
    out_path: str | Path,
) -> tuple[dict[tuple[str, int], list[tuple[int, str]]], list[str]]:
    """Scan an SFT JSONL file and group records by ``(screen, trace_idx)``.

    Returns ``(by_pair, misc_lines)`` where ``by_pair`` maps each
    ``(screen, trace_idx)`` to a list of ``(step, raw_json_line)`` tuples, and
    ``misc_lines`` holds any non-empty lines that could not be parsed/keyed
    (preserved verbatim so callers never silently drop data).
    """
    by_pair: dict[tuple[str, int], list[tuple[int, str]]] = {}
    misc_lines: list[str] = []
    out_path = Path(out_path)
    if not out_path.exists():
        return by_pair, misc_lines
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            md = json.loads(line).get("metadata", {})
            key = (str(md["screen"]), int(md["trace_idx"]))
            step = int(md["step"])
        except Exception:
            misc_lines.append(line)
            continue
        by_pair.setdefault(key, []).append((step, line))
    return by_pair, misc_lines


def missing_steps(steps: set[int], n_steps: int) -> list[int]:
    """Return the sorted steps in ``1..n_steps`` absent from ``steps``."""
    return sorted(set(range(1, n_steps + 1)) - set(steps))


def pair_is_complete(steps: set[int], n_steps: int) -> bool:
    """A ``(screen, trace)`` is complete iff every step ``1..n_steps`` is present."""
    return not missing_steps(steps, n_steps)


def _assistant_content(reasoning: str, answer: str) -> str:
    """Reconstruct the assistant turn with reasoning inline as <think>."""
    reasoning = (reasoning or "").strip()
    answer = (answer or "").strip()
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{answer}"
    return answer


def build_sft_records(
    result: RunResult,
    screen: Any,
    *,
    seed: int,
    trace_idx: int,
    teacher_label: str,
) -> list[dict[str, Any]]:
    """Build SFT records (one per LLM AL step) from a finished run.

    ``screen`` is the :class:`ScreenRecord` (used only for metadata tags).
    Returns a list of JSON-serialisable record dicts.
    """
    records: list[dict[str, Any]] = []
    screen_name = getattr(screen, "dataset_name", "") or result.task_id

    for s in result.history:
        if not isinstance(s, StepRecord):
            continue
        trace = s.acquisition_trace or {}
        # Skip warm-start (synthetic step 0), errored steps, and any
        # acquisition that didn't surface a full LLM trace.
        if s.step <= 0 or trace.get("warm_start"):
            continue
        if trace.get("error"):
            continue
        response_text = trace.get("response_text")
        prompt_user = trace.get("prompt_user")
        if not response_text or not prompt_user:
            continue

        messages: list[dict[str, str]] = []
        system = trace.get("prompt_system")
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(prompt_user)})
        messages.append({
            "role": "assistant",
            "content": _assistant_content(
                str(trace.get("reasoning") or ""), str(response_text)
            ),
        })

        hits = [_obs_hit(o) for o in s.new_observations]
        picked = trace.get("picked_genes") or [str(c) for c in s.acquired_batch]
        records.append({
            "messages": messages,
            "metadata": {
                "screen": screen_name,
                "split": getattr(screen, "split", ""),
                "organism": getattr(screen, "organism", ""),
                "num_genes": getattr(screen, "num_genes", None),
                "total_hits": getattr(screen, "total_hits", None),
                "step": int(s.step),
                "seed": int(seed),
                "trace_idx": int(trace_idx),
                "run_id": result.run_id,
                "teacher": teacher_label,
                "acquisition": trace.get("acquisition"),
                "reveal_labels": trace.get("reveal_labels"),
                "metrics": dict(s.metrics or {}),
                "hits": hits,
                "n_hits_in_batch": int(sum(1 for h in hits if h)),
                "picked_genes": list(picked),
                "n_picked": len(picked),
                "n_raw_picks": trace.get("n_raw_picks"),
                "n_out_of_pool": trace.get("n_out_of_pool"),
                "shortfall": trace.get("shortfall"),
                "has_reasoning": bool((trace.get("reasoning") or "").strip()),
            },
        })
    return records


def write_sft_records(
    result: RunResult,
    screen: Any,
    out_path: str | Path,
    *,
    seed: int,
    trace_idx: int,
    teacher_label: str,
) -> int:
    """Append this run's SFT records to ``out_path`` (JSONL).

    Thread-safe: appends are serialised so concurrent runs don't interleave.
    Returns the number of records written.
    """
    records = build_sft_records(
        result, screen, seed=seed, trace_idx=trace_idx, teacher_label=teacher_label
    )
    if not records:
        return 0
    out_path = Path(out_path)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    with _WRITE_LOCK:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    return len(records)


__all__ = [
    "build_sft_records",
    "write_sft_records",
    "scan_sft_records",
    "missing_steps",
    "pair_is_complete",
]
