"""Replay saved open-vocabulary LLM answers against a gene universe.

Older runs stored both the raw LLM answer and the library-filtered acquired
batch. These helpers recover the policy's actual valid-gene choices so old
answers can be evaluated under the current full-universe convention without
calling the LLM again.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from assayloop.llm.parse_genes import extract_gene_list


def normalise_gene_batches(
    batches: Iterable[Iterable[Any]],
    universe_genes: Iterable[Any],
    *,
    batch_size: int = 100,
) -> list[list[str]]:
    """Canonicalise, validate, globally deduplicate, and cap gene batches."""
    canonical = {str(g).upper(): str(g) for g in universe_genes}
    seen: set[str] = set()
    out: list[list[str]] = []
    for batch in batches:
        picked: list[str] = []
        for raw_gene in batch or []:
            gene = canonical.get(str(raw_gene).upper())
            if gene is None or gene in seen:
                continue
            seen.add(gene)
            picked.append(gene)
            if len(picked) >= batch_size:
                break
        out.append(picked)
    return out


def replay_llm_steps(
    steps: Sequence[Mapping[str, Any]],
    universe_genes: Iterable[Any],
    *,
    batch_size: int = 100,
    response_texts: Sequence[str] | None = None,
) -> list[list[str]]:
    """Recover per-round valid picks, preferring each raw LLM response.

    ``response_texts`` should contain the unabridged responses from the run's
    ``llm_calls.jsonl``. When they are unavailable, the copy embedded in each
    result step is used; ``acquired_batch`` remains the legacy fallback.
    """
    if response_texts is not None and len(response_texts) != len(steps):
        raise ValueError(
            "response_texts must have exactly one entry per replayed step "
            f"({len(response_texts)} != {len(steps)})"
        )

    raw_batches: list[list[str]] = []
    for i, step in enumerate(steps):
        trace = step.get("acquisition_trace") or {}
        if response_texts is not None:
            raw_batches.append(extract_gene_list(str(response_texts[i] or "")))
        elif "response_text" in trace:
            text = str(trace.get("response_text") or "")
            if "...<truncated>" in text:
                raise ValueError(
                    "cannot replay a truncated acquisition_trace.response_text; "
                    "restore this run's llm_calls.jsonl (the published sweep "
                    "fetcher includes it by default)"
                )
            raw_batches.append(extract_gene_list(text))
        else:
            raw_batches.append(list(step.get("acquired_batch") or []))
    return normalise_gene_batches(raw_batches, universe_genes, batch_size=batch_size)


def load_logged_response_texts(
    run_dir: str | Path,
    *,
    expected_steps: int,
) -> list[str] | None:
    """Load one complete successful LLM response per acquisition step.

    ``result.json`` intentionally abbreviates trace strings at 1,000
    characters. The adjacent ``llm_calls.jsonl`` is the lossless record used
    for replay. A count mismatch returns ``None`` rather than guessing how
    retries or unrelated calls align to acquisition steps.
    """
    path = Path(run_dir) / "llm_calls.jsonl"
    if not path.is_file():
        return None

    responses: list[str] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("response_text")
                if rec.get("error") or not isinstance(text, str) or not text:
                    continue
                responses.append(text)
    except (OSError, json.JSONDecodeError):
        return None

    return responses if len(responses) == expected_steps else None


__all__ = [
    "load_logged_response_texts",
    "normalise_gene_batches",
    "replay_llm_steps",
]
