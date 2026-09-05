"""Parser used by the published open-vocabulary LLM evaluations.

This module intentionally preserves the parser used when the cached benchmark
runs were scored. Replay must be stable even if the companion AssayBench parser
later becomes more permissive, otherwise old model responses acquire new genes
and the published metrics change without a new model call.
"""

from __future__ import annotations

import json
import re

_FINAL_ANSWER_RE = re.compile(
    r"<Final Answer>(.*?)</Final Answer>", re.DOTALL | re.IGNORECASE
)
_NUM_PREFIX_RE = re.compile(r"^\s*\d+[.)\s]+")
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|\n?```")


def parse_json_genes(text: str) -> list[str] | None:
    """Return a recognized gene list from a JSON object, if present."""
    if "{" not in text or "}" not in text:
        return None
    try:
        cleaned = _CODE_FENCE_RE.sub("", text)
        first_brace = cleaned.index("{")
        last_brace = cleaned.rindex("}")
        obj = json.loads(cleaned[first_brace:last_brace + 1])
        for key in ("genes", "predictions", "ranking", "top_genes", "selected_genes"):
            if key in obj and isinstance(obj[key], list):
                return [str(value).strip().upper() for value in obj[key]]
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def extract_gene_list(text: str) -> list[str]:
    """Extract and order gene symbols using the published-run convention."""
    json_genes = parse_json_genes(text)
    if json_genes is not None:
        return _dedupe_clean(json_genes)

    match = _FINAL_ANSWER_RE.search(text)
    body = match.group(1) if match else text
    candidates = [gene.strip() for gene in body.replace("\n", ",").replace(";", ",").split(",")]
    return _dedupe_clean(candidates)


def _dedupe_clean(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        gene = _NUM_PREFIX_RE.sub("", item)
        gene = gene.strip().strip(",.;:'\"`*-")
        if not gene or (" " in gene and len(gene.split()) > 1):
            continue
        gene = gene.upper()
        if gene in seen:
            continue
        seen.add(gene)
        out.append(gene)
    return out


def organism_suffix(task_context: dict) -> str:
    """Clarify the gene-symbol namespace required by the screen."""
    organism = (
        (task_context or {}).get("organism", "Homo sapiens")
        or "Homo sapiens"
    )
    convention = (task_context or {}).get("gene_symbol_convention") or (
        "HGNC" if "sapiens" in str(organism).lower() else "MGI"
    )
    return (
        f"All gene symbols must use the {convention} convention for {organism} "
        f"(e.g. 'TP53' for HGNC, 'Trp53' for MGI). "
        "Do not return symbols from another organism."
    )


__all__ = ["extract_gene_list", "parse_json_genes", "organism_suffix"]
