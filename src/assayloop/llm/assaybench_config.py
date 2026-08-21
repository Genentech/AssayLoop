"""Load an AssayBench ``collect-*.yaml`` config into an
:class:`~assayloop.llm.client.LLMClientConfig`.

The AssayBench benchmarking harness configures every model it collects
predictions from with a small Hydra-style YAML under
``configs/lm/`` (e.g. ``collect-GLM-5.1.yaml``).
Each file inherits the shared ``collect-predictions.yaml`` base via a
``defaults:`` list and overrides the ``lm:`` block with the model's name,
endpoint, and recommended sampling preset:

    defaults:
      - collect-predictions
    model_name: "GLM-5"
    lm:
      provider: "local"
      model: "zai-org/GLM-5-FP8"
      max_tokens: 128000
      temperature: 1.0
      top_p: 0.95
      api_base: "http://localhost:8060/v1"
      api_key: "token-abc123"

Reusing these files keeps assayloop's LLM acquisitions sampling-identical
to the AssayBench leaderboard runs instead of re-specifying the preset by
hand. This module resolves the ``defaults`` inheritance (no Hydra/OmegaConf
dependency — the configs only ever inherit the single base) and maps the
merged ``lm`` block onto an ``LLMClientConfig``.

Provider mapping:

- ``local``     -> ``vllm`` (OpenAI-compatible local endpoint)
- ``anthropic`` -> ``anthropic``

``azure`` / ``openai`` configs are not wired (assayloop's acquisitions
target the local vLLM and Anthropic paths); loading one raises a clear
error.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .client import LLMClientConfig

__all__ = ["AssayBenchLM", "load_lm_config"]


@dataclass
class AssayBenchLM:
    """Result of loading an assaybench collect config.

    Attributes:
        client: the mapped :class:`LLMClientConfig` for the ``lm`` block.
        model_name: the config's ``model_name`` (used for output tagging),
            or ``None`` if unset.
        path: the resolved config path that was loaded.
    """

    client: LLMClientConfig
    model_name: Optional[str]
    path: Path


def _expand_env(value: Any, path: Path) -> Optional[str]:
    """Resolve a whole-value ``${VAR}`` reference in an LM config field.

    Endpoints and keys for hosted providers belong in the environment, not in
    a committed YAML. An unset variable raises here rather than passing the
    literal ``${VAR}`` through to the client, which would surface as an
    opaque 401.
    """
    if value is None:
        return None
    s = str(value)
    m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", s.strip())
    if not m:
        return s
    var = m.group(1)
    resolved = os.getenv(var, "").strip()
    if not resolved:
        raise KeyError(
            f"{path.name} needs ${{{var}}}, but {var} is unset. Export it "
            "(or add it to .env) before running this config."
        )
    return resolved


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_merged(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a config dict, resolving its Hydra-style ``defaults:`` list.

    Each entry in ``defaults`` names a sibling YAML (without extension)
    that is loaded first; the current file's own keys then override the
    merged bases. ``defaults`` itself is stripped from the result.
    """
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular defaults inheritance at {path}")
    _seen.add(path)

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    defaults = data.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for entry in defaults:
        # Entries are bare names like "collect-predictions"; Hydra also
        # allows dict entries (group: option) which these configs don't use.
        if not isinstance(entry, str):
            continue
        base_path = (path.parent / entry).with_suffix(".yaml")
        if not base_path.exists():
            raise FileNotFoundError(
                f"{path} inherits '{entry}' but {base_path} was not found"
            )
        merged = _deep_merge(merged, _load_merged(base_path, _seen))

    return _deep_merge(merged, data)


def load_lm_config(
    path: str | Path,
    *,
    disable_thinking: bool | None = None,
) -> AssayBenchLM:
    """Read an assaybench collect-*.yaml and map its ``lm`` block to an
    :class:`LLMClientConfig`.

    Args:
        path: path to a ``collect-*.yaml`` (or its base).
        disable_thinking: explicit override. When ``None`` (default) it is
            derived from ``lm.chat_template_kwargs.enable_thinking`` (thinking
            ON unless that flag is ``false``).

    Returns:
        :class:`AssayBenchLM` with the mapped client, the config's
        ``model_name``, and the resolved path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"lm config not found: {path}")

    merged = _load_merged(path)
    lm = merged.get("lm") or {}
    if not isinstance(lm, dict):
        raise ValueError(f"{path}: 'lm' block is not a mapping")

    provider_raw = str(lm.get("provider", "local")).lower()
    if provider_raw in ("local", "portkey"):
        provider = "vllm"
    elif provider_raw == "anthropic":
        provider = "anthropic"
    else:
        raise ValueError(
            f"{path}: provider {provider_raw!r} is not supported by "
            "assayloop's LLM acquisitions (only 'local'/'portkey' -> vllm "
            "and 'anthropic' are wired)."
        )

    # Thinking: assaybench expresses this as lm.chat_template_kwargs.
    if disable_thinking is None:
        ctk = lm.get("chat_template_kwargs") or {}
        enable_thinking = ctk.get("enable_thinking", True) if isinstance(ctk, dict) else True
        resolved_disable_thinking = not bool(enable_thinking)
    else:
        resolved_disable_thinking = bool(disable_thinking)

    def _num(key: str) -> Optional[float]:
        v = lm.get(key)
        return float(v) if v is not None else None

    def _int(key: str) -> Optional[int]:
        v = lm.get(key)
        return int(v) if v is not None else None

    # For anthropic, credentials come from the environment (.env:
    # ANTHROPIC_API_KEY, plus ANTHROPIC_BASE_URL for a gateway).
    # The shared ``collect-predictions`` base carries vLLM-oriented
    # ``api_base`` / ``api_key`` values; ignore them here so they don't leak
    # into the Anthropic client (which would point it at the local vLLM URL).
    if provider == "anthropic":
        base_url = None
        api_key = None
    else:
        base_url = _expand_env(lm.get("api_base"), path)
        api_key = _expand_env(lm.get("api_key"), path)

    extra_headers = lm.get("extra_headers") or {}
    if not isinstance(extra_headers, dict):
        extra_headers = {}

    client = LLMClientConfig(
        provider=provider,
        model=lm.get("model"),
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(lm["max_tokens"]) if lm.get("max_tokens") is not None else 32000,
        temperature=float(lm.get("temperature", 1.0)),
        top_p=_num("top_p"),
        top_k=_int("top_k"),
        min_p=_num("min_p"),
        presence_penalty=_num("presence_penalty"),
        repetition_penalty=_num("repetition_penalty"),
        disable_thinking=resolved_disable_thinking,
        use_max_completion_tokens=bool(lm.get("use_max_completion_tokens", False)),
        timeout=float(lm["timeout"]) if lm.get("timeout") is not None else 300.0,
        extra_headers={str(k): str(v) for k, v in extra_headers.items()},
        **{
            k: lm[k]
            for k in (
                "max_retries",
                "retry_initial_backoff",
                "retry_max_backoff",
                "retry_max_elapsed_s",
                "retry_on_empty",
                "max_empty_retries",
            )
            if lm.get(k) is not None
        },
    )

    model_name = merged.get("model_name")
    return AssayBenchLM(
        client=client,
        model_name=str(model_name) if model_name is not None else None,
        path=path.resolve(),
    )
