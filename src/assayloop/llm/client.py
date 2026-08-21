"""Unified LLM client factory.

Three providers are supported, keyed by ``LLMClientConfig.provider``:

- ``"vllm"``      (DEFAULT) — a local OpenAI-compatible vLLM endpoint,
                  GLM-5 by default. This is the ``ASSAYLLM``/open-weights
                  path; nothing leaves the machine.
- ``"anthropic"`` — the public Anthropic API, via
                  :func:`make_anthropic_client`. Authentication is a plain
                  ``ANTHROPIC_API_KEY``; a missing key raises
                  :class:`MissingAnthropicKey` rather than silently
                  falling back to another provider.
- ``"dspy"``      — DSPy ``dspy.LM`` for the baselines collector.

Every LLM-backed model and acquisition takes an ``LLMClientConfig``, so a
sweep can vary the provider without touching the components themselves.

Two constructors are exposed:

- :func:`complete(config, prompt)` → ``str``. For one-shot text generation.
- :func:`make_dspy_lm(config)`     → ``dspy.LM`` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import config as bl_config


@dataclass
class LLMClientConfig:
    """Configuration for an LLM call.

    Args:
        provider: one of ``"vllm"`` (default), ``"anthropic"``, ``"dspy"``.
        model: model name. Defaults to ``VLLM_MODEL_NAME`` for vllm and
            ``claude-opus-4-5`` for anthropic.
        base_url: API base URL. Defaults from env.
        api_key: API key. Defaults from env.
        max_tokens: hard cap for completion tokens (default 8192).
        temperature: sampling temperature (default 0.0).
        timeout: per-call timeout in seconds (default 120).
        extra_headers: passed through to the provider SDK.
        extra_kwargs: passed to the SDK as **kwargs.
    """

    provider: str = field(default_factory=lambda: bl_config.LLM_PROVIDER)
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Matches AssayBench's collect-predictions.yaml. Reasoning models
    # (GLM-5/5.1, DeepSeek-R1, ...) routinely use 5-30k tokens for
    # their <think> trace before emitting the final answer, so we need
    # plenty of headroom. AssayBench's GLM collection config bumps this to
    # 128000 — set higher per-acquisition if you want the full budget.
    max_tokens: int = 32000
    # Reasoning models produce degenerate output at temperature=0;
    # matches AssayBench (their collect-predictions.yaml and
    # configs/lm/collect-GLM-5.1.yaml both use 1.0).
    temperature: float = 1.0
    # Extra sampling controls. ``None`` => not sent on the request (server
    # default applies), which keeps existing GLM-5 runs unchanged. For
    # ``vllm`` these default-source from ``config.VLLM_*`` in ``resolved()``
    # so they can be pinned via .env (e.g. Qwen3 thinking-mode preset).
    # ``top_p`` / ``presence_penalty`` are standard OpenAI Chat params;
    # ``top_k`` / ``min_p`` / ``repetition_penalty`` are vLLM extensions
    # forwarded via ``extra_body``.
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    # Thinking is the whole point of GLM-5.1 — leave it ON by default.
    # If a vLLM backend supports the chat_template_kwargs hint and we
    # set this True, the model will skip its <think> phase entirely,
    # which is appropriate for tight tool-call inner loops but bad for
    # single-shot acquisitions that benefit from reasoning. The
    # ``complete()`` function always falls back to ``reasoning_content``
    # / ``reasoning`` when ``content`` is empty, so a truncated
    # thinking trace still produces output.
    disable_thinking: bool = False
    # Matches AssayBench's timeout (180s) with extra headroom.
    timeout: float = 300.0
    # Retry policy for transient failures (timeouts, connection resets, 5xx,
    # 429) AND empty completions. An AL step that ends in an empty batch
    # silently contaminates the trace and the metrics, so we retry instead of
    # degrading. ``max_retries`` is the number of *additional* attempts after
    # the first: 0 = no retries (one attempt), -1 = retry forever (until it
    # works). Backoff is exponential with jitter, capped at ``retry_max_backoff``.
    use_max_completion_tokens: bool = False
    max_retries: int = 8
    retry_initial_backoff: float = 2.0
    retry_max_backoff: float = 60.0
    # Wall-clock circuit breaker for a SINGLE completion call. ``None`` = no
    # cap (count-only, the historical behaviour). When set, the retry loop
    # gives up once cumulative elapsed time (since the first attempt, including
    # the next backoff) would exceed this many seconds — even if
    # ``max_retries`` is -1 (forever). This is what makes "retry forever" safe:
    # a brief blip (seconds/minutes) still recovers, but a wedged/too-slow
    # endpoint fails LOUDLY after a bounded time instead of spinning for hours.
    retry_max_elapsed_s: Optional[float] = None
    # Retry when the model returns a completely empty answer (no text and no
    # reasoning to fall back on). Default OFF: a `run` should let the model
    # emit an empty answer if it wants. ``collect-dataset`` turns this ON so
    # SFT batches always carry at least one gene. Empty-retries are capped
    # SEPARATELY from transient-error retries (``max_empty_retries``) so the
    # "retry timeouts forever" policy doesn't make an empty answer loop
    # forever.
    retry_on_empty: bool = False
    max_empty_retries: int = 8
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> "LLMClientConfig":
        """Return a copy with defaults filled in from env variables."""
        provider = (self.provider or "vllm").lower()
        if provider == "vllm":
            def _pick(val, env_default):
                return val if val is not None else env_default

            return LLMClientConfig(
                provider="vllm",
                model=self.model or bl_config.VLLM_MODEL_NAME,
                base_url=self.base_url or bl_config.VLLM_BASE_URL,
                api_key=self.api_key or bl_config.VLLM_API_KEY,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=_pick(self.top_p, bl_config.VLLM_TOP_P),
                top_k=_pick(self.top_k, bl_config.VLLM_TOP_K),
                min_p=_pick(self.min_p, bl_config.VLLM_MIN_P),
                presence_penalty=_pick(
                    self.presence_penalty, bl_config.VLLM_PRESENCE_PENALTY
                ),
                repetition_penalty=_pick(
                    self.repetition_penalty, bl_config.VLLM_REPETITION_PENALTY
                ),
                disable_thinking=self.disable_thinking,
                use_max_completion_tokens=self.use_max_completion_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_initial_backoff=self.retry_initial_backoff,
                retry_max_backoff=self.retry_max_backoff,
                retry_max_elapsed_s=self.retry_max_elapsed_s,
                retry_on_empty=self.retry_on_empty,
                max_empty_retries=self.max_empty_retries,
                extra_headers=dict(self.extra_headers),
                extra_kwargs=dict(self.extra_kwargs),
            )
        if provider == "anthropic":
            return LLMClientConfig(
                provider="anthropic",
                model=self.model or os.getenv("ANTHROPIC_MODEL_NAME", "claude-opus-4-5"),
                base_url=self.base_url or os.getenv("ANTHROPIC_BASE_URL") or None,
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY") or None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                disable_thinking=self.disable_thinking,
                use_max_completion_tokens=self.use_max_completion_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_initial_backoff=self.retry_initial_backoff,
                retry_max_backoff=self.retry_max_backoff,
                retry_max_elapsed_s=self.retry_max_elapsed_s,
                retry_on_empty=self.retry_on_empty,
                max_empty_retries=self.max_empty_retries,
                extra_headers=dict(self.extra_headers),
                extra_kwargs=dict(self.extra_kwargs),
            )
        if provider == "dspy":
            return LLMClientConfig(
                provider="dspy",
                model=self.model or os.getenv("DSPY_MODEL_NAME") or bl_config.VLLM_MODEL_NAME,
                base_url=self.base_url or bl_config.VLLM_BASE_URL,
                api_key=self.api_key or bl_config.VLLM_API_KEY,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                disable_thinking=self.disable_thinking,
                use_max_completion_tokens=self.use_max_completion_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_initial_backoff=self.retry_initial_backoff,
                retry_max_backoff=self.retry_max_backoff,
                retry_max_elapsed_s=self.retry_max_elapsed_s,
                retry_on_empty=self.retry_on_empty,
                max_empty_retries=self.max_empty_retries,
                extra_headers=dict(self.extra_headers),
                extra_kwargs=dict(self.extra_kwargs),
            )
        raise ValueError(f"Unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Anthropic SDK client (public API).
# ---------------------------------------------------------------------------


class MissingAnthropicKey(RuntimeError):
    """Raised when an Anthropic-backed component has no API key to use."""


def make_anthropic_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_headers: Optional[dict[str, str]] = None,
):
    """Return an ``anthropic.Anthropic`` client for the public API.

    The key is passed as ``api_key`` so the SDK sends it on ``x-api-key``,
    which is what the public API expects. There is no fallback: with no key
    available this raises :class:`MissingAnthropicKey` instead of building a
    client that fails later with an opaque 401.

    Args:
        api_key: explicit key; defaults to ``ANTHROPIC_API_KEY``.
        base_url: override the endpoint; defaults to ``ANTHROPIC_BASE_URL``
            when set. Only needed for a gateway speaking the Anthropic API.
        extra_headers: sent on every request as ``default_headers``.
    """
    import anthropic

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional at runtime
        pass

    key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if not key:
        raise MissingAnthropicKey(
            "No Anthropic API key. Set ANTHROPIC_API_KEY (create one at "
            "https://console.anthropic.com/settings/keys) or pass api_key=. "
            "Nothing is substituted for it."
        )

    kwargs: dict[str, Any] = {"api_key": key}
    url = base_url or os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    if url:
        kwargs["base_url"] = url
    if extra_headers:
        kwargs["default_headers"] = dict(extra_headers)

    return anthropic.Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# One-shot completion — used by LLMRanker / LLMSingleCallAcq.
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    """Structured result of a single-prompt completion.

    Attributes:
        text: the assistant's final answer text.
        reasoning: the model's internal reasoning trace
            (``reasoning_content`` for vLLM thinking models), empty when
            none was produced/surfaced.
        finish_reason: provider finish/stop reason, if known.
        usage: token-usage dict, if reported by the provider.
    """

    text: str
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None


# Substrings of exception type names we treat as transient (worth retrying).
# Matched against the whole MRO so subclasses are caught without importing the
# provider SDKs here.
_TRANSIENT_ERROR_NAMES = (
    "APITimeoutError",
    "APIConnectionError",
    "APIStatusError",
    "InternalServerError",
    "ServiceUnavailable",
    "RateLimitError",
    "Timeout",
    "ConnectionError",
    "ConnectError",
    "ReadTimeout",
    "RemoteProtocolError",
)
# Non-transient even if the name matches above (won't succeed on retry).
_PERMANENT_ERROR_NAMES = (
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
)

# Azure/OpenAI content-safety blocks surface as a 400 BadRequest ("we've limited
# access to this content for safety reasons" / Azure content-filter). Despite the
# 400 they are INTERMITTENT — the identical prompt frequently succeeds on a later
# attempt — so unlike a genuine malformed-request 400 they ARE worth retrying.
_CONTENT_FILTER_MARKERS = (
    "limited access to this content",
    "safety reasons",
    "content_filter",
    "content management policy",
    "responsibleaipolicy",
)


def _is_content_filter_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CONTENT_FILTER_MARKERS)


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristically classify an exception as a transient/retryable failure."""
    names = {klass.__name__ for klass in type(exc).__mro__}
    # Intermittent content-safety 400s are retryable despite being BadRequest.
    if _is_content_filter_error(exc):
        return True
    if names & set(_PERMANENT_ERROR_NAMES):
        return False
    if names & set(_TRANSIENT_ERROR_NAMES):
        return True
    # httpx/openai sometimes surface 5xx/429 via a status_code attribute.
    code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(code, int) and (code >= 500 or code == 429)


def _retry_backoff_seconds(attempt: int, cfg: "LLMClientConfig") -> float:
    """Exponential backoff with jitter for retry ``attempt`` (0-indexed)."""
    import random

    base = min(
        cfg.retry_max_backoff,
        cfg.retry_initial_backoff * (2 ** attempt),
    )
    return base + random.uniform(0.0, base * 0.25)


def _retries_remaining(attempt: int, max_retries: int) -> bool:
    """``True`` if another attempt is allowed (``max_retries < 0`` = forever)."""
    return max_retries < 0 or attempt < max_retries


def _within_time_budget(
    loop_start: float, cfg: "LLMClientConfig", next_delay: float = 0.0
) -> bool:
    """``True`` if retrying (after sleeping ``next_delay``) stays within the
    per-call wall-clock budget. Always ``True`` when no budget is configured.

    This is the circuit breaker that makes ``max_retries=-1`` (retry forever)
    safe: a wedged or far-too-slow endpoint is abandoned after a bounded time
    rather than spinning indefinitely.
    """
    import time as _t

    cap = cfg.retry_max_elapsed_s
    if cap is None or cap <= 0:
        return True
    return (_t.time() - loop_start + next_delay) < cap


def complete(
    config: LLMClientConfig,
    prompt: str,
    *,
    system: Optional[str] = None,
) -> str:
    """Single-prompt completion. Returns the assistant's text.

    Thin wrapper over :func:`complete_ex` for callers that only need the
    answer string (preserves the historical signature/behaviour).
    """
    return complete_ex(config, prompt, system=system).text


def complete_ex(
    config: LLMClientConfig,
    prompt: str,
    *,
    system: Optional[str] = None,
) -> "Completion":
    """Single-prompt completion returning text **and** reasoning.

    Same behaviour as :func:`complete` but exposes the model's internal
    reasoning trace (and finish reason / usage) so callers — e.g. the
    trace-collection path — can capture the full ``<think>`` content,
    not just the answer.
    """
    cfg = config.resolved()

    if cfg.provider == "vllm":
        import logging
        import time as _time

        from openai import OpenAI

        from ..tracing import log_completion_call

        client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=cfg.base_url)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # vLLM (and GLM-5 specifically) accepts chat_template_kwargs to
        # disable the model's internal <think>...</think> trace. Without
        # this, GLM-5 burns thousands of tokens reasoning silently and
        # returns an empty `message.content`. See
        # bridgebuilder/bridgebuilder/tools/bridge.py for the original
        # diagnosis. We also merge any user-provided extra_kwargs.
        extra_body: dict[str, Any] = {}
        if cfg.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        user_extra_body = (cfg.extra_kwargs or {}).get("extra_body")
        if isinstance(user_extra_body, dict):
            for k, v in user_extra_body.items():
                if k == "chat_template_kwargs" and isinstance(v, dict):
                    extra_body.setdefault("chat_template_kwargs", {}).update(v)
                else:
                    extra_body[k] = v
        # vLLM-only sampling extensions (sent via extra_body). Only set when
        # configured so the server default applies otherwise.
        for _k, _v in (
            ("top_k", cfg.top_k),
            ("min_p", cfg.min_p),
            ("repetition_penalty", cfg.repetition_penalty),
        ):
            if _v is not None:
                extra_body[_k] = _v
        # Standard OpenAI sampling params (top-level). Only include when set.
        sampling: dict[str, Any] = {}
        if cfg.top_p is not None:
            sampling["top_p"] = cfg.top_p
        if cfg.presence_penalty is not None:
            sampling["presence_penalty"] = cfg.presence_penalty
        _log = logging.getLogger(__name__)
        attempt = 0          # transient-error attempts (capped by max_retries)
        empty_attempt = 0    # empty-completion attempts (capped by max_empty_retries)
        loop_start = _time.time()  # for the wall-clock circuit breaker
        while True:
            t0 = _time.time()
            err: str | None = None
            exc: Exception | None = None
            content = ""
            reasoning = ""
            finish_reason: str | None = None
            usage_dict: dict[str, Any] | None = None
            response_model: str | None = None
            try:
                create_kwargs: dict[str, Any] = dict(
                    model=cfg.model,
                    messages=messages,
                    timeout=cfg.timeout,
                    extra_headers=cfg.extra_headers or None,
                    extra_body=extra_body or None,
                    **sampling,
                )
                if cfg.use_max_completion_tokens:
                    create_kwargs["max_completion_tokens"] = cfg.max_tokens
                else:
                    create_kwargs["temperature"] = cfg.temperature
                    create_kwargs["max_tokens"] = cfg.max_tokens
                resp = client.chat.completions.create(**create_kwargs)
                choice = resp.choices[0]
                msg = choice.message
                content = (getattr(msg, "content", None) or "").strip()
                reasoning = (
                    getattr(msg, "reasoning", None)
                    or getattr(msg, "reasoning_content", None)
                    or ""
                ) or ""
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                response_model = getattr(resp, "model", None)
                if response_model and response_model != cfg.model:
                    _log.warning(
                        "Model rerouted: requested %s, got %s",
                        cfg.model, response_model,
                    )
                # Fallback: thinking-mode backends sometimes put the answer
                # in `reasoning` / `reasoning_content` while `content` is
                # empty (e.g. when the budget runs out mid-stream, or when
                # the model was not invoked with enable_thinking=False).
                if not content and reasoning:
                    _log.warning(
                        "LLM returned empty content; falling back to reasoning "
                        "field (%d chars). Consider disable_thinking=True or a "
                        "higher max_tokens.",
                        len(reasoning),
                    )
                    content = reasoning.strip()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                exc = e
            finally:
                log_completion_call(
                    provider="vllm",
                    model=cfg.model or "",
                    messages=messages,
                    response_text=content,
                    reasoning_text=reasoning or None,
                    finish_reason=finish_reason,
                    usage=usage_dict,
                    latency_s=_time.time() - t0,
                    error=err,
                    extra={
                        "response_model": response_model,
                        "base_url": cfg.base_url,
                        "max_tokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "top_k": cfg.top_k,
                        "min_p": cfg.min_p,
                        "presence_penalty": cfg.presence_penalty,
                        "repetition_penalty": cfg.repetition_penalty,
                        "disable_thinking": cfg.disable_thinking,
                        "attempt": attempt,
                    },
                )

            # Decide whether to retry. Transient API errors and (optionally)
            # completely-empty completions are retried so a single flaky call
            # never silently turns into a zero-gene batch.
            if exc is not None:
                if _is_transient_error(exc) and _retries_remaining(attempt, cfg.max_retries):
                    delay = _retry_backoff_seconds(attempt, cfg)
                    if not _within_time_budget(loop_start, cfg, delay):
                        _log.error(
                            "vLLM call giving up after %.0fs wall-clock "
                            "(retry_max_elapsed_s=%.0fs, %d attempts); last "
                            "error: %s. The endpoint is unreachable or too slow "
                            "(check the server / lower max_tokens / raise timeout).",
                            _time.time() - loop_start, cfg.retry_max_elapsed_s,
                            attempt + 1, err,
                        )
                        raise exc
                    _log.warning(
                        "vLLM call failed (%s); retry %d%s in %.1fs.",
                        err, attempt + 1,
                        "" if cfg.max_retries < 0 else f"/{cfg.max_retries}",
                        delay,
                    )
                    _time.sleep(delay)
                    attempt += 1
                    continue
                raise exc
            if (
                not content
                and cfg.retry_on_empty
                and _retries_remaining(empty_attempt, cfg.max_empty_retries)
            ):
                delay = _retry_backoff_seconds(empty_attempt, cfg)
                _log.warning(
                    "vLLM call returned an empty completion; retry %d%s in %.1fs.",
                    empty_attempt + 1,
                    "" if cfg.max_empty_retries < 0 else f"/{cfg.max_empty_retries}",
                    delay,
                )
                _time.sleep(delay)
                empty_attempt += 1
                continue
            return Completion(
                text=content,
                reasoning=reasoning or "",
                finish_reason=finish_reason,
                usage=usage_dict,
            )

    if cfg.provider == "anthropic":
        import time as _time

        from ..tracing import log_completion_call

        import logging

        client = make_anthropic_client(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            extra_headers=cfg.extra_headers,
        )
        messages = [{"role": "user", "content": prompt}]
        _log = logging.getLogger(__name__)
        attempt = 0          # transient-error attempts (capped by max_retries)
        empty_attempt = 0    # empty-completion attempts (capped by max_empty_retries)
        loop_start = _time.time()  # for the wall-clock circuit breaker
        while True:
            t0 = _time.time()
            err: str | None = None
            exc: Exception | None = None
            content = ""
            usage_dict: dict[str, Any] | None = None
            stop_reason: str | None = None
            try:
                # Stream and accumulate: the Anthropic SDK refuses a non-streaming
                # create when max_tokens is large enough that the response could
                # exceed 10 minutes ("Streaming is required ..."). Streaming via the
                # SDK helper sidesteps that and still yields a fully-assembled final
                # Message (content blocks, stop_reason, usage).
                with client.messages.stream(
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    system=system or "",
                    messages=messages,
                ) as stream:
                    parts: list[str] = [text for text in stream.text_stream]
                    resp = stream.get_final_message()
                content = "".join(parts).strip()
                if not content:
                    # Fallback: reassemble from the final message's text blocks
                    # (e.g. if no text deltas streamed).
                    block_parts = [
                        t for b in resp.content if (t := getattr(b, "text", None))
                    ]
                    content = "\n".join(block_parts).strip()
                stop_reason = getattr(resp, "stop_reason", None)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    usage_dict = {
                        "input_tokens": getattr(usage, "input_tokens", None),
                        "output_tokens": getattr(usage, "output_tokens", None),
                    }
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                exc = e
            finally:
                full_messages = (
                    [{"role": "system", "content": system}] if system else []
                ) + messages
                log_completion_call(
                    provider="anthropic",
                    model=cfg.model or "",
                    messages=full_messages,
                    response_text=content,
                    reasoning_text=None,
                    finish_reason=stop_reason,
                    usage=usage_dict,
                    latency_s=_time.time() - t0,
                    error=err,
                    extra={
                        "max_tokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                        "attempt": attempt,
                    },
                )

            if exc is not None:
                # A 401 is an AuthenticationError, which is in
                # _PERMANENT_ERROR_NAMES: a bad ANTHROPIC_API_KEY fails fast
                # here rather than burning the retry budget.
                if _is_transient_error(exc) and _retries_remaining(attempt, cfg.max_retries):
                    delay = _retry_backoff_seconds(attempt, cfg)
                    if not _within_time_budget(loop_start, cfg, delay):
                        _log.error(
                            "Anthropic call giving up after %.0fs wall-clock "
                            "(retry_max_elapsed_s=%.0fs, %d attempts); last "
                            "error: %s.",
                            _time.time() - loop_start, cfg.retry_max_elapsed_s,
                            attempt + 1, err,
                        )
                        raise exc
                    _log.warning(
                        "Anthropic call failed (%s); retry %d%s in %.1fs.",
                        err, attempt + 1,
                        "" if cfg.max_retries < 0 else f"/{cfg.max_retries}",
                        delay,
                    )
                    _time.sleep(delay)
                    attempt += 1
                    continue
                raise exc
            if (
                not content
                and cfg.retry_on_empty
                and _retries_remaining(empty_attempt, cfg.max_empty_retries)
            ):
                delay = _retry_backoff_seconds(empty_attempt, cfg)
                _log.warning(
                    "Anthropic call returned an empty completion; retry %d%s in %.1fs.",
                    empty_attempt + 1,
                    "" if cfg.max_empty_retries < 0 else f"/{cfg.max_empty_retries}",
                    delay,
                )
                _time.sleep(delay)
                empty_attempt += 1
                continue
            return Completion(
                text=content,
                reasoning="",
                finish_reason=stop_reason,
                usage=usage_dict,
            )

    if cfg.provider == "dspy":
        lm = make_dspy_lm(cfg)
        out = lm(prompt)
        text = out[0] if isinstance(out, list) else out
        return Completion(text=text, reasoning="")

    raise ValueError(f"complete_ex: unsupported provider {cfg.provider!r}")


# ---------------------------------------------------------------------------
# DSPy LM — used by baselines collector
# ---------------------------------------------------------------------------


def make_dspy_lm(config: LLMClientConfig):
    """Return a ``dspy.LM`` instance configured for the given provider."""
    cfg = config.resolved()
    try:
        import dspy
    except ImportError as e:
        raise RuntimeError("dspy not installed") from e

    if cfg.provider == "vllm":
        return dspy.LM(
            model=f"openai/{cfg.model}",
            api_base=cfg.base_url,
            api_key=cfg.api_key or "not-needed",
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    if cfg.provider == "anthropic":
        # DSPy uses LiteLLM under the hood. Everything else in this file
        # routes through the anthropic SDK directly; this branch exists
        # only for the DSPy-based baselines collector.
        return dspy.LM(
            model=f"anthropic/{cfg.model}",
            api_key=cfg.api_key,
            api_base=cfg.base_url,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    if cfg.provider == "dspy":
        return dspy.LM(
            model=cfg.model,
            api_base=cfg.base_url,
            api_key=cfg.api_key,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    raise ValueError(f"make_dspy_lm: unsupported provider {cfg.provider!r}")


__all__ = [
    "LLMClientConfig",
    "Completion",
    "complete",
    "complete_ex",
    "make_dspy_lm",
]
