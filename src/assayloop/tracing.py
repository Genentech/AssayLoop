"""Local, on-disk tracing of every LLM call.

Each call made inside an active scope is appended to
``output/runs/<run_id>/llm_calls.jsonl``: the full prompt, the response
text and any reasoning text, finish_reason, token usage, and latency.
Nothing is uploaded anywhere; this is a plain file you can read, diff,
and replay offline. It is the per-call audit record behind every
LLM-backed baseline in the paper, and what you look at when a run
produced fewer genes than it was asked for.

Public API
----------

- :func:`run_trace_scope`     — context manager; wrap a run so that all
                                LLM calls made underneath are associated
                                with this run/sweep/task.
- :func:`set_current_run` / :func:`reset_current_run` — the manual form
                                of the same thing.
- :func:`log_completion_call` — the logger itself, called by
                                :func:`assayloop.llm.client.complete`
                                and by the agent baseline. A no-op when
                                no scope is active.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager as _contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config as bl_config

log = logging.getLogger("assayloop.tracing")

# ---------------------------------------------------------------------------
# Context variable: which run/sweep is the active LLM-traceable scope.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TraceScope:
    run_id: str
    sweep_id: Optional[str]
    task_id: Optional[str]


_CURRENT: contextvars.ContextVar[Optional[_TraceScope]] = contextvars.ContextVar(
    "assayloop_trace_scope", default=None
)


def set_current_run(
    run_id: str | None,
    *,
    sweep_id: str | None = None,
    task_id: str | None = None,
) -> contextvars.Token:
    """Mark ``run_id`` as the active LLM-trace scope for this thread/task.

    Returns a context-var token; pass it to :func:`reset_current_run`
    in a ``finally:`` block to restore the previous scope.
    """
    return _CURRENT.set(_TraceScope(run_id=run_id or "no-run",
                                    sweep_id=sweep_id,
                                    task_id=task_id))


def reset_current_run(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def current_run() -> _TraceScope | None:
    return _CURRENT.get()


@_contextmanager
def run_trace_scope(
    run_id: str | None,
    *,
    sweep_id: str | None = None,
    task_id: str | None = None,
):
    """Activate the trace scope for the duration of the ``with`` block.

    Preferred over the manual ``set_current_run`` / ``reset_current_run``
    pair, which it wraps: the scope is always reset, even on exception.
    """
    token = set_current_run(run_id, sweep_id=sweep_id, task_id=task_id)
    try:
        yield
    finally:
        reset_current_run(token)


# ---------------------------------------------------------------------------
# JSONL writer (thread-safe append, one file per run).
# ---------------------------------------------------------------------------


_WRITE_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _trace_path(run_id: str) -> Path:
    base = bl_config.OUTPUT_PATH / "runs" / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "llm_calls.jsonl"


def _lock_for(run_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(run_id)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[run_id] = lock
        return lock


def _append_jsonl(run_id: str, payload: dict[str, Any]) -> None:
    path = _trace_path(run_id)
    line = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
    lock = _lock_for(run_id)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def log_completion_call(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    response_text: str,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    latency_s: float,
    extra: dict[str, Any] | None = None,
    error: str | None = None,
    reasoning_text: str | None = None,
) -> None:
    """Append one LLM call to the active run's ``llm_calls.jsonl``.

    No-ops if no scope is active (e.g. when called from a unit test or
    a one-off REPL session) so this is always safe to call.
    """
    scope = current_run()
    if scope is None:
        return
    payload: dict[str, Any] = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "call_id": str(uuid.uuid4()),
        "run_id": scope.run_id,
        "sweep_id": scope.sweep_id,
        "task_id": scope.task_id,
        "provider": provider,
        "model": model,
        "messages": messages,
        "response_text": response_text,
        "reasoning_text": reasoning_text,
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_s": latency_s,
        "error": error,
    }
    if extra:
        payload["extra"] = extra
    try:
        _append_jsonl(scope.run_id, payload)
    except Exception as e:  # noqa: BLE001
        log.warning("Local LLM-trace write failed: %s", e)


__all__ = [
    "set_current_run",
    "reset_current_run",
    "run_trace_scope",
    "current_run",
    "log_completion_call",
]
