"""GLM-5.1 warm-start traces for the hybrid-handoff ranker.

At deployment GLM-5.1 acquires the first ``n`` active-learning rounds and the
transformer continues for the remaining ``10 - n`` rounds. GLM is never run live
during training/eval: per-screen, per-round acquisitions come from stored traces.

We only need the *gene symbols* GLM acquired each round - the true hit labels are
read from the screen's own data (``ScreenExample.hit`` at the matched position),
which guarantees the warm-start labels are exactly consistent with training.

Sources
-------
- Train: ``output/datasets/GLM-5-1_train.jsonl`` (from ``collect-dataset``). Each
  record's step-``(k+1)`` user message embeds an "Active-Learning History" listing
  rounds ``1..k`` with explicit Hits/Non-hits. The max-step (step 10) record for a
  trace therefore lists rounds 1..9. ~3 traces per screen ((seed,trace_idx)).
- Val/test: shared run dirs ``result.json`` -> ``steps[].acquired_batch`` (genes,
  in acquisition order) gives a single clean 10-round trace per screen.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger(__name__)

# "### Round 3 (93 genes, hit rate 53/93 = 57.0%)"
_ROUND_RE = re.compile(r"^###\s*Round\s+(\d+)\b", re.MULTILINE)
# "  Hits (89): RPL5, RPL11, ..."  /  "  Non-hits (8): RPL10, ..."
_GENE_LINE_RE = re.compile(r"^\s*(?:Hits|Non-hits)\s*\([^)]*\)\s*:\s*(.+)$", re.MULTILINE)

# A per-screen trace is an ordered list of rounds; each round is a list of gene
# symbols (hits and non-hits combined - acquisition membership is all we need).
Trace = list[list[str]]


def _parse_al_history(text: str) -> Trace:
    """Extract per-round acquired gene symbols from an AL-History user message."""
    idx = text.find("Active-Learning History")
    if idx == -1:
        return []
    hist = text[idx:]
    headers = list(_ROUND_RE.finditer(hist))
    rounds: Trace = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(hist)
        block = hist[start:end]
        genes: list[str] = []
        for m in _GENE_LINE_RE.finditer(block):
            genes.extend(g.strip() for g in m.group(1).split(",") if g.strip())
        rounds.append(genes)
    return rounds


def load_train_traces(jsonl_path: str | Path) -> dict[str, list[Trace]]:
    """Parse ``GLM-5-1_train.jsonl`` -> ``{screen_name: [trace, ...]}``.

    For each ``(screen, seed, trace_idx)`` we keep the largest-``step`` record (its
    history covers the most rounds) and parse rounds 1..(step-1) from it.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"warm-start train traces not found: {path}")

    # Pass 1: find the max step per trace key (cheap, no parsing).
    max_step: dict[tuple, int] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)["metadata"]
            except (json.JSONDecodeError, KeyError):
                continue
            key = (str(meta["screen"]), meta.get("seed"), meta.get("trace_idx"))
            step = int(meta.get("step", 0))
            if step > max_step.get(key, -1):
                max_step[key] = step

    # Pass 2: parse only the max-step record for each trace.
    traces: dict[str, list[Trace]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                meta = rec["metadata"]
            except (json.JSONDecodeError, KeyError):
                continue
            key = (str(meta["screen"]), meta.get("seed"), meta.get("trace_idx"))
            if int(meta.get("step", 0)) != max_step.get(key):
                continue
            user = next((m["content"] for m in rec.get("messages", [])
                         if m.get("role") == "user"), "")
            rounds = _parse_al_history(user)
            if rounds:
                traces.setdefault(str(meta["screen"]), []).append(rounds)
    log.info("warm-start: loaded %d train screens (%d traces) from %s",
             len(traces), sum(len(v) for v in traces.values()), path.name)
    return traces


def load_run_traces(runs_dir: str | Path, prefix: str) -> dict[str, list[Trace]]:
    """Parse shared run dirs ``<prefix>*/result.json`` -> ``{screen_name: [trace]}``.

    ``steps[].acquired_batch`` holds the per-round acquired gene symbols in order.
    Screen name is taken from ``result.json``'s ``task_id`` (``option2/<name>``).
    """
    base = Path(runs_dir)
    traces: dict[str, list[Trace]] = {}
    for d in sorted(base.glob(prefix + "*")):
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        name = str(r.get("task_id", "")).split("/")[-1]
        if not name:
            continue
        rounds = [list(s.get("acquired_batch") or []) for s in (r.get("steps") or [])]
        if rounds:
            traces.setdefault(name, []).append(rounds)
    log.info("warm-start: loaded %d run screens (%d traces) from %s/%s*",
             len(traces), sum(len(v) for v in traces.values()), base, prefix)
    return traces


class WarmStart:
    """Per-screen GLM acquisition traces, mapped to per-example gene positions.

    Call :meth:`index_examples` once with the screen examples to be trained/eval'd;
    afterwards :meth:`sample_positions` returns the observed-gene positions (indices
    into ``ScreenExample.gene_idx``/``.hit``) for the first ``n`` rounds.
    """

    def __init__(self, traces_by_name: dict[str, list[Trace]]):
        self.traces = traces_by_name
        # name -> list of traces; each trace is a list of per-round int64 position arrays
        self._pos: dict[str, list[list[np.ndarray]]] | None = None

    @classmethod
    def from_train_jsonl(cls, path: str | Path) -> "WarmStart":
        return cls(load_train_traces(path))

    @classmethod
    def from_run_dirs(cls, runs_dir: str | Path, prefix: str) -> "WarmStart":
        return cls(load_run_traces(runs_dir, prefix))

    def has(self, name: str) -> bool:
        return bool(self.traces.get(name))

    def index_examples(self, examples: Iterable[Any]) -> "WarmStart":
        """Map gene symbols -> positions within each example's gene array."""
        pos: dict[str, list[list[np.ndarray]]] = {}
        matched = total = 0
        for ex in examples:
            ts = self.traces.get(ex.name)
            if not ts:
                continue
            sym2pos: dict[str, int] = {}
            for i, g in enumerate(ex.genes):
                sym2pos.setdefault(g, i)
                up = g.upper()
                if up not in sym2pos:
                    sym2pos[up] = i
            tlist: list[list[np.ndarray]] = []
            for rounds in ts:
                rl: list[np.ndarray] = []
                for rnd in rounds:
                    idxs: list[int] = []
                    for g in rnd:
                        total += 1
                        p = sym2pos.get(g)
                        if p is None:
                            p = sym2pos.get(g.upper())
                        if p is not None:
                            idxs.append(p)
                            matched += 1
                    rl.append(np.asarray(idxs, dtype=np.int64))
                tlist.append(rl)
            pos[ex.name] = tlist
        self._pos = pos
        if total:
            log.info("warm-start: indexed %d/%d screens; gene match rate %.3f",
                     len(pos), len(self.traces), matched / total)
        return self

    def max_rounds(self, name: str) -> int:
        if not self._pos:
            return 0
        tlist = self._pos.get(name)
        return max((len(t) for t in tlist), default=0) if tlist else 0

    def sample_positions(
        self, name: str, n_rounds: int, rng: random.Random | None = None
    ) -> np.ndarray | None:
        """Positions observed after GLM runs ``n_rounds`` rounds for ``name``.

        Picks a random available trace (deterministic when only one exists, e.g.
        val/test). Returns ``None`` for ``n_rounds <= 0`` or a missing screen, so
        callers transparently fall back to ``n=0`` (pure transformer).
        """
        if self._pos is None:
            raise RuntimeError("WarmStart.index_examples() must be called first")
        if n_rounds <= 0:
            return None
        tlist = self._pos.get(name)
        if not tlist:
            return None
        if len(tlist) == 1:
            trace = tlist[0]
        else:
            r = rng if rng is not None else random
            trace = tlist[r.randrange(len(tlist))]
        k = min(n_rounds, len(trace))
        parts = [trace[i] for i in range(k) if trace[i].size > 0]
        if not parts:
            return None
        return np.unique(np.concatenate(parts))

    def round_positions(self, name: str, trace_idx: int = 0) -> list[np.ndarray]:
        """Per-round *newly observed* positions for one trace of ``name``.

        Round ``j`` returns the positions seen for the first time at round ``j``
        (i.e. ``unique(concat(rounds[:j+1])) \\ unique(concat(rounds[:j]))``), so
        that concatenating the first ``n`` rounds reproduces
        :meth:`sample_positions` ``(name, n)``'s cumulative-unique observed set.
        Returns ``[]`` for a missing screen / out-of-range trace.
        """
        if self._pos is None:
            raise RuntimeError("WarmStart.index_examples() must be called first")
        tlist = self._pos.get(name)
        if not tlist or trace_idx >= len(tlist):
            return []
        trace = tlist[trace_idx]
        out: list[np.ndarray] = []
        seen: set[int] = set()
        for rnd in trace:
            new = [int(p) for p in rnd.tolist() if int(p) not in seen]
            seen.update(new)
            out.append(np.asarray(new, dtype=np.int64))
        return out
