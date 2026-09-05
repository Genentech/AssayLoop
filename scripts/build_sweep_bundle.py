#!/usr/bin/env python3
"""Build the published-sweep bundle that reproduces the paper's main table.

MAINTAINER TOOL. Users do not run this -- they run ``scripts/fetch_sweeps.sh``,
which downloads the archive this script produces. This script only works
against the original result directories, which are not public.

WHY A BUNDLE EXISTS AT ALL

Most rows in the full-genome table are deterministic given a checkpoint: the
AssayFormer, BPMF and random rows re-run from ``sweep-fg-f2-fg-*`` in minutes
once you have the released weights, so they are not bundled. The LLM rows are
not reproducible at any price we can ask of a reader -- regenerating the 20
LLM sweeps means ~400 paid API runs against nine different vendors' models,
several of which are already retired. Those runs are what this bundle ships.

WHAT GOES IN

For each required sweep:
    sweeps/<sweep_id>/sweep.json      aggregate + per-screen final metrics
    runs/<run_id>/result.json         per-round acquired batches and metrics

Per-run ``config.json`` (13.6 MB across the bundle, near-identical between
runs) is deliberately excluded -- nothing reads it, and it is a secret-bearing
file. ``llm_calls.jsonl`` (the raw prompts and completions) is published as a
separate archive via ``--with-llm-calls``. It is required for exact replay of
open-vocabulary LLM rows: ``result.json`` abbreviates long trace strings, while
the current metrics reparse the complete answer against the shared f2 universe.

The required sweep IDs are not hardcoded here. They are resolved from
``full_genome_table``'s own method tables at build time, so the bundle cannot
drift from the table it exists to support.

SECRETS

Every one of these files was written by a job running behind an internal
gateway, and they show it: 80 ``result.json`` files embed a (now revoked)
API key 800 times, 160 embed an internal hostname, and a handful embed a
colleague's home directory inside a captured traceback. Scrubbing is not
optional and it is not best-effort. This script scrubs string values in the
parsed JSON (never the raw bytes -- that would corrupt escaping), then
re-scans the exact bytes it is about to write. **If anything on the deny-list
survives, the build aborts and writes nothing.** There is no flag to override
that; if the scan trips, fix the pattern list, do not bypass it.

USAGE

    python scripts/build_sweep_bundle.py --out dist/ \\
        --results  /path/to/output \\
        --shared   /path/to/dashboard_files

    # also build the raw LLM call log archive required for exact metric replay
    python scripts/build_sweep_bundle.py --out dist/ --with-llm-calls ...

Roots may also come from ``ASSAYLOOP_RESULTS`` / ``ASSAYLOOP_SHARED_PATH``.
The printed sha256 goes into ``scripts/fetch_sweeps.sh``'s EXPECTED map.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BUNDLE_VERSION = "v1"
SWEEP_ARCHIVE = f"assayloop-sweeps-{BUNDLE_VERSION}.tar.gz"
CALLS_ARCHIVE = f"assayloop-llm-calls-{BUNDLE_VERSION}.tar.gz"

# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------

REDACTED_HOST = "redacted-internal-host.invalid"
REDACTED_PATH = "<redacted-internal-path>"
REDACTED = "REDACTED"

# Credential-bearing *keys*. A text scan cannot see these -- in the parsed
# tree ``api_key`` is a dict key and its value is a separate string, so a
# regex over the serialized bytes never gets the two adjacent. Any string
# value under one of these keys is replaced wholesale, whatever it contains;
# that is what catches placeholders like vLLM's ``token-abc123``, which no
# secret-shaped pattern would ever match.
#
# Bare ``token`` is deliberately absent: in an LLM trace that can be a real
# decoded token, and over-redacting research content is its own failure.
CREDENTIAL_KEYS = re.compile(
    r"(?i)^(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|api[_-]?token|"
    r"refresh[_-]?token|secret|client[_-]?secret|password|passwd|"
    r"authorization|auth)$"
)

# Applied in order to every *decoded* string value. Each entry is
# (compiled pattern, replacement).
SCRUB_RULES: list[tuple[re.Pattern[str], str]] = [
    # Bearer-style API keys. Matches the one literal we know was written into
    # the sweep files, and is deliberately wider than it -- quoting the literal
    # here would put it back into the repo the scrubber exists to protect.
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "sk-REDACTED"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA-REDACTED"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_.\-]{16,}"), "Bearer REDACTED"),
    # api_key='token-abc123' inside a dataclass repr, and the JSON form.
    (re.compile(r"""(api[_-]?key\s*=\s*)(['"])[^'"]*\2"""), r"\1\2REDACTED\2"),
    (re.compile(r"""(["']api[_-]?key["']\s*:\s*)(["'])[^'"]*\2"""), r"\1\2REDACTED\2"),
    # Internal hostnames, URL-bearing or bare.
    (re.compile(r"[A-Za-z0-9_.\-]*\.(?:roche|gene)\.com"), REDACTED_HOST),
    # Absolute paths into the internal cluster (incl. a colleague's homedir,
    # which shows up inside captured tracebacks).
    (re.compile(r"/cv/home/[A-Za-z0-9_./\-]*"), REDACTED_PATH),
    (re.compile(r"/cv/data/[A-Za-z0-9_./\-]*"), REDACTED_PATH),
]

# Checked against the final serialized bytes. Any hit aborts the build.
# Written to tolerate the replacements above and nothing else.
DENY_PATTERNS: list[tuple[str, re.Pattern[bytes]]] = [
    ("api key", re.compile(rb"sk-(?!REDACTED)[A-Za-z0-9_\-]{12,}")),
    ("aws key", re.compile(rb"AKIA(?!-REDACTED)[0-9A-Z]{16}")),
    ("bearer token", re.compile(rb"(?i)\bbearer\s+(?!REDACTED)[A-Za-z0-9_.\-]{16,}")),
    ("internal host", re.compile(rb"\.(?:roche|gene)\.com")),
    ("cluster path", re.compile(rb"/cv/(?:home|data)/")),
    # Both the dataclass-repr form (api_key='...') and the JSON form
    # ("api_key": "..."). The lookahead permits only the exact placeholder.
    ("populated credential",
     re.compile(rb"""(?i)["']?(?:api[_-]?key|password|client[_-]?secret)["']?"""
                rb"""\s*[=:]\s*["'](?!REDACTED["'])[^"']+""")),
]


def scrub_obj(obj, counter: dict[str, int]):
    """Recursively rewrite string values (and dict keys) in a parsed JSON tree.

    Two independent mechanisms: pattern rules over every string value, and a
    key-driven blanket redaction for credential fields (see CREDENTIAL_KEYS).
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = scrub_obj(k, counter)
            if isinstance(k, str) and CREDENTIAL_KEYS.match(k) and isinstance(v, str):
                counter[f"<key> {k}"] = counter.get(f"<key> {k}", 0) + 1
                out[key] = REDACTED
            else:
                out[key] = scrub_obj(v, counter)
        return out
    if isinstance(obj, list):
        return [scrub_obj(v, counter) for v in obj]
    if isinstance(obj, str):
        out = obj
        for rx, repl in SCRUB_RULES:
            out, n = rx.subn(repl, out)
            if n:
                counter[rx.pattern] = counter.get(rx.pattern, 0) + n
        return out
    return obj


def deny_scan_tree(obj, where: str, path: str = "") -> list[str]:
    """Structural check: no credential key may carry a surviving value.

    This is the half ``deny_scan`` structurally cannot do -- see the comment
    on CREDENTIAL_KEYS.
    """
    problems = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}/{k}"
            if (isinstance(k, str) and CREDENTIAL_KEYS.match(k)
                    and isinstance(v, str) and v not in ("", REDACTED)):
                problems.append(f"{where}: credential key {sub} = {v[:40]!r}")
            problems.extend(deny_scan_tree(v, where, sub))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            problems.extend(deny_scan_tree(v, where, f"{path}[{i}]"))
    return problems


def deny_scan(data: bytes, where: str) -> list[str]:
    """Return a list of human-readable violations found in *data*."""
    problems = []
    for name, rx in DENY_PATTERNS:
        hits = rx.findall(data)
        if hits:
            sample = (hits[0] if isinstance(hits[0], bytes) else hits[0][0])
            problems.append(
                f"{where}: {name} x{len(hits)} "
                f"(e.g. {sample.decode('utf8', 'replace')[:60]!r})")
    return problems


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def required_sweep_ids() -> list[tuple[str, str]]:
    """Resolve (sweep_id, label) for every sweep the table cannot regenerate.

    Derived from ``full_genome_table``'s own method tables so the bundle stays
    in step with the table. Raises if any row fails to resolve -- an
    unresolvable row means the bundle would silently omit a published number.
    """
    from assayloop.scripts import full_genome_table as F
    from assayloop.scripts import results_index as R

    index = R.load_sweep_index(R._load_all_sweeps())
    found: dict[str, str] = {}
    unresolved: list[str] = []

    for label, key in F.LLM_METHODS:
        row = R.resolve_row(key, index)
        if row is None:
            unresolved.append(f"{label} <- {key}")
            continue
        found.setdefault(row[1], label)

    # RAW_SWEEP_METHODS name a sweep directory directly rather than going
    # through resolve_row.
    for label, sweep_id in F.RAW_SWEEP_METHODS:
        if sweep_id in index["by_sweep_id"]:
            found.setdefault(sweep_id, label)
        else:
            unresolved.append(f"{label} <- {sweep_id}")

    # Handoff rows replay warm-start traces out of the LLM sweeps' run dirs.
    # Every prefix should already be covered by LLM_METHODS above; assert it
    # rather than assume it.
    for label, prefix, _ranker, _n, _suffix in F.HANDOFF_METHODS:
        sid = prefix.rstrip("-")
        if sid in index["by_sweep_id"]:
            found.setdefault(sid, f"{label} (warm-start traces)")
        else:
            unresolved.append(f"{label} <- {sid}")

    if unresolved:
        raise SystemExit(
            "cannot build: these table rows did not resolve against the "
            "configured roots, so the bundle would be incomplete:\n  "
            + "\n  ".join(unresolved)
        )
    return sorted(found.items())


def find_in_roots(roots: list[Path], sub: str, name: str) -> Path | None:
    for root in roots:
        p = root / sub / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj, manifest: dict, violations: list[str]) -> int:
    """Serialize compactly, deny-scan (bytes + structure), write."""
    data = json.dumps(obj, separators=(",", ":"), sort_keys=False).encode()
    violations.extend(deny_scan(data, path.name))
    violations.extend(deny_scan_tree(obj, path.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    manifest[str(path)] = hashlib.sha256(data).hexdigest()
    return len(data)


def build(args) -> int:
    roots = [Path(p) for p in (args.results, args.shared) if p]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        raise SystemExit("no readable result roots; pass --results / --shared")

    sweeps = required_sweep_ids()
    print(f"required sweeps: {len(sweeps)}")

    stage = Path(args.out) / f"assayloop-sweeps-{BUNDLE_VERSION}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    calls_stage = None
    if args.with_llm_calls:
        calls_stage = Path(args.out) / f"assayloop-llm-calls-{BUNDLE_VERSION}"
        if calls_stage.exists():
            shutil.rmtree(calls_stage)
        calls_stage.mkdir(parents=True)

    scrub_counts: dict[str, int] = {}
    violations: list[str] = []
    hashes: dict[str, str] = {}
    entries = []
    n_runs = n_calls = 0
    bytes_sweeps = bytes_calls = 0

    for sweep_id, label in sweeps:
        sp = find_in_roots(roots, "sweeps", f"{sweep_id}/sweep.json")
        if sp is None:
            raise SystemExit(f"missing sweep.json for {sweep_id} ({label})")
        sweep = json.loads(sp.read_text())
        run_ids = [ps["run_id"] for ps in sweep.get("per_screen", []) if ps.get("run_id")]

        bytes_sweeps += write_json(
            stage / "sweeps" / sweep_id / "sweep.json",
            scrub_obj(sweep, scrub_counts), hashes, violations,
        )

        kept = []
        for run_id in run_ids:
            rd = find_in_roots(roots, "runs", run_id)
            if rd is None or not (rd / "result.json").is_file():
                raise SystemExit(
                    f"missing runs/{run_id}/result.json (sweep {sweep_id}, {label}). "
                    "Refusing to publish a partial sweep."
                )
            result = json.loads((rd / "result.json").read_text())
            bytes_sweeps += write_json(
                stage / "runs" / run_id / "result.json",
                scrub_obj(result, scrub_counts), hashes, violations,
            )
            kept.append(run_id)
            n_runs += 1

            if calls_stage is not None:
                src = rd / "llm_calls.jsonl"
                if not src.is_file():
                    continue
                lines = []
                for line in src.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = scrub_obj(json.loads(line), scrub_counts)
                    violations.extend(
                        deny_scan_tree(rec, f"{run_id}/llm_calls.jsonl"))
                    lines.append(json.dumps(rec, separators=(",", ":")))
                data = ("\n".join(lines) + "\n").encode()
                violations.extend(deny_scan(data, f"{run_id}/llm_calls.jsonl"))
                dst = calls_stage / "runs" / run_id / "llm_calls.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                hashes[str(dst)] = hashlib.sha256(data).hexdigest()
                bytes_calls += len(data)
                n_calls += 1

        entries.append({
            "sweep_id": sweep_id,
            "label": label,
            "display_name": sweep.get("display_name"),
            "model": sweep.get("config", {}).get("model"),
            "acq": sweep.get("config", {}).get("acq"),
            "tag": sweep.get("config", {}).get("tag"),
            "n_runs": len(kept),
            "run_ids": kept,
        })

    # ---- fail closed -----------------------------------------------------
    if violations:
        shutil.rmtree(stage)
        if calls_stage is not None:
            shutil.rmtree(calls_stage)
        print(f"\nSECRET SCAN FAILED ({len(violations)} violations) -- "
              "nothing was written.\n", file=sys.stderr)
        # Collapse the per-file repetition; the same leak in 400 runs is one
        # problem, and printing it 400 times buries the others.
        kinds: dict[str, tuple[int, str]] = {}
        for v in violations:
            where, _, what = v.partition(": ")
            n, first = kinds.get(what, (0, where))
            kinds[what] = (n + 1, first)
        for what, (n, first) in sorted(kinds.items(), key=lambda x: -x[1][0]):
            print(f"  x{n:<5d} {what}   (first: {first})", file=sys.stderr)
        print("\nAdd a rule to SCRUB_RULES or a key to CREDENTIAL_KEYS. "
              "Do not weaken the deny scans.", file=sys.stderr)
        return 1

    print("scrubbed (rule -> replacements):")
    for pat, n in sorted(scrub_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:7d}  {pat}")

    manifest = {
        "schema": 1,
        "bundle": f"assayloop-sweeps-{BUNDLE_VERSION}",
        "description": (
            "LLM and external-baseline sweeps behind the AssayLoop full-genome "
            "table. These runs cost paid API calls against models that are in "
            "part retired; they are published so the table can be recomputed "
            "without re-running them."
        ),
        "reproduces": "src/assayloop/scripts/full_genome_table.py",
        "unpack_to": "$ASSAYLOOP_PUBLISHED (default: <repo>/output/published)",
        "n_sweeps": len(entries),
        "n_runs": n_runs,
        "redaction": (
            "API keys, internal hostnames and internal filesystem paths were "
            "replaced with placeholders. No metric or acquired-gene field was "
            "modified."
        ),
        "sweeps": entries,
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "README.md").write_text(BUNDLE_README)

    out_dir = Path(args.out)
    tarpath = out_dir / SWEEP_ARCHIVE
    _tar(stage, tarpath)
    print(f"\n{tarpath}")
    print(f"  {len(entries)} sweeps, {n_runs} runs, "
          f"{bytes_sweeps/1e6:.1f} MB staged -> {tarpath.stat().st_size/1e6:.1f} MB gz")
    print(f"  sha256 {sha256_file(tarpath)}")

    if calls_stage is not None:
        # Not MANIFEST.json: both archives unpack into the same directory with
        # --strip-components=1, and their runs/ trees are meant to merge (one
        # contributes result.json, the other llm_calls.jsonl). A second
        # MANIFEST.json would silently overwrite the sweep provenance instead.
        (calls_stage / "MANIFEST-llm-calls.json").write_text(json.dumps({
            "schema": 1,
            "bundle": f"assayloop-llm-calls-{BUNDLE_VERSION}",
            "description": (
                "Raw per-call LLM prompts and completions for the sweeps in "
                f"{SWEEP_ARCHIVE}. Required for exact raw-response replay of "
                "the open-vocabulary LLM rows and also published for analysis "
                "of model behaviour."
            ),
            "n_runs": n_calls,
        }, indent=2) + "\n")
        cpath = out_dir / CALLS_ARCHIVE
        _tar(calls_stage, cpath)
        print(f"\n{cpath}")
        print(f"  {n_calls} runs, {bytes_calls/1e6:.1f} MB staged -> "
              f"{cpath.stat().st_size/1e6:.1f} MB gz")
        print(f"  sha256 {sha256_file(cpath)}")

    print("\nPut those sha256 values in scripts/fetch_sweeps.sh EXPECTED.")
    return 0


def _tar(stage: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    # Rebuilding identical inputs must give a byte-identical archive, or the
    # sha256 in fetch_sweeps.sh is a lie. Three sources of nondeterminism:
    # directory order, per-entry stat metadata, and -- easy to miss -- the
    # mtime gzip stamps into its own header, which tarfile's "w:gz" fills
    # from the clock. Hence the explicit GzipFile.
    with open(dest, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                           fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(stage.rglob("*")):
                    info = tf.gettarinfo(str(p),
                                         arcname=str(p.relative_to(stage.parent)))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if p.is_dir() else 0o644
                    if p.is_file():
                        with open(p, "rb") as fh:
                            tf.addfile(info, fh)
                    else:
                        tf.addfile(info)


BUNDLE_README = """\
# AssayLoop published sweeps

Cached results for the LLM and external-baseline rows of the AssayLoop
full-genome table. Unpack this directory so that its `sweeps/` and `runs/`
sit at `$ASSAYLOOP_PUBLISHED` (default `<repo>/output/published`); the normal
way to get that right is `scripts/fetch_sweeps.sh`.

    $ASSAYLOOP_PUBLISHED/
      MANIFEST.json
      sweeps/<sweep_id>/sweep.json
      runs/<run_id>/result.json

`MANIFEST.json` lists every sweep with the table row it backs, the model and
acquisition it used, and its run IDs.

Exact open-vocabulary LLM metrics also require the companion
`assayloop-llm-calls-v1.tar.gz`, because the evaluator reparses the lossless
completion against the shared gene universe. `scripts/fetch_sweeps.sh` fetches
and merges both archives by default.

## Why these are shipped and other sweeps are not

The AssayFormer, BPMF and random rows are deterministic given the released
checkpoints and re-run in minutes, so they are not shipped. The rows here
came from ~400 paid API runs across nine vendors' models, several since
retired, and cannot be regenerated by a reader at any reasonable cost.

## Redaction

These runs executed behind an internal gateway. API keys, internal hostnames
and internal filesystem paths have been replaced with placeholders such as
`sk-REDACTED` and `redacted-internal-host.invalid`. No metric, acquired-gene
list, prompt or model output was altered. The build refuses to produce an
archive if any known secret pattern survives.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="dist", help="output directory (default: dist)")
    ap.add_argument("--results", default=os.getenv("ASSAYLOOP_RESULTS"),
                    help="primary result root (dir containing sweeps/ and runs/)")
    ap.add_argument("--shared", default=os.getenv("ASSAYLOOP_SHARED_PATH"),
                    help="secondary result root")
    ap.add_argument("--with-llm-calls", action="store_true",
                    help=f"also build {CALLS_ARCHIVE}, required for exact LLM replay")
    ap.add_argument("--dry-run", action="store_true",
                    help="build into a throwaway directory and delete it; use "
                         "to check the secret scan and the sizes")
    args = ap.parse_args()
    if args.dry_run:
        args.out = args.out + "/.dryrun"
    rc = build(args)
    if args.dry_run and rc == 0:
        shutil.rmtree(args.out, ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
