#!/usr/bin/env bash
# Download the cached sweep results behind the paper's full-genome table.
# Run this once before scripts/full_genome_table.py or the figure scripts.
#
# Nothing downloads these for you at import time, and nothing substitutes a
# degraded stand-in: a table row whose sweep is missing raises and names this
# script, rather than printing "-" as though the method had scored nothing.
#
# WHAT IS AND IS NOT SHIPPED
#
# Not shipped (regenerate them yourself):
#   sweep-fg-f2-fg-*   The AssayFormer, BPMF and random rows. These are
#                      deterministic given the released checkpoints -- fetch
#                      those from HuggingFace (see README "Checkpoints") and
#                      full_genome_table.py recomputes the rows in minutes.
#
# Fetched by this script:
#   assayloop-sweeps-v1.tar.gz      5.5M, required.
#                      The 20 LLM and external-baseline sweeps (GLM, Gemini,
#                      GPT, Claude, Kimi, Qwen, BioBO, Haystacks, LLMNN,
#                      ICBR-EF, the Haiku agent) as 20 sweep.json plus 400
#                      per-run result.json. Reproducing these from scratch
#                      means ~400 paid API runs against nine vendors, several
#                      of whose models are already retired, so unlike the
#                      rows above they are not something a reader can rerun.
#
#   assayloop-llm-calls-v1.tar.gz   29M, required for exact LLM replay.
#                      The raw prompts and completions for those same runs.
#                      Current metrics reparse completions against the shared
#                      f2 universe, so this archive is fetched by default. Use
#                      --without-llm-calls only if you do not plan to rebuild
#                      metrics. 360 runs (BioBO and Haystacks issue no calls).
#
# REDACTION
#
# These runs executed behind an internal gateway. API keys, internal
# hostnames and internal filesystem paths were replaced with placeholders
# (sk-REDACTED, redacted-internal-host.invalid) before publication. No
# metric, acquired-gene list, prompt or model output was altered. See
# scripts/build_sweep_bundle.py.
#
# Default target: $ASSAYLOOP_PUBLISHED, falling back to <repo>/output/published
#
# Re-running is idempotent: an archive is re-downloaded and re-unpacked only
# if its target is missing. Pass --force to do it regardless.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ASSAYLOOP_PUBLISHED:-$REPO_ROOT/output/published}"
BASE="${ASSAYLOOP_BUNDLE_URL:-https://github.com/Genentech/AssayLoop/releases/download/data-v1}"

FORCE=0
WITH_CALLS=1
for arg in "$@"; do
  case "$arg" in
    --force)           FORCE=1 ;;
    --with-llm-calls)  WITH_CALLS=1 ;;
    --without-llm-calls) WITH_CALLS=0 ;;
    -h|--help)         sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Unlike the gene-set assets, these archives are immutable release artifacts,
# so a sha256 mismatch is an error rather than a warning: it means you did not
# get the bytes the paper was computed from.
declare -A URLS=(
  ["assayloop-sweeps-v1.tar.gz"]="$BASE/assayloop-sweeps-v1.tar.gz"
  ["assayloop-llm-calls-v1.tar.gz"]="$BASE/assayloop-llm-calls-v1.tar.gz"
)

declare -A EXPECTED=(
  ["assayloop-sweeps-v1.tar.gz"]="1073e6f543e278134d5b3bc1d41d1181e610157aad16053d7c8e44172181e50f"
  ["assayloop-llm-calls-v1.tar.gz"]="c36fc84ce2358c5a1dccecf9e4995328faf58d93e864c52c4ce5b4cd3169f3c7"
)

# A sentinel path per archive: present => already unpacked. The two archives
# merge into one runs/ tree (result.json from one, llm_calls.jsonl from the
# other), so each needs a marker the other does not also create.
declare -A SENTINEL=(
  ["assayloop-sweeps-v1.tar.gz"]="MANIFEST.json"
  ["assayloop-llm-calls-v1.tar.gz"]="MANIFEST-llm-calls.json"
)

mkdir -p "$TARGET"
echo "Fetching published sweeps into: $TARGET"

fetch_and_unpack() {
  local name="$1"
  local want="${EXPECTED[$name]}"
  local sentinel="$TARGET/${SENTINEL[$name]}"

  if [[ -e "$sentinel" && $FORCE -eq 0 ]]; then
    echo "  [skip] $name already unpacked"
    return 0
  fi

  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  echo "  [get ] $name"
  curl -fsSL -o "$tmp/$name" "${URLS[$name]}"

  local got
  got="$(sha256sum "$tmp/$name" | cut -d' ' -f1)"
  if [[ "$got" != "$want" ]]; then
    echo "  ERROR: sha256 mismatch for $name" >&2
    echo "         expected $want" >&2
    echo "         got      $got" >&2
    echo "         These are immutable release artifacts. Do not use this file." >&2
    return 1
  fi

  # --strip-components=1 drops the archive's top-level assayloop-*-v1/ dir so
  # sweeps/ and runs/ land directly under $TARGET, which is the layout
  # results_index.py scans.
  tar xzf "$tmp/$name" -C "$TARGET" --strip-components=1
  echo "         -> unpacked"
}

fetch_and_unpack "assayloop-sweeps-v1.tar.gz"
if [[ $WITH_CALLS -eq 1 ]]; then
  fetch_and_unpack "assayloop-llm-calls-v1.tar.gz"
else
  echo "  [skip] assayloop-llm-calls-v1.tar.gz (exact raw-response replay unavailable)"
fi

n_sweeps=$(find "$TARGET/sweeps" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
n_runs=$(find "$TARGET/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo
echo "Done. $n_sweeps sweeps, $n_runs runs under $TARGET"
echo "Provenance: $TARGET/MANIFEST.json"
echo "The AssayFormer/BPMF rows are not in here -- they recompute from the"
echo "released checkpoints. See README 'Reproducing the tables'."
