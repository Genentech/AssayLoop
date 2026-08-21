#!/usr/bin/env bash
# Download the public STRING / CORUM / SIGNOR releases and rebuild the
# interaction ground-truth tables that the network-recovery analysis (paper
# Figure 9) and the CORUM-labelled UMAPs read.
#
# The paper's tables came from an internal data platform that cannot be
# redistributed. scripts/convert_ground_truth.py rebuilds the same layout from
# the public releases, pinned to the same versions:
#
#   STRING v11.5 (2021-08-12)   CC BY 4.0, https://string-db.org/cgi/access
#   CORUM  4.1   (2022-11-28)   CC BY 4.0 (CORUM releases before 5.0)
#   SIGNOR 3.0                  CC BY 4.0 (SIGNOR 3.0 and later)
#
# All three are CC BY 4.0 at these pinned versions, so tables derived from them
# may be redistributed with attribution. This repository ships none of their
# data regardless -- you download it yourself.
#
# STRING downloads unattended. CORUM and SIGNOR serve their bulk files through
# pages that may require a form or may be unreachable from a locked-down
# network; if a fetch fails, this script says exactly which file to place
# where, and the converter refuses to build a partial ground truth.
#
# Usage:
#   scripts/fetch_ground_truth.sh [target-dir] [--force]
#
# Default target: $ASSAYLOOP_GROUND_TRUTH, else <repo>/data/ground_truth.
# Raw downloads land in <target>/_raw/ and are kept, so re-running is cheap.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET=""
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    # Print the header block, however long it has grown, minus the shebang.
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) TARGET="$arg" ;;
  esac
done
TARGET="${TARGET:-${ASSAYLOOP_GROUND_TRUTH:-$REPO_ROOT/data/ground_truth}}"
RAW="$TARGET/_raw"
mkdir -p "$RAW"

STRING_BASE="https://stringdb-downloads.org/download"
CORUM_TXT="$RAW/allComplexes.txt"
SIGNOR_TSV="$RAW/signor_human_all.tsv"

echo "Ground truth target: $TARGET"
echo "Raw downloads:       $RAW"
echo

get() {  # get <url> <dest> <label>
  local url="$1" dest="$2" label="$3"
  if [[ -s "$dest" && $FORCE -eq 0 ]]; then
    echo "  [skip] $label ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  echo "  [get ] $label"
  # Download to a temp file so an interrupted transfer never leaves a
  # truncated input that the converter would happily parse.
  if curl -fsSL --retry 2 -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
    echo "         -> $(du -h "$dest" | cut -f1)"
    return 0
  fi
  rm -f "$dest.part"
  echo "         FAILED: $url" >&2
  return 1
}

echo "STRING v11.5 (CC BY 4.0)"
get "$STRING_BASE/protein.links.full.v11.5/9606.protein.links.full.v11.5.txt.gz" \
    "$RAW/9606.protein.links.full.v11.5.txt.gz" "protein.links.full (127M)"
get "$STRING_BASE/protein.info.v11.5/9606.protein.info.v11.5.txt.gz" \
    "$RAW/9606.protein.info.v11.5.txt.gz" "protein.info"
get "$STRING_BASE/protein.aliases.v11.5/9606.protein.aliases.v11.5.txt.gz" \
    "$RAW/9606.protein.aliases.v11.5.txt.gz" "protein.aliases"

echo
echo "CORUM 4.1"
if [[ -s "$CORUM_TXT" && $FORCE -eq 0 ]]; then
  echo "  [skip] allComplexes.txt already present"
elif get "https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip" \
         "$RAW/allComplexes.txt.zip" "allComplexes.txt.zip"; then
  unzip -o -q -d "$RAW" "$RAW/allComplexes.txt.zip"
else
  cat >&2 <<EOF
  CORUM could not be fetched automatically. Its site is a single-page app and
  the bulk path moves between releases.

    1. Open https://mips.helmholtz-muenchen.de/corum/ -> Download
    2. Take the "All complexes" table (allComplexes.txt, or its .zip)
    3. Save it as: $CORUM_TXT

  CORUM releases before 5.0 are CC BY 4.0; attribute it if you redistribute
  anything derived from it. This repository ships none of its data.
EOF
fi

echo
echo "SIGNOR 3.0"
if [[ -s "$SIGNOR_TSV" && $FORCE -eq 0 ]]; then
  echo "  [skip] signor_human_all.tsv already present"
elif ! get "https://signor.uniroma2.it/getData.php?organism=9606" \
           "$SIGNOR_TSV" "human interactions (all)"; then
  cat >&2 <<EOF
  SIGNOR could not be fetched automatically.

    1. Open https://signor.uniroma2.it/downloads.php
    2. Take the complete human dataset, tab-separated
    3. Save it as: $SIGNOR_TSV

  It must be the full export with ENTITYA / TYPEA / IDA / ENTITYB / ... columns,
  not a single-pathway download.
EOF
fi

echo
echo "Converting..."
PY="${PYTHON:-python3}"
"$PY" "$REPO_ROOT/scripts/convert_ground_truth.py" --raw "$RAW" --out "$TARGET"
