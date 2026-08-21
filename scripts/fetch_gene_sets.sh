#!/usr/bin/env bash
# Download the third-party gene-set and interaction assets that assayloop
# does NOT redistribute. Run this once before the pathway-diversity metric,
# the pathway sunburst figure, or the Figure 9 network-recovery analysis.
#
# Nothing downloads these for you at import time, and nothing substitutes a
# degraded stand-in: a consumer that needs a missing file raises and names
# what to fetch.
#
# WHAT IS AND IS NOT SHIPPED
#
# No MSigDB collection is committed here. The three the paper uses are
# registered assets in assaybench, checksummed and fetched on demand:
#
#   assaybench download msigdb-hallmark              # h.all, 48K
#   assaybench download msigdb-go-bp                 # c5.go.bp, 4.8M
#   assaybench download msigdb-canonical-pathways    # c2.cp, 1.5M
#
# This script downloads the same three into the local gene-set directory (which
# takes precedence over the assaybench cache) so one command sets up the repo,
# plus the Reactome files, which are not in the assaybench registry.
#
# Committed in the repo:
#   reactome_two_level.json        Small derived hierarchy, built from the
#                                  Reactome files below.
#
# Terms worth knowing before you run this:
#   h.all     MSigDB Hallmark. CC BY 4.0. Attribution: Broad Institute, Inc.,
#             Massachusetts Institute of Technology, and Regents of the
#             University of California.
#   c2.cp     MSigDB curated canonical pathways. NOT redistributable as a
#             single file: it mixes 186 KEGG legacy sets and 292 BioCarta
#             sets that the Broad holds only under "qualified permission"
#             with 619 KEGG_MEDICUS sets under CC BY-SA 4.0. Downloading it
#             yourself from MSigDB is how you accept those terms.
#   c5.go.bp  MSigDB GO biological process. CC BY 4.0.
#   Reactome  Pathway hierarchy + the 56M human interactor table. CC BY 4.0.
#
# Default target: $ASSAYLOOP_GENE_SETS, falling back to
#   <repo>/src/assayloop/data/gene_sets/
#
# Re-running is idempotent: a file is re-downloaded only if missing or empty.
# Pass --force to re-download regardless.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ASSAYLOOP_GENE_SETS:-$REPO_ROOT/src/assayloop/data/gene_sets}"
MSIGDB="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs"
REACTOME="https://reactome.org/download/current"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# name -> url. sha256 of the exact copy used for the paper is listed in
# EXPECTED below; a mismatch is a warning, not an error, because Reactome's
# /download/current/ rolls over with each quarterly release.
declare -A URLS=(
  ["h.all.v2023.2.Hs.symbols.gmt"]="$MSIGDB/h.all.v2023.2.Hs.symbols.gmt"
  ["c5.go.bp.v2023.2.Hs.symbols.gmt"]="$MSIGDB/c5.go.bp.v2023.2.Hs.symbols.gmt"
  ["c2.cp.v2023.2.Hs.symbols.gmt"]="$MSIGDB/c2.cp.v2023.2.Hs.symbols.gmt"
  ["ReactomePathways.txt"]="$REACTOME/ReactomePathways.txt"
  ["ReactomePathwaysRelation.txt"]="$REACTOME/ReactomePathwaysRelation.txt"
  ["reactome_interactions.txt"]="$REACTOME/interactors/reactome.homo_sapiens.interactions.tab-delimited.txt"
)

declare -A EXPECTED=(
  ["h.all.v2023.2.Hs.symbols.gmt"]="8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780"
  ["c5.go.bp.v2023.2.Hs.symbols.gmt"]="b4b3b49a267c2239fad40d39a56f2146a57743992943915e80ebed34ec7ebaae"
  ["c2.cp.v2023.2.Hs.symbols.gmt"]="4826118ed2df8af485d524395425a77d5563681417097bc38c3393640a5bde54"
  ["ReactomePathways.txt"]="f6d7a2bf89b5bcfe0250a0bc7f51bff94641447911712b8ff129f5b55e52df3a"
  ["ReactomePathwaysRelation.txt"]="fd49a624d80c14eb37ae57a02e141d574d5ede3f60022bb99edbd909448a3f1e"
  ["reactome_interactions.txt"]="a5ed8376d7da82b0dbfc6f0d243b422b2635482025b74e434d3b9be48cb55e5d"
  ["ReactomePathways.gmt"]="89983d5c1f0af11c52edfeee7323eb425580ac6281d387a528562ab1787ce56b"
)

mkdir -p "$TARGET"
echo "Fetching gene-set assets into: $TARGET"

check_sha() {
  local name="$1" dest="$2"
  local want="${EXPECTED[$name]:-}"
  [[ -z "$want" ]] && return 0
  local got
  got="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [[ "$got" != "$want" ]]; then
    echo "         WARNING: sha256 differs from the copy used for the paper."
    echo "                  expected $want"
    echo "                  got      $got"
    echo "                  Reactome's /download/current/ advances every release;"
    echo "                  network-recovery numbers may shift slightly."
  fi
}

for name in "${!URLS[@]}"; do
  dest="$TARGET/$name"
  if [[ -s "$dest" && $FORCE -eq 0 ]]; then
    echo "  [skip] $name already present"
    continue
  fi
  echo "  [get ] $name"
  curl -fsSL -o "$dest" "${URLS[$name]}"
  echo "         -> $(du -h "$dest" | cut -f1)"
  check_sha "$name" "$dest"
done

# ReactomePathways.gmt ships zipped; unpack it.
gmt="$TARGET/ReactomePathways.gmt"
if [[ -s "$gmt" && $FORCE -eq 0 ]]; then
  echo "  [skip] ReactomePathways.gmt already present"
else
  echo "  [get ] ReactomePathways.gmt (from ReactomePathways.gmt.zip)"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/rp.zip" "$REACTOME/ReactomePathways.gmt.zip"
  unzip -p "$tmp/rp.zip" > "$gmt"
  echo "         -> $(du -h "$gmt" | cut -f1)"
  check_sha "ReactomePathways.gmt" "$gmt"
fi

echo
echo "Done. Default pathway source: c5.go.bp (override with ASSAYLOOP_GENE_SETS_SOURCE)."
echo "MSigDB terms: https://www.gsea-msigdb.org/gsea/msigdb"
echo "Reactome: CC BY 4.0, https://reactome.org"
