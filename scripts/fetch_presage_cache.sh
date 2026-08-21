#!/usr/bin/env bash
# Download and unpack the PRESAGE gene-embedding cache (3.4 GB).
#
# Source: https://github.com/Genentech/PRESAGE
# Data:   https://zenodo.org/records/15587986/files/cache.tar.gz
# License note: pathway / other_embeddings subdirs are CC BY-NC 4.0.
#
# Default target: $ASSAYLOOP_PRESAGE_CACHE, falling back to
#   <repo>/src/assayloop/data/presage_cache/
#
# The unpacked layout is:
#   <target>/data/
#   <target>/splits/
#   <target>/configs/
#   <target>/pathway_embeddings/<source>.embeddings.pkl
#   <target>/other_embeddings/<source>.embeddings.pkl
#
# Re-running is idempotent: re-download is skipped if cache.tar.gz already
# exists with the expected size; unpack is skipped if expected subdirs
# are already present.

set -euo pipefail

ZENODO_URL="https://zenodo.org/records/15587986/files/cache.tar.gz?download=1"
DEFAULT_TARGET="src/assayloop/data/presage_cache"
TARGET="${ASSAYLOOP_PRESAGE_CACHE:-$DEFAULT_TARGET}"
TAR_PATH="${TARGET}/cache.tar.gz"

mkdir -p "$TARGET"
cd "$TARGET"

if [[ ! -f cache.tar.gz ]]; then
    echo "[presage] downloading cache.tar.gz to $(pwd) (3.4 GB) ..."
    wget --no-verbose -c -O cache.tar.gz "$ZENODO_URL"
else
    echo "[presage] cache.tar.gz already present in $(pwd), skipping download"
fi

if [[ ! -d data || ! -d splits || ! -d pathway_embeddings ]]; then
    echo "[presage] unpacking cache.tar.gz ..."
    tar -xzf cache.tar.gz
    # PRESAGE's unpack_cache.sh moves subdirs up one level.
    if [[ -d cache ]]; then
        mv -nT cache/data data 2>/dev/null || true
        mv -nT cache/splits splits 2>/dev/null || true
        mv -nT cache/configs configs 2>/dev/null || true
        # Some bundles ship pathway/other_embeddings nested under cache/.
        for sub in pathway_embeddings other_embeddings; do
            if [[ -d "cache/$sub" && ! -d "$sub" ]]; then
                mv "cache/$sub" "$sub"
            fi
        done
        rmdir cache 2>/dev/null || true
    fi
    # Gunzip any json.gz files (PRESAGE convention).
    find . -name '*.json.gz' -print0 | xargs -0 -r -n 64 gunzip -f
else
    echo "[presage] looks already unpacked in $(pwd); skipping tar extract"
fi

echo "[presage] done. Cache root: $(pwd)"
echo "[presage] set ASSAYLOOP_PRESAGE_CACHE=$(pwd) in your .env to use it."
