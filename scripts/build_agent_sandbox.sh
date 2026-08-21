#!/bin/bash
# Build the Apptainer sandbox the `agent_ranker` baseline executes code in,
# with the exported public training screens baked in at /data/screens.json.
#
# The image is a stock python:3.11-slim plus numpy/scipy/pandas/scikit-learn
# and that one JSON file, so it is fully reproducible from this repo -- nothing
# is shipped or downloaded from us.
#
# Usage:
#     bash scripts/build_agent_sandbox.sh
#     bash scripts/build_agent_sandbox.sh path/to/screens.json
#
# Produces: output/containers/agent_sandbox.sif
# Override with ASSAYLOOP_AGENT_SANDBOX_SIF (the model reads the same variable).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-${REPO_ROOT}/output/datasets/public_train_screens.json}"
SIF_PATH="${ASSAYLOOP_AGENT_SANDBOX_SIF:-${REPO_ROOT}/output/containers/agent_sandbox.sif}"
DEF_FILE=$(mktemp /tmp/agent_sandbox_XXXXXX.def)

if ! command -v apptainer >/dev/null 2>&1; then
    echo "error: apptainer is not on PATH. Install it from https://apptainer.org"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "error: dataset not found at $DATASET"
    echo "Run: uv run python -m assayloop.scripts.export_screen_dataset"
    exit 1
fi

mkdir -p "$(dirname "$SIF_PATH")"

DATASET_ABS=$(realpath "$DATASET")

cat > "$DEF_FILE" <<EOF
Bootstrap: docker
From: python:3.11-slim

%files
    ${DATASET_ABS} /data/screens.json

%post
    pip install --no-cache-dir numpy scipy pandas scikit-learn

%runscript
    exec python3 "\$@"
EOF

echo "=== Building Apptainer sandbox ==="
echo "Dataset: $DATASET ($(du -h "$DATASET" | cut -f1))"
echo "Definition: $DEF_FILE"
echo "Output: $SIF_PATH"
echo ""

apptainer build --force "$SIF_PATH" "$DEF_FILE"
rm -f "$DEF_FILE"

echo ""
echo "=== Verifying ==="
apptainer exec --contain --network none "$SIF_PATH" \
    python3 -c "import json; d=json.load(open('/data/screens.json')); print(f'OK: {len(d)} screens loaded from /data/screens.json')"

echo ""
echo "Container ready: $SIF_PATH ($(du -h "$SIF_PATH" | cut -f1))"
