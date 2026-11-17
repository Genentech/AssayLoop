# Publishing the cached result bundles

The table fetcher expects two assets on the GitHub release `data-v1`:

- `assayloop-sweeps-v1.tar.gz`
- `assayloop-llm-calls-v1.tar.gz`

Build them from the original result roots, using the public checkout:

```bash
uv run python scripts/build_sweep_bundle.py \
  --results /path/to/results --shared /path/to/shared-results \
  --with-llm-calls --out dist/data-v1
```

The inputs must contain `sweeps/` and `runs/`. The builder refuses unresolved
rows and missing result files. It removes credential values, internal hosts,
and internal paths; verifies that metrics, acquisition lists, observations,
and parsed response genes are unchanged; and scans for remaining known secret
patterns. Manifests record the checks and source sweep IDs. Review the staged
files before publication: pattern scans cannot identify every possible secret.

Historical prompt text is preserved apart from these redactions. Differences
from the current prompt templates are provenance, not edits to be hidden.
Saved-response replay preserves the collected experiment; it does not imply
that a fresh API call with a different prompt produces equivalent results.

Copy the printed SHA-256 hashes into `scripts/fetch_sweeps.sh`. Test both
archives together using a local mirror:

```bash
ASSAYLOOP_BUNDLE_URL="file://$PWD/dist/data-v1" \
ASSAYLOOP_PUBLISHED=/tmp/assayloop-bundle-check \
  bash scripts/fetch_sweeps.sh
```

Publish the two exact archives as release assets and verify a clean download
using the default URL. Keep the hashes and assets together: changing only the
fetcher does not fix a missing GitHub release. Do not replace already-published
immutable assets in place; use a new version for later changes.
