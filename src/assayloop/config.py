"""assayloop configuration loaded from environment variables.

Every machine-local path and credential the package reads is declared
here, so a fresh checkout can be pointed at new locations by setting
environment variables (or a ``.env`` at the project root) rather than by
editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Default LLM provider for assayloop: "vllm" (matches BridgeBuilder),
# "anthropic", or "dspy".
LLM_PROVIDER: str = os.getenv("ASSAYLOOP_LLM_PROVIDER", "vllm")

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "glm-5")
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "not-needed")


def _opt_float(name: str) -> float | None:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else None


def _opt_int(name: str) -> int | None:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else None


def _opt_path(name: str) -> Path | None:
    v = os.getenv(name, "").strip()
    return Path(v).expanduser() if v else None


class MissingConfiguredPath(FileNotFoundError):
    """Raised when a script needs an external artefact that is unset or absent.

    These are things the repo cannot ship: a trained checkpoint, a shared
    results directory, a third-party ground-truth dump. The rule is that a
    missing one stops the script and names the environment variable, rather
    than producing a figure or a table computed from whatever happened to be
    lying around.
    """


def require_path(value: Path | None, *, env_var: str, what: str, hint: str = "") -> Path:
    """Return ``value``, or raise :class:`MissingConfiguredPath` naming ``env_var``."""
    suffix = f" {hint}" if hint else ""
    if value is None:
        raise MissingConfiguredPath(f"{what} is not configured. Set {env_var}=<path>.{suffix}")
    if not value.exists():
        raise MissingConfiguredPath(f"{what} not found at {value} (from {env_var}).{suffix}")
    return value


# Optional sampling overrides for the vLLM endpoint. Left unset they are
# NOT sent on the request, so the server's own defaults apply and existing
# runs (GLM-5 tuned for temperature=1.0 only) are unchanged. Set them (e.g.
# in .env) to pin a model's recommended preset — e.g. Qwen3's thinking-mode
# sampling: VLLM_TOP_P=0.95, VLLM_TOP_K=20, VLLM_MIN_P=0.0,
# VLLM_PRESENCE_PENALTY=0.0, VLLM_REPETITION_PENALTY=1.0 (temperature is set
# via the existing LLMClientConfig.temperature, default 1.0).
# ``top_p`` / ``presence_penalty`` are standard OpenAI params; ``top_k`` /
# ``min_p`` / ``repetition_penalty`` are vLLM extensions sent via extra_body.
VLLM_TOP_P: float | None = _opt_float("VLLM_TOP_P")
VLLM_TOP_K: int | None = _opt_int("VLLM_TOP_K")
VLLM_MIN_P: float | None = _opt_float("VLLM_MIN_P")
VLLM_PRESENCE_PENALTY: float | None = _opt_float("VLLM_PRESENCE_PENALTY")
VLLM_REPETITION_PENALTY: float | None = _opt_float("VLLM_REPETITION_PENALTY")

# ---------------------------------------------------------------------------
# Paths (machine-local, gitignored)
# ---------------------------------------------------------------------------

SCRATCH_PATH: Path = _PROJECT_ROOT / os.getenv("SCRATCH_PATH", "scratch")
OUTPUT_PATH: Path = _PROJECT_ROOT / "output"

# Shared (cross-user) results directory. Sweeps "pushed" here become visible
# to every teammate pointing at the same path, so a team can share baseline
# sweeps. Layout mirrors OUTPUT_PATH: ``<SHARED_PATH>/sweeps/<sweep_id>/
# sweep.json`` and ``<SHARED_PATH>/runs/<run_id>/result.json``. Unset by
# default -- there is no shared directory unless you configure one. Not
# created at import time (it may live on read-only storage). Defaults to a
# local subdirectory, so an unconfigured checkout has no shared directory at
# all rather than pointing at somebody else's filesystem.
SHARED_PATH: Path = Path(
    os.getenv("ASSAYLOOP_SHARED_PATH", str(OUTPUT_PATH / "shared"))
)

# Where the paper's analysis and figure scripts look for finished runs and
# sweeps. Defaults to this repo's own OUTPUT_PATH, which is exactly where
# ``assayloop run`` and ``assayloop sweep`` write them. Point it at a shared
# directory (or at SHARED_PATH) to plot someone else's runs.
RESULTS_PATH: Path = Path(os.getenv("ASSAYLOOP_RESULTS", str(OUTPUT_PATH)))

# Downloaded published sweeps (scripts/fetch_sweeps.sh). Same layout as
# RESULTS_PATH: ``<PUBLISHED_PATH>/sweeps/<sweep_id>/sweep.json`` and
# ``<PUBLISHED_PATH>/runs/<run_id>/result.json``.
#
# These are the LLM and external-baseline rows of the full-genome table. They
# cost ~400 paid API runs across nine vendors' models, several since retired,
# so unlike the AssayFormer/BPMF rows they cannot be regenerated from a
# checkpoint. Nothing downloads them implicitly and nothing substitutes a
# stand-in: a table row whose sweep is absent raises and names the fetch
# script rather than printing "--".
PUBLISHED_PATH: Path = Path(
    os.getenv("ASSAYLOOP_PUBLISHED", str(OUTPUT_PATH / "published"))
)

PUBLISHED_HINT = (
    "Run scripts/fetch_sweeps.sh to download the published sweep bundle "
    "(~5.5 MB), or set ASSAYLOOP_PUBLISHED to where you unpacked it."
)

# BPMF checkpoint (a ``bpmf_result.pkl`` from scripts/train_bpmf_gpu.py).
# Defaults inside OUTPUT_PATH; there is no bundled checkpoint, and nothing
# falls back to a random factorisation when it is missing.
BPMF_CHECKPOINT: Path = Path(
    os.getenv("ASSAYLOOP_BPMF_CHECKPOINT", str(OUTPUT_PATH / "bpmf" / "bpmf_result.pkl"))
)

# Third-party interaction ground truth (STRING / CORUM / SIGNOR parquet
# tables) for the Figure 9 network-recovery analysis. No default: these are
# not redistributed with the repo and there is no single canonical layout.
# Expected subdirectories: STRING-HUMAN/, CORUM-HUMAN/, SIGNOR-HUMAN/.
GROUND_TRUTH_PATH: Path | None = _opt_path("ASSAYLOOP_GROUND_TRUTH")

GROUND_TRUTH_HINT = (
    "Expected subdirectories: STRING-HUMAN/, CORUM-HUMAN/, SIGNOR-HUMAN/. "
    "See README 'Interaction ground truth'."
)


def ground_truth_dir(name: str) -> Path:
    """Return ``<ASSAYLOOP_GROUND_TRUTH>/<name>``, or raise naming the env var.

    Callers must not degrade gracefully when this is missing: an absent
    source turns into an empty edge set or an unlabelled figure, which reads
    as a result rather than as a missing input.
    """
    root = require_path(
        GROUND_TRUTH_PATH,
        env_var="ASSAYLOOP_GROUND_TRUTH",
        what="the interaction ground-truth directory",
        hint=GROUND_TRUTH_HINT,
    )
    return require_path(
        root / name,
        env_var="ASSAYLOOP_GROUND_TRUTH",
        what=f"the {name} interaction tables",
        hint=GROUND_TRUTH_HINT,
    )

# JSONL prediction dumps from the fine-tuned LLM (AssayLLM base / SFT / GRPO
# and the handoff traces) that the paper's tables read. Produced by the
# training pipeline, not committed. Rows whose file is absent are reported as
# "--" in the generated table and logged as NOT FOUND.
LLM_PREDICTIONS_PATH: Path = Path(
    os.getenv("ASSAYLOOP_LLM_PREDICTIONS", str(OUTPUT_PATH / "llm_predictions"))
)

SKILLS_PATH: Path = Path(__file__).resolve().parent / "skills"

# PRESAGE gene-embedding cache (3.4 GB, downloaded via scripts/fetch_presage_cache.sh).
_DEFAULT_PRESAGE = Path(__file__).resolve().parent / "data" / "presage_cache"
PRESAGE_CACHE_PATH: Path = Path(os.getenv("ASSAYLOOP_PRESAGE_CACHE", str(_DEFAULT_PRESAGE)))
ORTHOLOG_CACHE_PATH: Path = Path(__file__).resolve().parent / "data" / "ortholog_cache"

# MSigDB gene-set (.gmt) membership files for pathway-based batch diversity,
# downloaded via scripts/fetch_gene_sets.sh.
_DEFAULT_GENE_SETS = Path(__file__).resolve().parent / "data" / "gene_sets"
GENE_SETS_PATH: Path = Path(os.getenv("ASSAYLOOP_GENE_SETS", str(_DEFAULT_GENE_SETS)))
# Default MSigDB collection used for pathway diversity (file basename without
# the .gmt extension). GO biological process gives the broadest gene coverage.
GENE_SETS_SOURCE: str = os.getenv("ASSAYLOOP_GENE_SETS_SOURCE", "c5.go.bp.v2023.2.Hs.symbols")

# ---------------------------------------------------------------------------
# Create directories at import time
# ---------------------------------------------------------------------------

SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
