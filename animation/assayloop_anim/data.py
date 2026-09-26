"""Numbers behind the explainer.

Everything here is transcribed from the paper (arXiv:2609.11877), Table 1 and
Figure 4. Nothing is invented; where a per-round curve is reconstructed rather
than read off a data file, `CURVE_SOURCE` says so and the reconstruction is
pinned to the published endpoints.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# --- benchmark shape ------------------------------------------------------
N_SCREENS = 1389
N_TRAIN = 1349
N_VAL = 20
N_TEST = 20
N_ROUNDS = 10           # T
BATCH = 100             # b
BUDGET = N_ROUNDS * BATCH
LIBRARY = 20000         # order of magnitude of a genome-wide library
HANDOFF_K = 3           # rounds of LLM warm start, chosen on validation
PHENOTYPES = ["Fitness", "Drug", "Infection", "Molecular", "Trafficking"]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    ef: float            # enrichment factor over random
    nauc: float          # normalized AUC, %
    fh: float            # fraction of all hits found at budget, %


# Table 1, test split (20 temporally held-out screens).
TABLE1 = {
    m.key: m
    for m in [
        Method("random", "Random", 1.04, 3.4, 4.9),
        Method("prior_hit", "Prior hit baseline", 2.87, 11.5, 14.9),
        Method("screen_knn", "Screen-kNN", 3.40, 12.5, 17.3),
        Method("bpmf", "BPMF", 4.49, 12.3, 19.7),
        Method("maml", "MAML", 3.77, 14.3, 18.6),
        Method("biobo", "BioBO", 2.59, 5.9, 12.2),
        Method("poh", "Probability-of-hit", 2.41, 7.3, 11.8),
        Method("llmnn", "LLMNN", 2.39, 9.4, 12.5),
        Method("icbr", "ICBR-EF", 2.76, 9.3, 14.5),
        Method("haiku_agent", "Haiku-4.5 Agent", 3.20, 14.3, 16.8),
        Method("glm", "GLM-5.1", 4.00, 16.3, 21.0),
        Method("gemini", "Gemini 3.1 Pro", 4.71, 20.6, 24.6),
        Method("gpt", "GPT-5.6 Sol", 4.81, 20.5, 25.2),
        Method("assayllm", "AssayLLM", 3.69, 15.5, 19.3),
        Method("assayformer", "AssayFormer", 4.83, 17.2, 23.2),
        Method("assayloop", "AssayLoop", 5.67, 21.7, 27.7),
    ]
}

# Ablations and scaling (Figure 4B-E).
LOPO = {"assayformer": (4.83, 4.59), "screen_knn": (3.40, 2.98)}
ABLATION = {
    "AssayFormer": 4.83,
    "w/o screen description": 4.52,
    "w/o context-delta reward": 4.29,
}
TOKEN_INIT = {"Random": 2.47, "GenePT": 2.95, "K562": 3.15, "SVD": 4.02, "MF": 4.40, "BPMF": 4.83}
SCALING = [  # (n training screens, EF after RL fine-tuning)
    (1, 1.63), (2, 1.98), (5, 2.60), (10, 3.09), (20, 3.24),
    (50, 3.28), (100, 3.94), (200, 4.01), (400, 4.05),
    (800, 4.50), (1349, 4.83),
]

# Worked example: genome-wide screen for regulators of TNF-alpha-induced
# NF-kB activity in HeLa cells (Figure 3E-G).
NFKB = {
    "total_hits": 169,
    "recovered": 60,
    "per_round_hits": [8, 9, 7, 11, 8, 2, 6, 6, 3, 0],
    "round4_without_labels": 7,
    "round4_with_labels": 11,
    "highlight_round": 4,
}

# --- Figure 4A recall curves ---------------------------------------------
# Measured per-round recall, exported from the actual test-set runs by
# src/bridgeloop/scripts/export_recovery_curves.py and re-keyed to the paper's
# final method names. Round-10 values reproduce Table 1's Frac.-hits column
# exactly, which is what pins the mapping.
#
# One caveat carried through to the screen: Random is never written to disk
# (it is regenerated live per evaluation), so its curve is the only
# interpolated one. `is_measured()` reports that honestly.
_CURVE_FILE = os.path.join(os.path.dirname(__file__), "curves.json")

with open(_CURVE_FILE) as _fh:
    _CURVES = json.load(_fh)

CURVE_SOURCE = _CURVES["source"]
MEASURED = set(_CURVES["measured"])
ANALYTIC = {"random"}


def is_measured(key: str) -> bool:
    return key in MEASURED


def random_expectation(n: int = N_ROUNDS) -> np.ndarray:
    """Recall of a uniformly random policy, after each round.

    Not a fit and not a measured run: a random draw recovers hits in
    proportion to the fraction of the library it has queried, so recall is
    linear in the budget spent and exact by construction. The paper's
    Figure 4A draws the same thing as the identity line y = x.
    """
    return np.linspace(0.0, TABLE1["random"].fh / 100.0, n + 1)


def recall_curves() -> dict[str, np.ndarray]:
    """Fraction of all hits recovered after each round, index 0..10."""
    out = {k: np.asarray(v, dtype=float) for k, v in _CURVES["curves"].items()}
    out["random"] = random_expectation()
    return out


def recall_sem() -> dict[str, np.ndarray]:
    """Standard error across the 20 test screens, same indexing."""
    return {k: np.asarray(v, dtype=float) for k, v in _CURVES["sem"].items()}


def effective_fraction() -> dict[str, np.ndarray]:
    """Per-method effective fraction of the library acquired after each round.

    Methods differ here because picks that fall outside a screen's library are
    forgiven rather than charged, so this is *not* a shared x-grid. The
    explainer plots against round index instead, and only uses these to state
    that ten rounds lands at roughly 5% of the library.
    """
    return {k: np.asarray(v, dtype=float) for k, v in _CURVES["x_effective"].items()}


def handoff_split(curve: np.ndarray, k: int = HANDOFF_K):
    """Split a curve at the handoff so the warm start and the policy can be
    drawn as two strokes. In the measured data AssayLoop is identical to the
    LLM through round k, so the green trace genuinely overlaps yellow here."""
    return curve[: k + 1], curve[k:]
