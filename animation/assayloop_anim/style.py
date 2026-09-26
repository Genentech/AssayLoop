"""Visual language for the AssayLoop explainer.

Deliberately close to the 3Blue1Brown palette: a near-black ground, saturated
but slightly desaturated primaries, and a Computer-Modern-ish serif so that
text sits next to LaTeX-style math without a seam.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# --- canvas ---------------------------------------------------------------
W, H = 1920, 1080
DPI = 100
FIG_W, FIG_H = W / DPI, H / DPI
FPS = 60

# --- palette (manim's, with a couple of custom mixes) ---------------------
BG = "#0B0E13"
WHITE = "#ECECEC"
GREY = "#7A8290"
GREY_D = "#3A414E"

BLUE = "#58C4DD"
BLUE_D = "#3A8FAB"
TEAL = "#5CD0B3"
GREEN = "#83C167"
YELLOW = "#FFD24A"
GOLD = "#F0AC5F"
RED = "#FC6255"
MAROON = "#C55F73"
PURPLE = "#9A72AC"

# Semantic roles. Keeping these named by meaning (not hue) means a palette
# change never turns into a hunt through the scene code.
C_RANDOM = GREY
C_LLM = YELLOW          # biological prior / language model
C_FORMER = BLUE         # the learned, feedback-conditioned policy
C_LOOP = GREEN          # the combination
C_HIT = RED             # a true hit
C_MISS = GREY_D         # an assayed non-hit
C_ACCENT = GOLD

# --- type -----------------------------------------------------------------
# STIXGeneral is the only serif here with full bold/italic + the punctuation
# and Greek we need (NF-kB, arrows, times sign).
SERIF = "STIXGeneral"
MONO = "DejaVu Sans Mono"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [SERIF, "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": WHITE,
        "axes.edgecolor": GREY_D,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "font.size": 22,
    }
)

# Type scale, in points at DPI=100.
T_TITLE = 60
T_SUB = 36
T_BODY = 30
T_CAPTION = 26
T_SMALL = 21
T_TINY = 17
