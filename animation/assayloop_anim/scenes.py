"""The storyboard.

Every scene is a function `(canvas, t, T) -> None` that draws the single frame
at absolute time `t`. There is no retained state between frames, which makes
any moment independently renderable (handy for previews) at the cost of
recomputing a little geometry each time.

`T` is a `Timing`, which exposes the *measured* voiceover beat boundaries, so
the visuals are pinned to the audio rather than to guessed durations.
"""

from __future__ import annotations

import numpy as np

from . import data as D
from . import style as S
from .canvas import Canvas, mix
from .ease import (ease_out_back, fade_in_out, seg, smooth, smoother, stagger,
                   there_and_back)

RNG = np.random.default_rng(7)


class Timing:
    """Beat boundaries read off the synthesized narration."""

    def __init__(self, schedule: list[dict]):
        self.b = schedule
        self.total = schedule[-1]["end"] + schedule[-1]["pad"]

    def s(self, i: int) -> float:
        return self.b[i]["start"]

    def e(self, i: int) -> float:
        return self.b[i]["end"]

    def span(self, i: int) -> tuple[float, float]:
        return self.b[i]["start"], self.b[i]["end"]

    def of(self, scene: str) -> list[tuple[float, float]]:
        """Beat windows for one scene, in order.

        Scenes bind their beats by name so that inserting a beat anywhere in
        the script cannot silently re-point a later scene at the wrong audio.
        """
        return [(b["start"], b["end"]) for b in self.b if b["scene"] == scene]

    def scene_of(self, t: float) -> str:
        cur = self.b[0]["scene"]
        for beat in self.b:
            if t >= beat["start"]:
                cur = beat["scene"]
        return cur


# ---------------------------------------------------------------------------
# shared plot frame — scenes 3-5 all draw into the same axes so the curves
# persist across the cuts instead of being rebuilt each time
# ---------------------------------------------------------------------------
PX0, PY0, PW, PH = -5.0, -2.75, 9.4, 5.15
YTOP = 0.30


def to_xy(round_idx, recall):
    x = PX0 + PW * np.asarray(round_idx, float) / D.N_ROUNDS
    y = PY0 + PH * np.asarray(recall, float) / YTOP
    return x, y


def plot_frame(c: Canvas, a: float, rounds_alpha: float = 0.0,
               labels: bool = True) -> None:
    """Axes, ticks and labels for the recovery curve.

    The x axis is the fraction of the library sampled, which is how the paper
    frames it. Each of the ten rounds spends 100 genes of a ~20,000-gene
    library, so a round is exactly half a percent and the two readings sit on
    the same ticks; `rounds_alpha` fades in the round numbering where the
    handoff needs it.
    """
    if a <= 0.002:
        return
    c.axes2d(PX0, PY0, PW, PH, alpha=a)
    for r in range(0, D.N_ROUNDS + 1):
        x, _ = to_xy(r, 0)
        major = r % 2 == 0
        c.line(x, PY0, x, PY0 - (0.10 if major else 0.06), color=S.GREY_D,
               lw=1.6 if major else 1.1, alpha=a)
        if major:
            c.text(x, PY0 - 0.34, f"{r * 0.5:.0f}%" if r % 4 == 0 else f"{r*0.5:g}%",
                   size=S.T_SMALL, color=S.GREY, alpha=a)
    if labels:
        c.text(PX0 + PW / 2, PY0 - 0.92, "fraction of the library sampled",
               size=S.T_CAPTION, color=S.GREY, alpha=a)
    if rounds_alpha > 0.01:
        for r in range(0, D.N_ROUNDS + 1, 2):
            x, _ = to_xy(r, 0)
            c.text(x, PY0 - 1.32, f"{r}", size=S.T_TINY, color=S.GREY_D,
                   alpha=a * rounds_alpha)
        c.text(PX0 - 0.30, PY0 - 1.32, "round", size=S.T_TINY, color=S.GREY_D,
               alpha=a * rounds_alpha, ha="right")
    for v in (0.0, 0.10, 0.20, 0.30):
        _, y = to_xy(0, v)
        c.line(PX0, y, PX0 - 0.10, y, color=S.GREY_D, lw=1.6, alpha=a)
        c.text(PX0 - 0.30, y, f"{v*100:.0f}%", size=S.T_SMALL,
               color=S.GREY, alpha=a, ha="right")
    if labels:
        c.text(PX0 - 1.08, PY0 + PH / 2, "fraction of all\nhits found",
               size=S.T_CAPTION, color=S.GREY, alpha=a, rotation=90)


def curve_xy(key: str):
    y = D.recall_curves()[key]
    return to_xy(np.arange(len(y)), y)


# ---------------------------------------------------------------------------
# 1. the library, the assay, the hits, the budget
# ---------------------------------------------------------------------------
NCOL, NROW = 200, 100                      # 20,000 genes, one dot each
_GX, _GY = np.meshgrid(np.arange(NCOL), np.arange(NROW))
GRID_X = -6.4 + 12.8 * (_GX.ravel() + 0.5) / NCOL
GRID_Y = -2.35 + 5.0 * (_GY.ravel() + 0.5) / NROW
N_GENES = GRID_X.size
HIT_IDX = RNG.choice(N_GENES, size=180, replace=False)      # ~0.9%, as in a real screen
PICK_IDX = RNG.choice(N_GENES, size=1000, replace=False)    # the 5% budget
_ORDER = np.argsort(GRID_X)                                 # left-to-right wash-in


def draw_grid(c: Canvas, appear: float = 1.0, dim: float = 0.0) -> None:
    """The library of candidate genes. Shared by the scenes that sit on it."""
    n = int(N_GENES * appear)
    if n <= 0:
        return
    vis = _ORDER[:n]
    c.dots(GRID_X[vis], GRID_Y[vis], size=7.0,
           color=mix(S.GREY, S.GREY_D, dim), alpha=0.9)


def scene_intro(c: Canvas, t: float, T: Timing) -> None:
    (b0,) = T.of("intro")
    out = seg(t, b0[1] - 0.2, b0[1] + 0.3)
    c.fade_text(0, 3.62, "A genome-wide CRISPR screen",
                fade_in_out(t, b0[0] + 0.2, b0[0] + 1.6, b0[1] - 0.2, b0[1] + 0.3),
                size=S.T_TITLE)
    c.fade_text(0, 2.92, "asks a question of every gene",
                fade_in_out(t, b0[0] + 0.7, b0[0] + 2.2, b0[1] - 0.2, b0[1] + 0.3),
                size=S.T_SUB, color=S.GREY)
    draw_grid(c, seg(t, b0[0] + 1.4, b0[0] + 4.4, smoother))
    c.fade_text(0, -3.30, f"{N_GENES:,} candidate genes",
                seg(t, b0[0] + 3.6, b0[0] + 4.8) * (1 - out),
                size=S.T_SUB, color=S.WHITE)


def scene_screen(c: Canvas, t: float, T: Timing) -> None:
    """What a CRISPR screen actually does, on one worked example."""
    b1, b2, b3 = T.of("screen")

    c.fade_text(0, 3.62, "What is a CRISPR screen?",
                fade_in_out(t, b1[0] + 0.1, b1[0] + 1.2, b3[0] + 0.1, b3[0] + 0.6),
                size=S.T_TITLE)
    c.fade_text(0, 2.88, "which genes control a cellular behaviour or trait",
                fade_in_out(t, b1[0] + 1.8, b1[0] + 3.0, b3[0] + 0.1, b3[0] + 0.6),
                size=S.T_SUB, color=S.GREY)

    # Everything the worked example puts on screen clears out before the grid
    # comes back, so the two never share the same space.
    gone = seg(t, b3[0] + 0.1, b3[0] + 0.8)

    # the question, in the form a real screen poses it
    card = fade_in_out(t, b1[0] + 2.6, b1[0] + 3.6, b3[0] + 0.1, b3[0] + 0.8)
    if card > 0.01:
        # Box spans 0.50..2.14; the kicker sits above it, not on the edge.
        c.rect(0.0, 1.32, 8.6, 1.64, color=S.GREY_D, lw=1.8, alpha=card,
               fill=mix(S.BG, S.WHITE, 0.03), radius=0.06)
        c.text(0.0, 2.38, "a screen asks one question", size=S.T_SMALL,
               color=S.GREY, alpha=card)
        for i, ln in enumerate(["Which genes, when activated in neuroblastoma,",
                                "confer resistance to RSL3?"]):
            c.typewriter(0.0, 1.72 - 0.40 * i, ln,
                         seg(t, b1[0] + 3.3 + 0.55 * i, b1[0] + 4.5 + 0.55 * i),
                         size=S.T_CAPTION, color=S.WHITE, alpha=card)
        # RSL3 inhibits GPX4, so the readout is ferroptosis resistance.
        c.fade_text(0.0, 0.86, "RSL3 triggers ferroptosis  —  iron-mediated cell death",
                    seg(t, b1[0] + 4.6, b1[0] + 5.6) * card, size=S.T_SMALL,
                    color=S.C_HIT)

    # set up the assay -> perturb one gene -> did the readout move?
    steps = [("set up the assay", "neuroblastoma cells", S.GREY),
             ("switch one gene off, or on", "CRISPR perturbation", S.C_FORMER),
             ("did the outcome change?", "if yes, it is a hit", S.C_HIT)]
    for i, (lab, sub, col) in enumerate(steps):
        p = stagger(i, 3, t, b2[0] + 0.8, b2[0] + 5.6, overlap=0.42) * (1.0 - gone)
        if p <= 0.01:
            continue
        x = -4.55 + i * 4.55
        c.rect(x, -0.95, 3.85, 1.40, color=col, lw=2.0, alpha=p,
               fill=mix(S.BG, col, 0.10), radius=0.06)
        c.text(x, -0.70, lab, size=S.T_CAPTION, color=col, alpha=p)
        c.text(x, -1.22, sub, size=S.T_SMALL, color=S.GREY, alpha=p)
        if i:
            a = stagger(i - 1, 2, t, b2[0] + 1.9, b2[0] + 5.4, overlap=0.4)
            c.arrow(x - 2.62, -0.95, x - 1.98, -0.95, color=S.GREY, lw=2.0,
                    alpha=a * (1.0 - gone))

    # ...and then across the whole genome
    if gone > 0.01:
        draw_grid(c, seg(t, b3[0] + 0.7, b3[0] + 2.9, smoother))
        c.fade_text(0, 3.62, "Repeat for every gene", seg(t, b3[0] + 0.4, b3[0] + 1.4),
                    size=S.T_TITLE)
        c.fade_text(0, -3.30, "20,000 experiments, one per gene",
                    seg(t, b3[0] + 2.4, b3[0] + 3.4), size=S.T_SUB, color=S.WHITE)


def scene_hits(c: Canvas, t: float, T: Timing) -> None:
    b4, b5 = T.of("hits")
    draw_grid(c)

    # The reveal is held for the whole explanation, then deliberately taken
    # away again: not knowing which ones they are is the point.
    hits_in = seg(t, b4[0] + 0.4, b4[0] + 1.8, smoother)
    hits = hits_in * (1.0 - seg(t, b5[0] + 1.6, b5[0] + 3.2))
    if hits > 0.01:
        n_hit = int(HIT_IDX.size * hits_in)
        c.dots(GRID_X[HIT_IDX[:n_hit]], GRID_Y[HIT_IDX[:n_hit]], size=26,
               color=S.C_HIT, alpha=hits, zorder=9)

    c.fade_text(0, 3.62, "What is a hit?",
                fade_in_out(t, b4[0] + 0.1, b4[0] + 1.1, b5[0] - 0.5, b5[0] + 0.1),
                size=S.T_TITLE, color=S.C_HIT)
    c.fade_text(0, -3.30, "perturb it, and the phenotype changes",
                fade_in_out(t, b4[0] + 2.2, b4[0] + 3.4, b4[0] + 7.0, b4[0] + 7.8),
                size=S.T_SUB, color=S.C_HIT)
    c.fade_text(0, -3.30, "only about 1 in 100 are hits",
                fade_in_out(t, b4[0] + 7.6, b4[0] + 8.6, b5[0] + 0.6, b5[0] + 1.4),
                size=S.T_SUB, color=S.C_HIT)
    c.fade_text(0, 3.62, "But you cannot see them",
                fade_in_out(t, b5[0] + 0.2, b5[0] + 1.2, b5[1] + 0.1, b5[1] + 0.5),
                size=S.T_TITLE, color=S.WHITE)
    c.fade_text(0, -3.30, "that is what the screen is for",
                fade_in_out(t, b5[0] + 2.8, b5[0] + 3.8, b5[1] + 0.1, b5[1] + 0.5),
                size=S.T_SUB, color=S.GREY)


def scene_budget(c: Canvas, t: float, T: Timing) -> None:
    b6, b7 = T.of("budget")
    draw_grid(c, dim=seg(t, b6[0] + 0.6, b6[0] + 2.2))

    c.fade_text(0, 3.62, "The budget",
                fade_in_out(t, b6[0] + 0.3, b6[0] + 1.3, b7[0] - 0.9, b7[0] - 0.3),
                size=S.T_TITLE, color=S.C_ACCENT)
    pick = seg(t, b6[0] + 3.4, b6[0] + 6.0, smoother)
    if pick > 0:
        k = int(1000 * pick)
        c.dots(GRID_X[PICK_IDX[:k]], GRID_Y[PICK_IDX[:k]], size=11.0,
               color=S.C_ACCENT, alpha=0.95, zorder=7)
    c.fade_text(0, -3.30, "1,000 genes  ·  5% of the library",
                seg(t, b6[0] + 4.4, b6[0] + 5.6), size=S.T_SUB, color=S.C_ACCENT)
    c.fade_text(0, 3.62, "Which 1,000?", seg(t, b7[0] - 0.4, b7[0] + 0.4),
                size=S.T_TITLE, color=S.WHITE)


# ---------------------------------------------------------------------------
# 2. the loop, and the benchmark
# ---------------------------------------------------------------------------
def scene_loop(c: Canvas, t: float, T: Timing) -> None:
    b3, b4, b5 = T.of("loop")

    # --- ten batches of a hundred ---
    alive = 1.0 - seg(t, b4[0] - 0.5, b4[0] + 0.5)
    if alive > 0.01:
        c.fade_text(0, 3.5, "Spend it in rounds", seg(t, b3[0], b3[0] + 1.2),
                    size=S.T_TITLE, color=S.WHITE)
        for r in range(D.N_ROUNDS):
            p = stagger(r, D.N_ROUNDS, t, b3[0] + 0.9, b3[0] + 4.3, overlap=0.72)
            if p <= 0.01:
                continue
            cx = -6.3 + r * 1.4
            c.rect(cx, 0.55, 1.12, 1.9, color=S.GREY_D, lw=1.8,
                   alpha=alive * p * 0.9, radius=0.06)
            m = int(round(10 * p))
            gx, gy = np.meshgrid(np.arange(10), np.arange(10))
            gx = cx - 0.36 + 0.08 * gx.ravel()[: m * 10]
            gy = 0.19 + 0.08 * gy.ravel()[: m * 10]
            c.dots(gx, gy, size=7.5, color=S.C_FORMER, alpha=alive * 0.95)
            c.text(cx, -0.72, f"{r+1}", size=S.T_TINY, color=S.GREY,
                   alpha=alive * p)
        c.fade_text(0, -1.55, "10 rounds  ×  100 genes  =  1,000",
                    seg(t, b3[0] + 3.2, b3[0] + 4.4) * alive, size=S.T_SUB,
                    color=S.GREY)

    # --- propose / observe / adapt ---
    cyc = fade_in_out(t, b4[0] - 0.3, b4[0] + 1.0, b5[0] - 0.9, b5[0] - 0.1)
    if cyc > 0.01:
        R = 1.85
        nodes = [("propose 100 genes", S.C_FORMER), ("observe which are hits", S.C_HIT),
                 ("adapt the ranking", S.C_LOOP)]
        angs = [90, -30, 210]
        pos = [(R * np.cos(np.radians(a)), 0.35 + R * np.sin(np.radians(a))) for a in angs]
        for i, ((lab, col), (x, y)) in enumerate(zip(nodes, pos)):
            p = stagger(i, 3, t, b4[0] + 0.2, b4[0] + 2.4, overlap=0.5)
            c.circle(x, y, 0.30, color=col, lw=2.6, fill=S.BG, alpha=cyc * p)
            c.circle(x, y, 0.13, color=col, lw=0, fill=col, alpha=cyc * p)
            dy = 0.62 if y > 0.35 else -0.62
            c.text(x, y + dy, lab, size=S.T_CAPTION, color=col, alpha=cyc * p)
        for i in range(3):
            j = (i + 1) % 3
            p = stagger(i, 3, t, b4[0] + 1.1, b4[0] + 3.2, overlap=0.5)
            if p <= 0.01:
                continue
            (x0, y0), (x1, y1) = pos[i], pos[j]
            vx, vy = x1 - x0, y1 - y0
            L = np.hypot(vx, vy)
            ux, uy = vx / L, vy / L
            c.arrow(x0 + 0.44 * ux, y0 + 0.44 * uy,
                    x0 + (L - 0.46) * ux * p + 0.44 * ux * (1 - p),
                    y0 + (L - 0.46) * uy * p + 0.44 * uy * (1 - p),
                    color=S.GREY, lw=2.0, alpha=cyc * 0.8, curve=-0.22)
        c.fade_text(0, 3.5, "The loop", seg(t, b4[0], b4[0] + 1.0),
                    size=S.T_TITLE)
        c.fade_text(0, -3.05, "each round's answer changes the next question",
                    seg(t, b4[0] + 2.8, b4[0] + 4.0) * cyc, size=S.T_CAPTION,
                    color=S.GREY)

    # --- AssayBench-Loop: 1,389 screens, split chronologically ---
    bench = seg(t, b5[0] - 0.2, b5[0] + 1.4)
    if bench > 0.01:
        c.fade_text(0, 3.5, "AssayBench-Loop", bench, size=S.T_TITLE)
        c.fade_text(0, 2.82, f"{D.N_SCREENS:,} real CRISPR screens, split by time",
                    seg(t, b5[0] + 0.5, b5[0] + 1.9), size=S.T_SUB, color=S.GREY)

        splits = [(D.N_TRAIN, "1,349 train", "< 2021", S.C_FORMER),
                  (D.N_VAL, "20 val", "2021", S.C_ACCENT),
                  (D.N_TEST, "20 test", "> 2021", S.C_HIT)]
        total = sum(s[0] for s in splits)
        full_w, x = 11.5, -5.9
        for i, (n, lab, yr, col) in enumerate(splits):
            p = stagger(i, 3, t, b5[0] + 1.6, b5[0] + 4.2, overlap=0.45)
            w = max(full_w * n / total, 0.34)     # val/test would be invisible to scale
            if p > 0.01:
                c.rect(x + w / 2, 0.55, w * p, 1.30, color=col, lw=2.0,
                       fill=mix(S.BG, col, 0.16), alpha=p, radius=0.05)
                if w > 1.0:
                    c.text(x + w / 2, 0.55, lab, size=S.T_CAPTION, color=col, alpha=p)
                    c.text(x + w / 2, -0.44, yr, size=S.T_SMALL, color=S.GREY, alpha=p)
                else:
                    # The val/test slivers are too narrow to label in place, so
                    # their labels ladder upwards on short leaders instead.
                    top = 1.20 + 0.62 * i
                    c.line(x + w / 2, 1.22, x + w / 2, top, color=col,
                           lw=1.2, alpha=p * 0.7)
                    c.text(x + w / 2, top + 0.20, f"{lab}   {yr}", size=S.T_SMALL,
                           color=col, alpha=p)
            x += w + 0.10

        c.arrow(-6.2, -1.25, 6.2, -1.25, color=S.GREY_D, lw=1.8,
                alpha=seg(t, b5[0] + 3.4, b5[0] + 4.4))
        c.fade_text(0, -1.85, "time", seg(t, b5[0] + 3.6, b5[0] + 4.6),
                    size=S.T_SMALL, color=S.GREY)
        c.fade_text(0, -2.85, "train on the past  ·  test on the future",
                    seg(t, b5[0] + 6.2, b5[0] + 7.6), size=S.T_SUB,
                    color=S.WHITE)


# ---------------------------------------------------------------------------
# 3. how a strategy is measured
# ---------------------------------------------------------------------------
def scene_curve(c: Canvas, t: float, T: Timing) -> None:
    b8, b9 = T.of("curve")

    c.fade_text(0, 3.72, "The recovery curve", seg(t, b8[0], b8[0] + 1.1),
                size=S.T_TITLE)
    plot_frame(c, seg(t, b8[0] + 0.5, b8[0] + 1.8),
               rounds_alpha=seg(t, b8[0] + 5.6, b8[0] + 6.8), labels=False)

    # Name the axes in the order the narration does, sweeping each one as it is
    # named so the empty plot has something to watch.
    # The sweep is a transient highlight: it hands the axis back to grey before
    # the random line is drawn, so it never competes with the data.
    done = seg(t, b9[0] - 1.3, b9[0] - 0.4)
    hl_y = seg(t, b8[0] + 2.4, b8[0] + 3.4) * (1.0 - done)
    hl_x = seg(t, b8[0] + 4.2, b8[0] + 5.2) * (1.0 - done)

    py = seg(t, b8[0] + 2.4, b8[0] + 3.8, smoother) * (1.0 - done)
    if py > 0.01:
        c.path([PX0, PX0], [PY0, PY0 + PH], 1.0, color=S.C_LOOP, lw=3.4, alpha=py)
    c.fade_text(PX0 - 1.08, PY0 + PH / 2, "fraction of all\nhits found",
                seg(t, b8[0] + 2.4, b8[0] + 3.4), size=S.T_CAPTION,
                color=mix(S.GREY, S.C_LOOP, hl_y), rotation=90)
    px = seg(t, b8[0] + 4.2, b8[0] + 5.6, smoother) * (1.0 - done)
    if px > 0.01:
        c.path([PX0, PX0 + PW], [PY0, PY0], 1.0, color=S.C_ACCENT, lw=3.4, alpha=px)
    c.fade_text(PX0 + PW / 2, PY0 - 0.92, "fraction of the library sampled",
                seg(t, b8[0] + 4.2, b8[0] + 5.2), size=S.T_CAPTION,
                color=mix(S.GREY, S.C_ACCENT, hl_x))

    # Random: recall equals the fraction sampled, so the curve is the diagonal.
    rx, ry = curve_xy("random")
    prog = seg(t, b9[0] + 0.3, b9[0] + 3.0, smoother)
    c.path(rx, ry, prog, color=S.C_RANDOM, lw=2.6, alpha=1.0, ls=(0, (5, 4)))
    if prog > 0.15:
        c.fade_text(rx[-1] - 0.15, ry[-1] + 0.36, "random expectation",
                    seg(t, b9[0] + 2.4, b9[0] + 3.4), size=S.T_CAPTION,
                    color=S.C_RANDOM, ha="right")
    c.fade_text(0, 2.95, "sample 5% at random, find 5% of the hits",
                fade_in_out(t, b9[0] + 3.2, b9[0] + 4.4, b9[0] + 5.0, b9[0] + 5.6),
                size=S.T_SUB, color=S.WHITE)
    c.fade_text(0, 2.95, "everything that follows is measured against this line",
                seg(t, b9[0] + 5.6, b9[0] + 6.6), size=S.T_SUB, color=S.GREY)


# ---------------------------------------------------------------------------
# 4. the language model's prior
# ---------------------------------------------------------------------------
def scene_llm(c: Canvas, t: float, T: Timing) -> None:
    b6, b7, b8 = T.of("llm")

    # --- description -> LLM -> ranked genes ---
    card = fade_in_out(t, b6[0] - 0.2, b6[0] + 1.2, b7[0] - 1.1, b7[0] - 0.35)
    if card > 0.01:
        c.fade_text(0, 3.55, "Ask a language model",
                    seg(t, b6[0], b6[0] + 1.1), size=S.T_TITLE)

        c.rect(-4.55, 0.55, 4.5, 2.5, color=S.GREY_D, lw=1.8, alpha=card,
               fill=mix(S.BG, S.WHITE, 0.03), radius=0.06)
        c.text(-4.55, 1.44, "screen description", size=S.T_SMALL, color=S.GREY,
               alpha=card)
        desc = ["Which genes, when activated", "in neuroblastoma, confer",
                "resistance to RSL3?"]
        for i, ln in enumerate(desc):
            p = seg(t, b6[0] + 0.9 + 0.30 * i, b6[0] + 1.8 + 0.30 * i)
            c.typewriter(-4.55, 0.78 - 0.52 * i, ln, p, size=S.T_CAPTION,
                         color=S.WHITE, alpha=card)

        c.arrow(-2.15, 0.55, -1.25, 0.55, color=S.GREY, lw=2.2,
                alpha=card * seg(t, b6[0] + 2.4, b6[0] + 3.0))
        llm = seg(t, b6[0] + 2.7, b6[0] + 3.5)
        c.rect(-0.30, 0.55, 1.65, 1.25, color=S.C_LLM, lw=2.4, alpha=card * llm,
               fill=mix(S.BG, S.C_LLM, 0.12), radius=0.06)
        c.text(-0.30, 0.55, "LLM", size=S.T_SUB, color=S.C_LLM, alpha=card * llm)
        c.arrow(0.60, 0.55, 1.50, 0.55, color=S.GREY, lw=2.2,
                alpha=card * seg(t, b6[0] + 3.3, b6[0] + 3.9))

        for i, g in enumerate(["GPX4", "ERMAP", "GCH1", "SLC7A11", "ACSL4"]):
            p = stagger(i, 5, t, b6[0] + 3.6, b6[0] + 5.4, overlap=0.55)
            c.fade_text(2.95, 1.75 - 0.60 * i, g, p * card, size=S.T_BODY,
                        color=S.C_LLM, family="DejaVu Sans Mono")
        c.fade_text(2.95, -1.35, "a ranking, before any measurement",
                    seg(t, b6[0] + 5.2, b6[0] + 6.4) * card, size=S.T_SMALL,
                    color=S.GREY)
        c.fade_text(0, -3.25, "an opinion from the literature  —  before any measurement",
                    seg(t, b6[0] + 6.0, b6[0] + 7.0) * card, size=S.T_CAPTION,
                    color=S.GREY)

    # --- the curve ---
    pa = seg(t, b7[0] - 0.7, b7[0] + 0.5)
    plot_frame(c, pa)
    if pa > 0.01:
        gx, gy = curve_xy("gemini")
        prog = seg(t, b7[0] + 0.2, b7[0] + 3.4, smoother)
        c.path(gx, gy, prog, color=S.C_LLM, lw=4.0, alpha=pa)
        head = int(round(prog * D.N_ROUNDS))
        if head >= 1:
            c.dots(gx[:head + 1], gy[:head + 1], size=34, color=S.C_LLM,
                   alpha=pa, zorder=9)
        rx, ry = curve_xy("random")
        c.path(rx, ry, seg(t, b7[0] + 0.6, b7[0] + 2.6), color=S.C_RANDOM,
               lw=2.2, alpha=pa * 0.65, ls=(0, (5, 4)))
        c.fade_text(rx[-1] - 0.15, ry[-1] + 0.34, "random expectation",
                    seg(t, b7[0] + 2.2, b7[0] + 3.0) * pa, size=S.T_SMALL,
                    color=S.C_RANDOM, ha="right")
        c.fade_text(gx[-1] + 0.20, gy[-1], "Gemini 3.1 Pro",
                    seg(t, b7[0] + 3.0, b7[0] + 3.9) * pa, size=S.T_CAPTION,
                    color=S.C_LLM, ha="left")
        c.fade_text(gx[-1] + 0.20, gy[-1] - 0.42, "EF 4.71",
                    seg(t, b7[0] + 3.3, b7[0] + 4.2) * pa, size=S.T_SMALL,
                    color=S.C_LLM, ha="left")
        c.fade_text(0, 3.72, "A strong prior", seg(t, b7[0], b7[0] + 1.0),
                    size=S.T_TITLE)

    # --- but it flattens ---
    fl = seg(t, b8[0] + 0.8, b8[0] + 2.0)
    if fl > 0.01:
        gx, gy = curve_xy("gemini")
        # Labels sit inside their own triangle: below the axis they would land
        # on the tick row.
        for i, (a, b, col, lab, dy) in enumerate([(0, 3, S.C_LLM, "steep", 0.34),
                                                  (7, 10, S.GREY, "flat", -0.44)]):
            p = stagger(i, 2, t, b8[0] + 0.8, b8[0] + 3.0, overlap=0.4)
            if p <= 0.01:
                continue
            c.tri([(gx[a], gy[a]), (gx[b], gy[a]), (gx[b], gy[b])],
                  color=col, alpha=0.20 * p)
            c.line(gx[a], gy[a], gx[b], gy[b], color=col, lw=2.6, alpha=p)
            c.text((gx[a] + gx[b]) / 2, gy[a] + dy, lab, size=S.T_SMALL,
                   color=col, alpha=p)
        c.fade_text(0, 2.95, "LLMs:  strong prior, weak adaptation",
                    seg(t, b8[0] + 3.2, b8[0] + 4.4), size=S.T_SUB,
                    color=S.WHITE)


# ---------------------------------------------------------------------------
# 4. AssayFormer
# ---------------------------------------------------------------------------
SCREEN_X = -6.35          # the description token, leftmost in the sequence
GENE_X0 = -4.70           # first gene token


def scene_former(c: Canvas, t: float, T: Timing) -> None:
    b9, b10, b11 = T.of("former")

    arch = fade_in_out(t, b9[0] - 0.2, b9[0] + 1.3, b10[0] - 1.2, b10[0] - 0.4)
    if arch > 0.01:
        c.fade_text(0, 3.55, "AssayFormer", seg(t, b9[0], b9[0] + 1.1),
                    size=S.T_TITLE, color=S.C_FORMER)
        c.fade_text(0, 2.88, "a policy trained across 1,349 completed screens",
                    seg(t, b9[0] + 0.6, b9[0] + 1.9), size=S.T_SUB, color=S.GREY)

        # history tokens
        hist = [("MYC", 1), ("STAT1", 0), ("APOE", 1), ("E2F1", 0), ("EGFR", 1)]
        for i, (g, hit) in enumerate(hist):
            p = stagger(i, len(hist), t, b9[0] + 1.8, b9[0] + 4.0, overlap=0.55)
            if p <= 0.01:
                continue
            x = GENE_X0 + i * 1.50
            col = S.C_HIT if hit else S.GREY
            c.rect(x, 1.30, 1.30, 0.94, color=col, lw=2.0, alpha=arch * p,
                   fill=mix(S.BG, col, 0.13), radius=0.05)
            c.text(x, 1.50, g, size=S.T_SMALL, color=S.WHITE, alpha=arch * p,
                   family="DejaVu Sans Mono")
            c.text(x, 1.04, f"hit = {hit}", size=S.T_TINY, color=col, alpha=arch * p)
        ph = seg(t, b9[0] + 1.8, b9[0] + 2.6)
        c.text(GENE_X0 + 2 * 1.50, 2.18, "what you have tested so far",
               size=S.T_SMALL, color=S.GREY, alpha=arch * ph)

        # The screen description is another token in the same sequence, and it
        # sits leftmost so its line into the encoder never crosses the genes'.
        pd = seg(t, b9[0] + 3.9, b9[0] + 4.7)
        c.rect(SCREEN_X, 1.30, 1.45, 0.94, color=S.C_LLM, lw=2.0, alpha=arch * pd,
               fill=mix(S.BG, S.C_LLM, 0.13), radius=0.05)
        c.text(SCREEN_X, 1.46, "screen", size=S.T_TINY, color=S.C_LLM, alpha=arch * pd)
        c.text(SCREEN_X, 1.12, "description", size=S.T_TINY, color=S.C_LLM,
               alpha=arch * pd)
        c.text(SCREEN_X, 2.18, "this assay", size=S.T_SMALL, color=S.GREY,
               alpha=arch * pd)

        # Each token enters the encoder at its own evenly spaced position, the
        # way a real sequence does — not all funnelled into one point. Sources
        # and destinations are both left-to-right, so no two lines cross.
        pt = seg(t, b9[0] + 4.8, b9[0] + 5.8)
        enc_w, enc_top = 7.4, -1.42 + 0.40
        srcs = [SCREEN_X] + [GENE_X0 + i * 1.50 for i in range(len(hist))]
        for i, src_x in enumerate(srcs):
            dst_x = -enc_w / 2 + enc_w * (i + 0.5) / len(srcs)
            c.line(src_x, 1.30 - 0.47, dst_x, enc_top, color=S.GREY_D, lw=1.3,
                   alpha=arch * pt * 0.7)
            c.dots([dst_x], [enc_top], size=14, color=S.C_FORMER,
                   alpha=arch * pt * 0.8, zorder=8)
        c.rect(0.0, -1.42, enc_w, 0.80, color=S.C_FORMER, lw=2.4, alpha=arch * pt,
               fill=mix(S.BG, S.C_FORMER, 0.12), radius=0.06)
        c.text(0.0, -1.42, "transformer encoder", size=S.T_BODY,
               color=S.C_FORMER, alpha=arch * pt)

        ps = seg(t, b9[0] + 6.0, b9[0] + 7.2)
        c.arrow(0.0, -1.86, 0.0, -2.36, color=S.GREY, lw=2.0, alpha=arch * ps)
        c.text(0.0, -2.72, "a score for every untested gene", size=S.T_CAPTION,
               color=S.WHITE, alpha=arch * ps)
        bars = RNG.random(34) ** 2.2
        for i, h in enumerate(sorted(bars, reverse=True)):
            p = stagger(i, 34, t, b9[0] + 7.0, b9[0] + 8.6, overlap=0.88)
            x = -4.0 + i * 0.242
            c.line(x, -3.05, x, -3.05 - 0.62 * h, color=S.C_FORMER, lw=4.0,
                   alpha=arch * p * 0.9)
        c.fade_text(0, -4.00, "it reads the feedback, not just the description",
                    seg(t, b9[0] + 8.6, b9[0] + 9.8) * arch, size=S.T_CAPTION,
                    color=S.GREY)

    # --- the curve, alongside the LLM's ---
    pa = seg(t, b10[0] - 0.8, b10[0] + 0.4)
    plot_frame(c, pa)
    if pa > 0.01:
        gx, gy = curve_xy("gemini")
        c.path(gx, gy, 1.0, color=S.C_LLM, lw=3.4, alpha=pa * 0.55)
        c.fade_text(gx[-1] + 0.20, gy[-1] + 0.10, "Gemini 3.1 Pro", pa,
                    size=S.T_SMALL, color=S.C_LLM, ha="left")

        fx, fy = curve_xy("assayformer")
        prog = seg(t, b10[0] + 0.1, b10[0] + 3.2, smoother)
        c.path(fx, fy, prog, color=S.C_FORMER, lw=4.0, alpha=pa)
        head = int(round(prog * D.N_ROUNDS))
        if head >= 1:
            c.dots(fx[:head + 1], fy[:head + 1], size=34, color=S.C_FORMER,
                   alpha=pa, zorder=9)
        c.fade_text(fx[-1] + 0.20, fy[-1] - 0.34, "AssayFormer",
                    seg(t, b10[0] + 2.6, b10[0] + 3.5) * pa, size=S.T_CAPTION,
                    color=S.C_FORMER, ha="left")
        c.fade_text(0, 3.72, "A cold start that learns",
                    seg(t, b10[0] - 0.3, b10[0] + 0.8), size=S.T_TITLE)

        # call out the cold start, then the climb
        cs = fade_in_out(t, b10[0] + 0.7, b10[0] + 1.4, b11[0] + 0.2, b11[0] + 0.9)
        if cs > 0.01:
            # The gap between the two curves here is barely taller than the
            # text, so the callout sits well above both and reaches down with
            # a leader instead of squeezing into the gap.
            c.circle(fx[1], fy[1], 0.30, color=S.WHITE, lw=2.0, alpha=cs * 0.8)
            c.line(fx[1] + 0.22, fy[1] + 1.52, fx[1] + 0.12, fy[1] + 0.36,
                   color=S.WHITE, lw=1.2, alpha=cs * 0.45)
            c.text(fx[1] + 0.26, fy[1] + 1.68, "starts behind", size=S.T_SMALL,
                   color=S.WHITE, alpha=cs, ha="left")
        cl = seg(t, b11[0] + 0.4, b11[0] + 1.6)
        if cl > 0.01:
            c.tri([(fx[6], fy[6]), (fx[10], fy[6]), (fx[10], fy[10])],
                  color=S.C_FORMER, alpha=0.20 * cl)
            c.line(fx[6], fy[6], fx[10], fy[10], color=S.C_FORMER, lw=2.6, alpha=cl)
            c.fade_text(0, 2.95, "AssayFormer:  weak prior, strong adaptation",
                        seg(t, b11[0] + 0.9, b11[0] + 2.0), size=S.T_SUB,
                        color=S.WHITE)


# ---------------------------------------------------------------------------
# 5. the handoff
# ---------------------------------------------------------------------------
def scene_handoff(c: Canvas, t: float, T: Timing) -> None:
    b12, b13, b14 = T.of("handoff")

    # k is a round index, so surface the round numbering under the % axis here.
    plot_frame(c, 1.0, rounds_alpha=seg(t, b13[0] + 0.1, b13[0] + 1.1))
    gx, gy = curve_xy("gemini")
    fx, fy = curve_xy("assayformer")
    lx, ly = curve_xy("assayloop")

    # fade the two ingredients back once AssayLoop takes over
    ghost = 1.0 - 0.45 * seg(t, b13[0] + 1.6, b13[0] + 3.0)
    c.path(gx, gy, 1.0, color=S.C_LLM, lw=3.4, alpha=ghost)
    c.path(fx, fy, 1.0, color=S.C_FORMER, lw=3.4, alpha=ghost)

    # The descriptor labels and the opening title both have to clear out before
    # the AssayLoop labels land in the same space.
    gone = seg(t, b14[0] - 0.7, b14[0] + 0.1)
    p12 = seg(t, b12[0], b12[0] + 1.2) * (1.0 - gone)
    p12b = seg(t, b12[0] + 1.4, b12[0] + 2.6) * (1.0 - gone)
    c.fade_text(gx[-1] + 0.20, gy[-1] + 0.56, "strong prior", p12,
                size=S.T_SMALL, color=S.C_LLM, ha="left")
    c.fade_text(gx[-1] + 0.20, gy[-1] + 0.20, "weak adaptation", p12,
                size=S.T_SMALL, color=S.C_LLM, ha="left")
    c.fade_text(fx[-1] + 0.20, fy[-1] - 0.22, "weak prior", p12b,
                size=S.T_SMALL, color=S.C_FORMER, ha="left")
    c.fade_text(fx[-1] + 0.20, fy[-1] - 0.58, "strong adaptation", p12b,
                size=S.T_SMALL, color=S.C_FORMER, ha="left")
    c.fade_text(0, 3.72, "Use each where it wins",
                seg(t, b12[0], b12[0] + 1.1) * (1.0 - gone), size=S.T_TITLE)

    # the handoff marker
    hm = seg(t, b13[0] + 0.3, b13[0] + 1.3)
    if hm > 0.01:
        hx, _ = to_xy(D.HANDOFF_K, 0)
        c.line(hx, PY0, hx, PY0 + PH * 0.96, color=S.WHITE, lw=1.8,
               alpha=hm * 0.5, ls=(0, (4, 5)))
        c.text(hx, PY0 + PH * 0.99 + 0.22, f"handoff  ·  k = {D.HANDOFF_K}",
               size=S.T_CAPTION, color=S.WHITE, alpha=hm)

    # AssayLoop: literally the LLM for 3 rounds, then its own path
    pw = seg(t, b13[0] + 0.6, b13[0] + 2.2, smoother)     # warm start
    pr = seg(t, b13[0] + 2.0, b13[0] + 4.4, smoother)     # after the handoff
    if pw > 0.01:
        c.path(lx[:D.HANDOFF_K + 1], ly[:D.HANDOFF_K + 1], pw, color=S.C_LOOP,
               lw=5.0, alpha=1.0, zorder=11)
    if pr > 0.01:
        c.path(lx[D.HANDOFF_K:], ly[D.HANDOFF_K:], pr, color=S.C_LOOP,
               lw=5.0, alpha=1.0, zorder=11)
        head = D.HANDOFF_K + int(round(pr * (D.N_ROUNDS - D.HANDOFF_K)))
        c.dots(lx[:head + 1], ly[:head + 1], size=40, color=S.C_LOOP,
               alpha=1.0, zorder=12)

    if pw > 0.4:
        # Lifted clear of the curve, with a leader so it still points at the
        # segment where green is running on top of yellow.
        lab = fade_in_out(t, b13[0] + 1.0, b13[0] + 1.7,
                          b13[0] + 2.6, b13[0] + 3.3)
        if lab > 0.01:
            c.line(lx[1] + 0.30, ly[1] + 1.32, lx[1] + 0.06, ly[1] + 0.20,
                   color=S.C_LOOP, lw=1.2, alpha=lab * 0.55)
            c.text(lx[1] + 0.34, ly[1] + 1.46, "the LLM's own trace",
                   size=S.T_SMALL, color=S.C_LOOP, alpha=lab, ha="left")

    p14 = min(seg(t, b14[0] - 0.5, b14[0] + 0.5, ease_out_back), 1.0)
    if p14 > 0.01:
        # Three curves, three labels. The EF number is deliberately not
        # repeated here — it arrives a beat later at full size.
        c.fade_text(lx[-1] + 0.20, ly[-1] + 0.20, "AssayLoop", p14,
                    size=S.T_BODY, color=S.C_LOOP, ha="left")
        c.fade_text(gx[-1] + 0.20, gy[-1] + 0.02, "Gemini 3.1 Pro", p14,
                    size=S.T_SMALL, color=S.C_LLM, ha="left")
        c.fade_text(fx[-1] + 0.20, fy[-1] - 0.26, "AssayFormer", p14,
                    size=S.T_SMALL, color=S.C_FORMER, ha="left")
        c.fade_text(0, 3.72, "AssayLoop", p14, size=S.T_TITLE, color=S.C_LOOP)


# ---------------------------------------------------------------------------
# 6. the result
# ---------------------------------------------------------------------------
def scene_result(c: Canvas, t: float, T: Timing) -> None:
    b15, b16, b17 = T.of("result")

    stats = fade_in_out(t, b15[0] - 0.2, b15[0] + 1.0, b16[0] - 1.0, b16[0] - 0.3)
    if stats > 0.01:
        c.fade_text(0, 3.45, "On 20 screens from the future",
                    seg(t, b15[0], b15[0] + 1.1), size=S.T_SUB, color=S.GREY)
        cards = [("5.67×", "better than random", S.C_LOOP),
                 ("27.7%", "of every hit found", S.C_LOOP),
                 ("5%", "of the library assayed", S.C_ACCENT)]
        for i, (big, sub, col) in enumerate(cards):
            p = stagger(i, 3, t, b15[0] + 0.9, b15[0] + 3.2, overlap=0.42)
            if p <= 0.01:
                continue
            x = -4.35 + i * 4.35
            c.text(x, 1.62, big, size=86, color=col,
                   alpha=stats * min(p, 1.0),
                   shift=0.35 * (1 - ease_out_back(min(p, 1.0))))
            c.text(x, 0.72, sub, size=S.T_CAPTION, color=S.GREY, alpha=stats * p)

        # where that sits against everything else
        pb = seg(t, b15[0] + 3.4, b15[0] + 4.6)
        if pb > 0.01:
            bars = [("Random", "random"), ("Probability-of-hit", "poh"),
                    ("Screen-kNN", "screen_knn"), ("BPMF", "bpmf"),
                    ("Gemini 3.1 Pro", "gemini"), ("AssayFormer", "assayformer"),
                    ("AssayLoop", "assayloop")]
            x0, y0, bw, gap, maxw = -1.75, -3.30, 0.32, 0.42, 5.7
            for i, (lab, key) in enumerate(bars):
                p = stagger(i, len(bars), t, b15[0] + 3.5, b15[0] + 6.0, overlap=0.7)
                if p <= 0.01:
                    continue
                m = D.TABLE1[key]
                hero = key in ("assayloop", "assayformer", "gemini")
                col = (S.C_LOOP if key == "assayloop"
                       else S.C_FORMER if key == "assayformer"
                       else S.C_LLM if key == "gemini" else S.GREY_D)
                y = y0 + i * gap
                w = maxw * (m.ef / 6.0) * p
                c.rect(x0 + w / 2, y, w, bw, color="none", lw=0, fill=col,
                       alpha=stats * (1.0 if hero else 0.55))
                c.text(x0 - 0.16, y, lab, size=S.T_TINY,
                       color=S.WHITE if hero else S.GREY,
                       alpha=stats * p, ha="right")
                c.text(x0 + w + 0.16, y, f"{m.ef:.2f}", size=S.T_TINY,
                       color=col, alpha=stats * p, ha="left")
            c.fade_text(x0, y0 + len(bars) * gap + 0.02, "enrichment factor",
                        pb, size=S.T_SMALL, color=S.GREY, ha="left")

    # --- and it scales with history ---
    sc = fade_in_out(t, b16[0] - 0.4, b16[0] + 0.8, b17[0] - 0.6, b17[0] - 0.05)
    if sc > 0.01:
        c.fade_text(0, 3.45, "It improves from past experiments",
                    seg(t, b16[0], b16[0] + 1.1),
                    size=S.T_TITLE)
        ns = np.array([n for n, _ in D.SCALING], float)
        efs = np.array([e for _, e in D.SCALING], float)
        ax0, ay0, aw, ah = -4.6, -2.5, 9.2, 4.6
        lx = np.log10(ns)
        X = ax0 + aw * (lx - lx.min()) / (lx.max() - lx.min())
        Y = ay0 + ah * (efs - 1.0) / (5.0 - 1.0)
        c.axes2d(ax0, ay0, aw, ah, alpha=sc)
        for v in (1, 2, 3, 4, 5):
            y = ay0 + ah * (v - 1.0) / 4.0
            c.line(ax0, y, ax0 - 0.10, y, color=S.GREY_D, lw=1.5, alpha=sc)
            c.text(ax0 - 0.28, y, f"{v}", size=S.T_SMALL, color=S.GREY,
                   alpha=sc, ha="right")
        for n in (1, 10, 100, 1349):
            x = ax0 + aw * (np.log10(n) - lx.min()) / (lx.max() - lx.min())
            c.line(x, ay0, x, ay0 - 0.10, color=S.GREY_D, lw=1.5, alpha=sc)
            c.text(x, ay0 - 0.34, f"{n:,}", size=S.T_SMALL, color=S.GREY, alpha=sc)
        c.text(ax0 + aw / 2, ay0 - 0.92, "training screens", size=S.T_CAPTION,
               color=S.GREY, alpha=sc)
        c.text(ax0 - 0.95, ay0 + ah / 2, "EF", size=S.T_CAPTION, color=S.GREY,
               alpha=sc, rotation=90)

        prog = seg(t, b16[0] + 0.6, b16[0] + 3.6, smoother)
        c.path(X, Y, prog, color=S.C_FORMER, lw=4.0, alpha=sc)
        nvis = int(round(prog * (len(X) - 1)))
        c.dots(X[:nvis + 1], Y[:nvis + 1], size=36, color=S.C_FORMER,
               alpha=sc, zorder=9)
        c.fade_text(X[-1] - 0.10, Y[-1] + 0.46, "no sign of saturating",
                    seg(t, b16[0] + 3.4, b16[0] + 4.5) * sc, size=S.T_CAPTION,
                    color=S.C_FORMER, ha="right")

    # --- title card ---
    tc = seg(t, b17[0] - 0.5, b17[0] + 0.9, smoother)
    if tc > 0.01:
        c.fade_text(0, 1.55, "Biology in the loop", tc, size=78, color=S.WHITE)
        c.fade_text(0, 0.62, "Amortized Adaptive Hit Discovery in CRISPR Screens",
                    seg(t, b17[0] - 0.2, b17[0] + 1.2), size=S.T_SUB, color=S.GREY)
        c.fade_text(0, -0.60, "arXiv:2609.11877",
                    seg(t, b17[0] + 0.3, b17[0] + 1.5), size=S.T_BODY,
                    color=S.C_LOOP, family="DejaVu Sans Mono")
        c.fade_text(0, -1.28, "genentech.github.io/AssayLoop",
                    seg(t, b17[0] + 0.5, b17[0] + 1.7), size=S.T_CAPTION,
                    color=S.GREY, family="DejaVu Sans Mono")
        c.fade_text(0, -1.95, "Try it on our website today", 
                    seg(t, b17[0] + 1.4, b17[0] + 2.4), size=S.T_SUB,
                    color=S.C_LOOP)
        c.fade_text(0, -2.75, "Edwards · De Brouwer · Li · Lee · Hajiramezanali\n"
                              "Biton · Mostafavi · Scalia   —   Genentech",
                    seg(t, b17[0] + 0.8, b17[0] + 2.0), size=S.T_SMALL,
                    color=S.GREY_D)


SCENES = {
    "intro": scene_intro,
    "screen": scene_screen,
    "hits": scene_hits,
    "budget": scene_budget,
    "loop": scene_loop,
    "curve": scene_curve,
    "llm": scene_llm,
    "former": scene_former,
    "handoff": scene_handoff,
    "result": scene_result,
}


def draw_frame(c: Canvas, t: float, T: Timing) -> None:
    # Captions are not burned in: each scene already carries its own typography,
    # and a second text band on top of it would just fight for the same space.
    # The narration is shipped as a sidecar .srt instead, which a viewer can
    # toggle and a search engine can read.
    c.begin()
    SCENES[T.scene_of(t)](c, t, T)
    # A hairline source credit, so a viewer can tell measured from modelled.
    c.text(7.82, -4.34, "test split, n = 20 screens", size=11, color="#252B36",
           ha="right", va="bottom")
