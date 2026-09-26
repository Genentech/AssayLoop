"""Build the teaser/poster image that links to the explainer.

    uv run python animation/make_poster.py

Composed rather than screen-grabbed: it uses the same canvas, palette and the
same measured curves as the animation, so the still reads as a title card
instead of an arbitrary frame. Writes a 1920x1080 PNG plus a 960-wide copy for
the web page to load.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation.assayloop_anim import data as D          # noqa: E402
from animation.assayloop_anim import narration as N     # noqa: E402
from animation.assayloop_anim import style as S         # noqa: E402
from animation.assayloop_anim.canvas import Canvas, mix  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Plot geometry for the poster: pushed right, to leave a column for the title.
PX0, PY0, PW, PH, YTOP = -1.0, -3.05, 6.5, 5.6, 0.30


def to_xy(r, v):
    return (PX0 + PW * np.asarray(r, float) / D.N_ROUNDS,
            PY0 + PH * np.asarray(v, float) / YTOP)


def runtime() -> str:
    """Read the duration off the schedule so the poster cannot go stale."""
    sch = N.load(os.path.join(OUT, "schedule.json"))
    total = sch[-1]["end"] + sch[-1]["pad"]
    m, sec = divmod(int(round(total)), 60)
    return f"{m} min {sec:02d} s"


def build() -> Canvas:
    c = Canvas().begin()
    curves = D.recall_curves()

    # --- left column: the title block ---
    c.text(-7.3, 2.55, "Biology", size=92, color=S.WHITE, ha="left")
    c.text(-7.3, 1.52, "in the loop", size=92, color=S.C_LOOP, ha="left")
    c.line(-7.25, 0.86, -3.4, 0.86, color=S.GREY_D, lw=2.0)
    c.text(-7.3, 0.28, "Amortized adaptive hit discovery", size=S.T_CAPTION,
           color=S.GREY, ha="left")
    c.text(-7.3, -0.18, "in CRISPR screens", size=S.T_CAPTION,
           color=S.GREY, ha="left")

    c.text(-7.3, -0.95, "5.67× random  ·  27.7% of hits at 5%", size=S.T_BODY,
           color=S.C_LOOP, ha="left")

    # play affordance — this still is a link to a video
    c.circle(-6.72, -2.05, 0.52, color=S.C_LOOP, lw=2.6, fill=mix(S.BG, S.C_LOOP, 0.14))
    c.tri([(-6.90, -1.78), (-6.90, -2.32), (-6.44, -2.05)], color=S.C_LOOP)
    c.text(-5.95, -1.92, "Watch the explainer", size=S.T_BODY,
           color=S.WHITE, ha="left")
    c.text(-5.95, -2.36, f"{runtime()}  ·  with narration", size=S.T_SMALL,
           color=S.GREY, ha="left")

    c.text(-7.3, -3.35, "arXiv:2609.11877", size=S.T_SMALL, color=S.GREY_D,
           ha="left", family="DejaVu Sans Mono")

    # --- right: the result, as the animation draws it ---
    c.axes2d(PX0, PY0, PW, PH, alpha=1.0)
    for v in (0.0, 0.10, 0.20, 0.30):
        _, y = to_xy(0, v)
        c.line(PX0, y, PX0 - 0.09, y, color=S.GREY_D, lw=1.4)
        c.text(PX0 - 0.26, y, f"{v*100:.0f}%", size=S.T_TINY, color=S.GREY, ha="right")
    for r in range(0, D.N_ROUNDS + 1, 2):
        x, _ = to_xy(r, 0)
        c.line(x, PY0, x, PY0 - 0.09, color=S.GREY_D, lw=1.4)
        c.text(x, PY0 - 0.30, f"{r*0.5:g}%", size=S.T_TINY, color=S.GREY)
    c.text(PX0 + PW / 2, PY0 - 0.78, "fraction of the library sampled",
           size=S.T_SMALL, color=S.GREY)
    c.text(PX0 - 0.95, PY0 + PH / 2, "fraction of all hits found",
           size=S.T_SMALL, color=S.GREY, rotation=90)

    # the handoff marker
    hx, _ = to_xy(D.HANDOFF_K, 0)
    c.line(hx, PY0, hx, PY0 + PH * 0.92, color=S.WHITE, lw=1.5, alpha=0.35,
           ls=(0, (4, 5)))
    c.text(hx, PY0 + PH * 0.92 + 0.22, f"handoff · k = {D.HANDOFF_K}",
           size=S.T_TINY, color=S.GREY)

    series = [("random", S.C_RANDOM, 2.0, "random", (0, (5, 4))),
              ("assayformer", S.C_FORMER, 3.2, "AssayFormer", "-"),
              ("gemini", S.C_LLM, 3.2, "Gemini 3.1 Pro", "-"),
              ("assayloop", S.C_LOOP, 5.0, "AssayLoop", "-")]
    for key, col, lw, label, ls in series:
        x, y = to_xy(np.arange(D.N_ROUNDS + 1), curves[key])
        c.path(x, y, 1.0, color=col, lw=lw, ls=ls,
               alpha=0.7 if key == "random" else 1.0)
        dy = {"gemini": 0.02, "assayformer": -0.30, "assayloop": 0.22,
              "random": 0.26}[key]
        c.text(x[-1] + 0.16, y[-1] + dy, label, size=S.T_TINY, color=col, ha="left")
    x, y = to_xy(np.arange(D.N_ROUNDS + 1), curves["assayloop"])
    c.dots(x, y, size=30, color=S.C_LOOP, zorder=12)

    return c


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    c = build()
    full = os.path.join(OUT, "poster.png")
    c.fig.savefig(full, dpi=S.DPI, facecolor=S.BG)
    print(f"  -> {full}")

    # half-scale copy for the web page
    small = os.path.join(OUT, "poster_960.png")
    c.fig.savefig(small, dpi=S.DPI // 2, facecolor=S.BG)
    print(f"  -> {small}")


if __name__ == "__main__":
    main()
