"""A very small animation toolkit.

Real Manim needs pangocairo, which we cannot install here, so this stands in
for the handful of its primitives the storyboard actually uses: draw-on paths,
fade-and-shift entrances, typewriter text, and partial-opacity groups.

Everything is drawn into one persistent Agg figure holding a single full-bleed
axes with a 16x9 unit coordinate system, mirroring manim's frame convention:
the origin is centre-screen, x runs -8..8 and y runs -4.5..4.5.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from . import style as S

XMAX, YMAX = 8.0, 4.5


class Canvas:
    """One reusable figure. `begin()` clears it; `rgb()` hands back pixels."""

    def __init__(self, width: int = S.W, height: int = S.H, dpi: int = S.DPI):
        self.fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=S.BG)
        self.cv = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self._reset_axes()

    def _reset_axes(self) -> None:
        ax = self.ax
        ax.set_xlim(-XMAX, XMAX)
        ax.set_ylim(-YMAX, YMAX)
        ax.set_facecolor(S.BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def begin(self) -> "Canvas":
        self.ax.clear()
        self._reset_axes()
        return self

    def rgb(self) -> bytes:
        self.cv.draw()
        return self.cv.buffer_rgba().tobytes()

    # --- text -------------------------------------------------------------
    def text(self, x, y, s, size=S.T_BODY, color=S.WHITE, alpha=1.0,
             ha="center", va="center", weight="normal", style="normal",
             family=None, shift=0.0, zorder=10, **kw):
        """Place text. `shift` lifts it from below as it fades in, which is the
        entrance 3b1b uses for almost everything."""
        if alpha <= 0.002:
            return None
        return self.ax.text(
            x, y - shift, s, fontsize=size, color=color, alpha=min(alpha, 1.0),
            ha=ha, va=va, fontweight=weight, fontstyle=style,
            family=family or "serif", zorder=zorder, **kw)

    def fade_text(self, x, y, s, prog, size=S.T_BODY, color=S.WHITE,
                  rise=0.28, **kw):
        """Fade + rise entrance driven by an eased 0..1 progress."""
        return self.text(x, y, s, size=size, color=color,
                         alpha=prog, shift=rise * (1.0 - prog), **kw)

    def typewriter(self, x, y, s, prog, size=S.T_BODY, color=S.WHITE, **kw):
        """Reveal a string character by character (manim's `Write`)."""
        n = int(round(len(s) * max(0.0, min(1.0, prog))))
        if n == 0:
            return None
        return self.text(x, y, s[:n], size=size, color=color, **kw)

    # --- marks ------------------------------------------------------------
    def dots(self, xs, ys, size=6, color=S.GREY, alpha=1.0, zorder=5, **kw):
        if alpha <= 0.002 or len(xs) == 0:
            return None
        return self.ax.scatter(xs, ys, s=size, c=color, alpha=alpha,
                               linewidths=0, zorder=zorder, **kw)

    def line(self, x0, y0, x1, y1, color=S.GREY_D, lw=2.0, alpha=1.0,
             ls="-", zorder=5, **kw):
        if alpha <= 0.002:
            return None
        return self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
                            alpha=alpha, ls=ls, zorder=zorder,
                            solid_capstyle="round", **kw)

    def path(self, xs, ys, prog=1.0, color=S.BLUE, lw=3.0, alpha=1.0,
             zorder=8, ls="-", **kw):
        """Draw a polyline, revealed left-to-right by arclength (ShowCreation).

        Interpolates inside the final segment so the head moves smoothly
        instead of snapping from vertex to vertex.
        """
        prog = max(0.0, min(1.0, prog))
        if alpha <= 0.002 or prog <= 0.0 or len(xs) < 2:
            return None
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        seglen = np.hypot(np.diff(xs), np.diff(ys))
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        target = cum[-1] * prog
        i = int(np.searchsorted(cum, target))
        if i >= len(xs):
            px, py = xs, ys
        else:
            f = (target - cum[i - 1]) / max(seglen[i - 1], 1e-12)
            px = np.concatenate([xs[:i], [xs[i - 1] + f * (xs[i] - xs[i - 1])]])
            py = np.concatenate([ys[:i], [ys[i - 1] + f * (ys[i] - ys[i - 1])]])
        return self.ax.plot(px, py, color=color, lw=lw, alpha=alpha, ls=ls,
                            zorder=zorder, solid_capstyle="round",
                            solid_joinstyle="round", **kw)

    def circle(self, x, y, r, color=S.BLUE, lw=2.5, fill=None, alpha=1.0, zorder=6):
        if alpha <= 0.002:
            return None
        c = Circle((x, y), r, edgecolor=color, lw=lw,
                   facecolor=fill or "none", alpha=alpha, zorder=zorder)
        self.ax.add_patch(c)
        return c

    def rect(self, x, y, w, h, color=S.GREY_D, lw=2.0, fill=None, alpha=1.0,
             zorder=6, radius=0.0):
        """Centre-anchored rectangle; `radius` rounds the corners."""
        if alpha <= 0.002:
            return None
        kw = dict(edgecolor=color, lw=lw, facecolor=fill or "none",
                  alpha=alpha, zorder=zorder)
        if radius > 0:
            from matplotlib.patches import FancyBboxPatch
            p = FancyBboxPatch((x - w / 2 + radius, y - h / 2 + radius),
                               w - 2 * radius, h - 2 * radius,
                               boxstyle=f"round,pad={radius}", **kw)
        else:
            p = Rectangle((x - w / 2, y - h / 2), w, h, **kw)
        self.ax.add_patch(p)
        return p

    def arrow(self, x0, y0, x1, y1, color=S.GREY, lw=2.2, alpha=1.0,
              head=10.0, curve=0.0, zorder=7):
        if alpha <= 0.002:
            return None
        a = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=f"-|>,head_width={head/28:.3f},head_length={head/18:.3f}",
            mutation_scale=head, color=color, lw=lw, alpha=alpha, zorder=zorder,
            connectionstyle=f"arc3,rad={curve}", shrinkA=0, shrinkB=0)
        self.ax.add_patch(a)
        return a

    def tri(self, pts, color=S.BLUE, alpha=1.0, zorder=6):
        if alpha <= 0.002:
            return None
        p = Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                    alpha=alpha, zorder=zorder)
        self.ax.add_patch(p)
        return p

    def band(self, xs, lo, hi, color=S.BLUE, alpha=0.15, zorder=4):
        """Uncertainty ribbon."""
        if alpha <= 0.002:
            return None
        return self.ax.fill_between(xs, lo, hi, color=color, alpha=alpha,
                                    linewidth=0, zorder=zorder)

    # --- composites -------------------------------------------------------
    def axes2d(self, x0, y0, w, h, alpha=1.0, color=None, lw=2.0):
        """Bare L-shaped axes anchored at the bottom-left corner (x0, y0)."""
        color = color or S.GREY_D
        self.line(x0, y0, x0 + w, y0, color=color, lw=lw, alpha=alpha, zorder=4)
        self.line(x0, y0, x0, y0 + h, color=color, lw=lw, alpha=alpha, zorder=4)

    def caption(self, s, prog, y=-3.55, size=S.T_CAPTION, color=S.GREY):
        """The persistent lower-third caption line."""
        return self.fade_text(0.0, y, s, prog, size=size, color=color)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1: str, c2: str, t: float) -> str:
    """Blend two hex colours."""
    import matplotlib.colors as mc
    a = np.array(mc.to_rgb(c1))
    b = np.array(mc.to_rgb(c2))
    return mc.to_hex(a + (b - a) * max(0.0, min(1.0, t)))
