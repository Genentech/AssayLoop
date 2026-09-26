"""Rate functions and small timing helpers.

`smooth` is manim's default easing (the smoothstep polynomial); the rest are
the handful of variants the storyboard actually reaches for.
"""

from __future__ import annotations

import math


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def linear(t: float) -> float:
    return clamp(t)


def smooth(t: float) -> float:
    """3t^2 - 2t^3 — ease in and out, zero velocity at both ends."""
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smoother(t: float) -> float:
    """6t^5 - 15t^4 + 10t^3 — zero acceleration at the ends too."""
    t = clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def rush_into(t: float) -> float:
    """Slow start, fastest at the end."""
    return 2.0 * smooth(0.5 * clamp(t))


def rush_from(t: float) -> float:
    """Fastest at the start, easing out."""
    return 2.0 * smooth(0.5 * clamp(t) + 0.5) - 1.0


def ease_out_back(t: float, overshoot: float = 1.5) -> float:
    """Overshoots past 1 then settles — good for things that 'land'."""
    t = clamp(t) - 1.0
    return t * t * ((overshoot + 1.0) * t + overshoot) + 1.0


def there_and_back(t: float) -> float:
    t = clamp(t)
    return smooth(2.0 * t) if t < 0.5 else smooth(2.0 * (1.0 - t))


def wiggle(t: float, n: int = 2) -> float:
    return math.sin(n * math.tau * clamp(t)) * there_and_back(t)


def seg(t: float, start: float, end: float, fn=smooth) -> float:
    """Map absolute time `t` onto an eased 0..1 across the window [start, end].

    The workhorse of the scene code: every animation is expressed as
    "between these two seconds, take this quantity from 0 to 1".
    """
    if end <= start:
        return 1.0 if t >= end else 0.0
    return fn(clamp((t - start) / (end - start)))


def stagger(i: int, n: int, t: float, start: float, end: float, overlap: float = 0.6):
    """Eased 0..1 for item `i` of `n`, where items start in sequence.

    `overlap` = 0 means strictly one after another; 1 means all at once.
    """
    if n <= 1:
        return seg(t, start, end)
    span = end - start
    each = span / (1.0 + (n - 1) * (1.0 - overlap))
    step = each * (1.0 - overlap)
    s = start + i * step
    return seg(t, s, s + each)


def fade_in_out(t: float, start: float, hold_from: float, hold_to: float, end: float) -> float:
    """Opacity envelope: fade up, hold at 1, fade down."""
    if t < start or t > end:
        return 0.0
    if t < hold_from:
        return smooth((t - start) / max(hold_from - start, 1e-9))
    if t <= hold_to:
        return 1.0
    return 1.0 - smooth((t - hold_to) / max(end - hold_to, 1e-9))
