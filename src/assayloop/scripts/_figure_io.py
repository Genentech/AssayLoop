"""Write one matplotlib figure to every format the paper and the site need.

Every ``plot_*``/``paper_*`` script in this package used to carry its own copy
of the same two lines::

    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(str(OUT).replace(".png", ".pdf"), facecolor=SURFACE, bbox_inches="tight")

which is fine until you want a third format. :func:`save_figure` is that pair
plus SVG, so ``docs/build_figures.py`` gets a vector asset for the web gallery
without anybody converting a PDF after the fact.

The PNG and the vector formats take separate DPI settings on purpose. DPI is
meaningless for the vector parts of a PDF or SVG, but it still controls the
resolution of anything rasterised inside them -- and several of these figures
are ``imshow`` heatmaps, which is exactly that case.
"""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["save_figure", "FORMATS"]

log = logging.getLogger(__name__)

#: Written for every figure, in this order.
FORMATS = ("png", "pdf", "svg")


def save_figure(fig, png_path, *, dpi: int | None = None,
                vector_dpi: int | None = None, **kw) -> dict[str, Path]:
    """Save ``fig`` as ``<stem>.png``, ``<stem>.pdf`` and ``<stem>.svg``.

    Args:
        fig: the matplotlib figure.
        png_path: the PNG path. The other two are the same path with the
            suffix swapped, which is the convention every caller already used.
        dpi: resolution for the PNG. ``None`` leaves matplotlib's default
            alone, which is what the call sites that passed no ``dpi`` had.
        vector_dpi: resolution for rasterised elements inside the PDF and SVG.
            Defaults to ``dpi``. Pass 300 for figures built out of ``imshow``.
        **kw: forwarded to every ``savefig`` call (``bbox_inches``,
            ``facecolor``, ...).

    Returns:
        Mapping of format name to the path written.
    """
    png_path = Path(png_path)
    if png_path.suffix != ".png":
        raise ValueError(
            f"save_figure expects a .png path so it can derive the other "
            f"formats; got {png_path.name!r}.")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    written = {}
    for fmt in FORMATS:
        path = png_path.with_suffix("." + fmt)
        res = dpi if fmt == "png" else (vector_dpi if vector_dpi is not None else dpi)
        fig.savefig(path, **({} if res is None else {"dpi": res}), **kw)
        written[fmt] = path
    log.info("wrote %s.{%s}", png_path.with_suffix(""), ",".join(FORMATS))
    return written
