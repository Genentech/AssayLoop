#!/usr/bin/env python
"""Rasterise the paper's three vector diagrams for the web.

The pipeline overview, the teaser, and the enrichment-factor explainer are
drawn as native vector PDFs (``paper_files/make_assayloop_paper_native.py``,
``make_assaybench_loop_figure1_v2.py``, ``make_assayloop_figures_v3.py``).
Shipping them to the browser as PDFs means an embedded PDF viewer with its own
scrollbar inside the page, which is a bad way to look at a figure. This turns
each one into a PNG the page can lay out like any other image.

The pipeline is additionally cut into its eight panels. At page width the
composed figure is a wall of 6pt type; one panel at a time is legible, and the
panels were drawn as separate boxes to begin with. The boxes are found by
looking for the large rectangles in the PDF's own drawing operators rather than
by hardcoded coordinates, so a re-exported figure with shifted panels either
still works or fails loudly.

Requires PyMuPDF, which the package does not depend on -- this is a
build-the-site step, not a runtime one::

    pip install pymupdf
    python docs/build_paper_panels.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("build_paper_panels")

DOCS = Path(__file__).resolve().parent
FIG_OUT = DOCS / "assets" / "figures"
DATA_OUT = DOCS / "assets" / "data"

#: Default render scale for the pipeline panel crops. The wider README teaser
#: uses 6x below so its small labels remain sharp on high-density displays.
ZOOM = 3.0

#: A panel box is at least this many points on a side. Everything smaller in
#: the pipeline figure is an inner card, a legend, or a matrix cell.
MIN_PANEL_W, MIN_PANEL_H = 150.0, 90.0

#: Panel rects are drawn twice, as a fill and a stroke, a fraction of a point
#: apart. Two boxes whose every edge agrees to within this are one panel.
#: Compared edge-by-edge rather than snapped to a grid -- a grid puts the two
#: copies of a box on opposite sides of a rounding boundary often enough to
#: matter (475.0 and 474.6 land in different 2pt cells).
DEDUP_TOL = 2.0

#: The composed figure carries flow arrows between panels that cross the panel
#: borders. Cropping exactly to the border leaves a few points of orphaned
#: arrow; trim the bottom edge past it.
BOTTOM_TRIM = 4.0

PANEL_LETTERS = "ABCDEFGH"

#: A heading for each panel, so the walkthrough reads as eight numbered steps
#: rather than as a grid of pictures the reader has to infer an order for. Kept
#: short: the caption underneath is where the sentence goes.
PANEL_TITLES = {
    "A": "From screens to a decision problem",
    "B": "What the splits contain",
    "C": "Initialising the gene embeddings",
    "D": "AssayFormer, trained supervised",
    "E": "Reinforcement learning on the context delta",
    "F": "Three ways to pick the next genes",
    "G": "The handoff",
    "H": "What it recovers",
}

#: Hand-written, one per panel of the pipeline figure. Prose lives with the
#: figure it describes, so the page is a caption edit away from being right.
PANEL_CAPTIONS = {
    "A": (
        "AssayBench-Loop turns 1,920 historical CRISPR screens into a lab-in-the-loop "
        "approach. A method sees the screen description and must rank the genes it wants "
        "assayed next; the data split is chronological, so the test split reflects the "
        "changing taste of scientists over time."),
    "B": (
        "The five broad phenotype families, across the train, validation and test "
        "splits. The validation and test split were selected to have broad phenotypic "
        "coverage and screen diversity, have ground truth for most genes (&gt; 18k), and "
        "have sufficient signal (&gt; 50 hits and &lt; 15% hits)."),
    "C": (
        "First, gene embeddings are initialised by applying Bayesian probabilistic "
        "matrix factorisation to the screen x gene hit matrix. Notably, we find this "
        "particular gene embedding initialization to be important for downstream "
        "performance"),
    "D": (
        "AssayFormer is a encoder-only transformer model which accepts the screen "
        "embedding and the already-assayed genes with their outcomes as input. Scoring "
        "is done using a bilinear scoring head, and produces a score for every untested "
        "gene. Gene embeddings come from BPMF initialization. Supervised finetuning is "
        "done on the training screens using a per-gene binary cross-entropy against the "
        "observed hit labels, given a randomly selected context of other genes from the "
        "screen."),
    "E": (
        "Next, RL fine-tuning with GRPO is applied to the model with eight full-screen "
        "rollouts. To sample different trajectories for each rollout, we use the Gumbel "
        "top-k sampling trick. We use a context delta approach as our reward: hits found "
        "by the policy that can see the history, minus hits found by a frozen copy that "
        "cannot. This encourages the model to learn how to leverage the context to "
        "outperform the frozen context-free model."),
    "F": (
        "This panel shows three paradigms for sequential experimental design on screens. "
        "Single-screen active learning adapts within one assay but starts cold. "
        "Amortized transfer learns how to adapt from previous screens (our approach). "
        "LLMs bring literature-scale priors but weak in-context learning."),
    "G": (
        "AssayLoop combines the benefits of an amortized model and LLM priors. An LLM "
        "ranks the first rounds from prior knowledge, then hands the results to "
        "AssayFormer once enough assay-specific evidence has accumulated for in-context "
        "adaptation to beat the prior."),
    "H": (
        "Held-out recovery curves. AssayLoop finds 27.7% of hits within an effective "
        "5% of the library, an enrichment factor of 5.66 over a random model."),
}


@dataclass
class Source:
    key: str
    pdf: str
    zoom: float
    #: Cut into panels using the figure's own panel boxes.
    split_panels: bool = False


SOURCES = [
    Source(key="figure1", pdf="figure1.pdf", zoom=6.0),
    Source(key="ef_metrics", pdf="ef_metrics.pdf", zoom=2.0),
    Source(key="pipeline", pdf="pipeline.pdf", zoom=2.4, split_panels=True),
]


def _panel_rects(page, pymupdf):
    """The figure's panel boxes, in reading order.

    Takes the large rectangles from the page's drawing operators, drops the
    page-sized one, collapses the fill/stroke duplicate of each box, then sorts
    into rows and left-to-right within a row.
    """
    page_r = page.rect
    rects: list = []
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.width < MIN_PANEL_W or r.height < MIN_PANEL_H:
            continue
        if r.width > page_r.width * 0.98 and r.height > page_r.height * 0.98:
            continue  # the page background
        for i, kept in enumerate(rects):
            if all(abs(a - b) <= DEDUP_TOL for a, b in
                   ((r.x0, kept.x0), (r.y0, kept.y0),
                    (r.x1, kept.x1), (r.y1, kept.y1))):
                # Same panel drawn again. Keep the union so the crop contains
                # the whole border rather than half of a 1pt stroke.
                rects[i] = kept | pymupdf.Rect(r)
                break
        else:
            rects.append(pymupdf.Rect(r))

    # Group into rows by top edge, then order within each row by left edge.
    rects.sort(key=lambda r: r.y0)
    rows: list[list] = []
    for r in rects:
        if rows and abs(r.y0 - rows[-1][0].y0) <= MIN_PANEL_H / 2:
            rows[-1].append(r)
        else:
            rows.append([r])
    out = []
    for row in rows:
        out.extend(sorted(row, key=lambda r: r.x0))
    return out


def _render(pdf_path: Path, out_path: Path, zoom: float, clip=None) -> dict:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    if clip is not None:
        page.set_cropbox(clip)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pix.save(out_path)
    doc.close()
    return {"width": pix.width, "height": pix.height}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", metavar="KEY",
                    help="render just these source keys")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        import pymupdf  # noqa: F401
    except ImportError:
        log.error("PyMuPDF is required to rasterise the paper's vector figures.\n"
                  "    pip install pymupdf")
        return 2

    sources = SOURCES
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {s.key for s in SOURCES}
        if unknown:
            log.error("unknown source key(s): %s", ", ".join(sorted(unknown)))
            return 2
        sources = [s for s in SOURCES if s.key in wanted]

    figures: dict[str, dict] = {}
    panels: list[dict] = []

    for src in sources:
        pdf = FIG_OUT / src.pdf
        if not pdf.is_file():
            # Never substitute: a missing source is a build error, not a
            # reason to ship the page with one fewer figure on it.
            log.error("missing source PDF: %s", pdf)
            return 1

        png = FIG_OUT / f"{src.key}.png"
        size = _render(pdf, png, src.zoom)
        figures[src.key] = {
            "image": f"assets/figures/{png.name}",
            "pdf": f"assets/figures/{pdf.name}",
            **size,
        }
        log.info("%-12s %s  %dx%d", src.key, png.name, size["width"], size["height"])

        if not src.split_panels:
            continue

        doc = pymupdf.open(pdf)
        rects = _panel_rects(doc[0], pymupdf)
        doc.close()
        if len(rects) != len(PANEL_LETTERS):
            log.error("expected %d panel boxes in %s, found %d -- the figure's "
                      "layout changed and the panel captions no longer line up",
                      len(PANEL_LETTERS), pdf.name, len(rects))
            return 1

        for letter, rect in zip(PANEL_LETTERS, rects):
            rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y1 - BOTTOM_TRIM)
            name = f"{src.key}_panel_{letter.lower()}.png"
            size = _render(pdf, FIG_OUT / name, ZOOM, clip=rect)
            panels.append({
                "key": f"{src.key}_panel_{letter.lower()}",
                "letter": letter,
                "title": PANEL_TITLES[letter],
                "caption": PANEL_CAPTIONS[letter],
                "image": f"assets/figures/{name}",
                **size,
            })
            log.info("  panel %s   %s  %dx%d", letter, name,
                     size["width"], size["height"])

    out = DATA_OUT / "panels.json"
    payload = {"figures": figures, "panels": panels}
    if args.only and out.is_file():
        # Same merge rule as build_figures.py --only: a partial rebuild must
        # not silently drop the panels it did not touch.
        prev = json.loads(out.read_text())
        merged_figs = {**prev.get("figures", {}), **figures}
        rebuilt = {p["key"] for p in panels}
        merged_panels = panels + [p for p in prev.get("panels", [])
                                  if p["key"] not in rebuilt]
        merged_panels.sort(key=lambda p: p["letter"])
        payload = {"figures": merged_figs, "panels": merged_panels}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
