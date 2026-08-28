#!/usr/bin/env python3
"""Render the released six-panel Reactome sunburst figure.

This renderer intentionally uses only the JSON artifacts shipped with the
website.  It does not need the original sweep directories or the Reactome GMT:

* ``assets/data/sunburst.json`` supplies the two ring geometries.
* ``assets/data/results.json`` supplies the reported non-random EP-B, EP-S,
  and EP-D values.

The script validates the overlapping values before it draws anything, so stale
sunburst data cannot quietly produce a plausible but numerically wrong figure.
The Random reference is the f2 acquisition universe (genes present in at least
two of the 20 test screens), whose published EP-B / EP-S / EP-D values are
21.8 / 56.4 / 82.1.

On Converge, load Matplotlib and TeX Live so the exact CMU Serif OpenType font
is available::

    module load matplotlib/3.9.2-gfbf-2024a-Python-3.12.3
    module load texlive/20240312-GCC-13.3.0-Python-3.12.3-Boost-1.91.0
    MPLCONFIGDIR=/tmp/assayloop-mpl python docs/plot_pathway_sunburst_release.py

The default outputs are the one-row website figure
``assets/figures/pathway_sunburst.{pdf,png,svg}`` and the two-row paper figure
``assets/figures/pathway_sunburst_appendix.{pdf,png,svg}``.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


DOCS = Path(__file__).resolve().parent
DEFAULT_DATA = DOCS / "assets" / "data"
DEFAULT_OUTPUT = DOCS / "assets" / "figures" / "pathway_sunburst"

ORDER = [
    "Random",
    "kNN baseline",
    "BPMF",
    "Gemini-3.1-Pro",
    "AssayFormer",
    "AssayLoop (Gemini handoff)",
]

# Join the figure's short labels to the exact rows in results.json.  Keeping
# this explicit is preferable to fuzzy matching names such as "AssayLoop" and
# "AssayFormer", which have changed during the release cleanup.
RESULT_KEYS = {
    "kNN baseline": "kNN baseline",
    "BPMF": "BPMF",
    "Gemini-3.1-Pro": "Gemini-3.1-pro",
    "AssayFormer": "Transformer + GRPO (= AssayLoop)",
    "AssayLoop (Gemini handoff)": "Gemini-3.1-Pro - AssayLoop Handoff",
}

RANDOM_REFERENCE = {"ep_batch": 21.8, "ep_screen": 56.4, "eff_pathways": 82.1}
SURFACE = "#ffffff"
INK = "#0b0b0b"
MUTED = "#77746d"
LABEL_MIN = 0.06

PANEL_STYLES = {
    "website": {
        "inner_radius": 0.75,
        "inner_linewidth": 1.25,
        "label_radius": 0.60,
        "label_size": 7.0,
        "number_y": 0.10,
        "number_size": 18,
        "metric_y": -0.15,
        "metric_size": 7.2,
        "title_y": 1.09,
        "title_size": 10.5,
        "footer_y": -1.15,
        "footer_size": 7.8,
        "xlim": (-1.14, 1.14),
        "ylim": (-1.23, 1.19),
    },
    # Match the compact 2 x 3 layout used by the research renderer for the
    # appendix, while drawing from the same validated release artifacts.
    "appendix": {
        "inner_radius": 0.72,
        "inner_linewidth": 1.4,
        "label_radius": 0.57,
        "label_size": 5.8,
        "number_y": 0.10,
        "number_size": 15,
        "metric_y": -0.14,
        "metric_size": 6.2,
        "title_y": 1.06,
        "title_size": 8.6,
        "footer_y": -1.14,
        "footer_size": 6.6,
        "xlim": (-1.30, 1.30),
        "ylim": (-1.30, 1.30),
    },
}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"missing release artifact: {path}") from exc


def _cmu_serif(font_path: Path | None) -> str:
    """Register CMU Serif and return its Matplotlib family name."""
    if font_path is None:
        kpsewhich = shutil.which("kpsewhich")
        if kpsewhich:
            found = subprocess.run(
                [kpsewhich, "cmunrm.otf"], check=True, capture_output=True,
                text=True,
            ).stdout.strip()
            if found:
                font_path = Path(found)
    if font_path is None:
        try:
            font_path = Path(font_manager.findfont(
                "CMU Serif", fallback_to_default=False,
            ))
        except ValueError as exc:
            raise SystemExit(
                "CMU Serif was not found. On Converge, load the texlive module "
                "shown in this script's docstring, or pass --font-path."
            ) from exc
    if not font_path.is_file():
        raise SystemExit(f"CMU Serif font does not exist: {font_path}")

    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    if "CMU Serif" not in family:
        raise SystemExit(f"expected CMU Serif, but {font_path} identifies as {family!r}")
    matplotlib.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return family


def _validated_metrics(sunburst: dict, results: dict) -> dict[str, dict[str, float]]:
    if sunburst.get("order") != ORDER:
        raise SystemExit(
            "sunburst method order changed; update ORDER deliberately before rendering"
        )
    rows = {row["key"]: row for row in results["rows"] if row.get("available")}
    metrics: dict[str, dict[str, float]] = {}

    random = sunburst["methods"]["Random"]
    for field, expected in RANDOM_REFERENCE.items():
        actual = random.get(field)
        if actual is None or not math.isclose(actual, expected, abs_tol=0.05):
            raise SystemExit(
                f"Random {field} is {actual!r}, expected the f2 reference {expected}"
            )
    metrics["Random"] = {
        "ep_b": RANDOM_REFERENCE["ep_batch"],
        "ep_s": RANDOM_REFERENCE["ep_screen"],
        "ep_d": RANDOM_REFERENCE["eff_pathways"],
    }

    for panel, key in RESULT_KEYS.items():
        if key not in rows:
            raise SystemExit(f"results.json has no available row {key!r}")
        row = rows[key]
        ring = sunburst["methods"][panel]
        for ring_field, row_field, label in (
            ("ep_batch", "ep_b", "EP-B"),
            ("ep_screen", "ep_s", "EP-S"),
            ("eff_pathways", "ep_d", "EP-D"),
        ):
            if not math.isclose(ring[ring_field], row[row_field], abs_tol=1e-9):
                raise SystemExit(
                    f"{panel}: sunburst {label} {ring[ring_field]} != "
                    f"results {label} {row[row_field]}"
                )
        metrics[panel] = {
            "ep_b": row["ep_b"], "ep_s": row["ep_s"], "ep_d": row["ep_d"],
        }
    return metrics


def _panel_rings(data: dict, name: str) -> tuple[list, list]:
    """Return ``(inner, outer)`` wedges as ``(value, colour, label)`` tuples."""
    method = data["methods"][name]
    rows = list(zip(
        method["ids"], method["labels"], method["parents"],
        method["values"], method["colors"],
    ))
    inner = [(value, colour, label) for _id, label, parent, value, colour in rows
             if parent == data["root"]]
    outer = [(value, colour, label) for _id, label, parent, value, colour in rows
             if parent in data["categories"]]
    if not inner or not outer:
        raise SystemExit(f"{name}: release JSON has an empty sunburst ring")
    return inner, outer


def _draw_panel(ax, data: dict, metrics: dict, name: str, style: dict) -> None:
    inner, outer = _panel_rings(data, name)
    inner_values = [w[0] for w in inner]
    inner_colours = [w[1] for w in inner]
    outer_values = [w[0] for w in outer]
    outer_colours = [w[1] for w in outer]

    wedges, _ = ax.pie(
        inner_values, radius=style["inner_radius"], startangle=90,
        counterclock=False,
        colors=inner_colours,
        wedgeprops={
            "width": 0.30,
            "edgecolor": SURFACE,
            "linewidth": style["inner_linewidth"],
        },
    )
    ax.pie(
        outer_values, radius=1.0, startangle=90, counterclock=False,
        colors=outer_colours,
        wedgeprops={"width": 0.25, "edgecolor": SURFACE, "linewidth": 0.8},
    )

    total = sum(inner_values)
    for wedge, value, colour in zip(wedges, inner_values, inner_colours):
        share = value / total
        if share < LABEL_MIN:
            continue
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        red, green, blue = matplotlib.colors.to_rgb(colour)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        ax.text(
            style["label_radius"] * math.cos(angle),
            style["label_radius"] * math.sin(angle),
            f"{share:.0%}", ha="center", va="center",
            fontsize=style["label_size"],
            color=INK if luminance > 0.56 else SURFACE, fontweight="bold",
        )

    m = metrics[name]
    ax.text(0, style["number_y"], f"{m['ep_d']:.1f}", ha="center", va="center",
            fontsize=style["number_size"], color=INK, fontweight="semibold")
    ax.text(0, style["metric_y"], "EP-D", ha="center", va="center",
            fontsize=style["metric_size"], color=MUTED)
    ax.text(0, style["title_y"], name, ha="center", va="bottom",
            fontsize=style["title_size"], color=INK, linespacing=1.15)
    footer = f"EP-B {m['ep_b']:.1f}    EP-S {m['ep_s']:.1f}"
    ax.text(0, style["footer_y"], footer, ha="center", va="center",
            fontsize=style["footer_size"], color=MUTED)
    ax.set(xlim=style["xlim"], ylim=style["ylim"], aspect="equal")
    ax.axis("off")


def _draw_figure(sunburst: dict, metrics: dict, layout: str):
    style = PANEL_STYLES[layout]
    if layout == "website":
        shape = (1, 6)
        figsize = (15.8, 3.35)
    else:
        shape = (2, 3)
        figsize = (7.2, 5.15)

    fig, axes = plt.subplots(*shape, figsize=figsize, dpi=300,
                             subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.ravel(), ORDER):
        _draw_panel(ax, sunburst, metrics, name, style)

    handles = [
        Patch(facecolor=sunburst["colors"][category], edgecolor=SURFACE,
              label=category)
        for category in sunburst["categories"]
    ]
    if layout == "website":
        legend = fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), frameon=False,
            bbox_to_anchor=(0.5, 0.005), fontsize=7.8, handlelength=1.0,
            handleheight=1.0, columnspacing=1.25,
        )
        fig.subplots_adjust(left=0.006, right=0.994, top=0.97, bottom=0.19,
                            wspace=0.025)
    else:
        legend = fig.legend(
            handles=handles, loc="lower center", ncol=3, frameon=False,
            bbox_to_anchor=(0.5, -0.005), fontsize=7.2, handlelength=1.0,
            handleheight=1.0, columnspacing=1.4, labelspacing=0.35,
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.125,
                            wspace=0.02, hspace=0.02)
    for text in legend.get_texts():
        text.set_color("#52514e")
    return fig


def _save_figure(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "AssayLoop release figure renderer"}
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"facecolor": SURFACE, "bbox_inches": "tight", "pad_inches": 0.02}
        if suffix == "pdf":
            kwargs["metadata"] = metadata
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(output_stem.with_suffix(f".{suffix}"), **kwargs)
    # Matplotlib emits spaces after SVG path commands.  They are harmless to a
    # browser but make ``git diff --check`` report thousands of false-positive
    # whitespace errors for this generated artifact.
    svg = output_stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def render(
    data_dir: Path,
    output_stem: Path,
    appendix_output_stem: Path,
    font_path: Path | None,
) -> None:
    family = _cmu_serif(font_path)
    sunburst = _read_json(data_dir / "sunburst.json")
    results = _read_json(data_dir / "results.json")
    metrics = _validated_metrics(sunburst, results)

    _save_figure(_draw_figure(sunburst, metrics, "website"), output_stem)
    _save_figure(
        _draw_figure(sunburst, metrics, "appendix"), appendix_output_stem,
    )
    print(f"CMU family: {family}")
    print("EP-D: " + ", ".join(f"{name}={metrics[name]['ep_d']:.1f}" for name in ORDER))
    print(f"wrote {output_stem}.{{pdf,svg,png}}")
    print(f"wrote {appendix_output_stem}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--appendix-output-stem", type=Path,
        help=("output stem for the 2 x 3 paper figure; defaults to "
              "<output-stem>_appendix"),
    )
    parser.add_argument("--font-path", type=Path,
                        help="path to cmunrm.otf (normally found through kpsewhich)")
    args = parser.parse_args()
    appendix_output_stem = args.appendix_output_stem or args.output_stem.with_name(
        f"{args.output_stem.name}_appendix"
    )
    render(args.data_dir, args.output_stem, appendix_output_stem, args.font_path)


if __name__ == "__main__":
    main()
