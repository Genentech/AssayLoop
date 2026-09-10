#!/usr/bin/env python
"""Regenerate every paper figure and record how it was made.

Runs each release figure script, copies its output into
``docs/assets/figures/``, and writes ``docs/assets/data/figures.json``.

Usage::

    ASSAYLOOP_RESULTS=/path/to/output \\
    python docs/build_figures.py --paper-dir /path/to/unzipped/paper

    # Just one, while iterating:
    python docs/build_figures.py --only scaling label_ablation

``--paper-dir`` is optional and only drives the reproduction check: each
regenerated PDF is compared against the corresponding file in the paper's
``files/figures/``. A figure that does not reproduce is *reported*, never
replaced with the paper's copy -- the whole point of regenerating is to find
out whether the released scripts still produce the released figures.

The reproduction report is for whoever is building the site, and it is printed
at the end of the run. It does not go into ``figures.json``: the site is an
introduction to the paper, not a lab notebook about it, so the JSON carries
only what a reader needs -- title, caption, where it sits in the paper, and the
image. Everything about scripts, commands, required inputs, and paper-vs-repo
drift stays on this side of the build. What *does* reach the JSON is the
failure list, because a figure that could not be built has to be visible on the
page as missing rather than quietly absent.

Three of the paper's figures are hand-drawn vector diagrams with no generating
script. Their PDFs are carried through from the paper and rasterised for the
web by ``build_paper_panels.py``, which must have run first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assayloop import config  # noqa: E402

log = logging.getLogger("build_figures")

DOCS = Path(__file__).resolve().parent
FIG_OUT = DOCS / "assets" / "figures"
DATA_OUT = DOCS / "assets" / "data"
ANALYSIS = config.OUTPUT_PATH / "analysis"
REPO_URL = "https://github.com/Genentech/AssayLoop"
SRC_PREFIX = "src/assayloop/scripts"

#: An SVG bigger than this is shipped as PNG instead. The sunburst and the
#: gene-embedding scatter have tens of thousands of vector elements each; a
#: 6 MB SVG is a worse web asset than a 300 dpi PNG, and the PDF link is
#: already there for anyone who wants the vector.
SVG_BUDGET = 900_000


@dataclass
class Figure:
    key: str
    title: str
    caption: str
    section: str
    paper_pdf: str
    #: Module to run, or None for a hand-drawn diagram with no script.
    module: str | None = None
    args: list[str] = field(default_factory=list)
    #: Basename (no suffix) the script writes into output/analysis/.
    stem: str | None = None
    #: For the build report, not the site: how this figure differs from the
    #: paper's copy, or anything else the next person to rebuild it needs.
    note: str = ""
    #: For the build report, not the site: inputs beyond the standard caches.
    needs: str = ""

    @property
    def hand_drawn(self) -> bool:
        return self.module is None


FIGURES = [
    Figure(
        key="pipeline",
        title="Overview of AssayLoop.",
        caption=(
            "AssayBench becomes a sequential decision problem (A&ndash;B). AssayFormer is "
            "initialised from BPMF embeddings, trained to predict assay hits from prior "
            "experimental context, then improved with RL under a context-delta reward that "
            "pays only for using the feedback (C&ndash;E). Three paradigms for hit discovery "
            "(F); AssayLoop combines the second and third (G&ndash;H)."),
        section="Figure 3 &middot; Methods",
        paper_pdf="assayloop-pipeline-paper-v6-native.pdf",
        note="Hand-drawn schematic. No script generates it. method.html shows "
             "it panel by panel from build_paper_panels.py.",
    ),
    Figure(
        key="figure1",
        title="The task, end to end.",
        caption=(
            "A screen is a library of genes, a phenotype, and a hit set. A method sees the "
            "phenotype and whatever it has already assayed, and must choose the next hundred "
            "genes. Ten rounds, then the score."),
        section="Figure 1 &middot; Introduction",
        paper_pdf="assaybench-loop-figure1-v19.pdf",
        note="Hand-drawn schematic. No script generates it.",
    ),
    Figure(
        key="ef_metrics",
        title="Metrics used in AssayBench-Loop.",
        caption=(
            "Adjusted EF and nAUC, effective-budget terms, a worked comparison across "
            "hit-rich and hit-sparse screens, recovery trajectories, and the complementary "
            "diagnostics reported in the paper."),
        section="Supplementary Figure 6 &middot; Metrics",
        paper_pdf="assayloop-ef-metrics-v15.pdf",
        note="Hand-drawn schematic. No script generates it. "
             "The metric itself is <code>assaybench.benchmark.sequential."
             "enrichment_factor_from_value</code>.",
    ),
    Figure(
        key="scaling",
        title="Performance scales with data, but not model size.",
        caption=(
            "AssayFormer models trained on 1 to all training screens. Both SFT and RL improve "
            "with more screens, and RL sits above SFT throughout. The gain concentrates in the "
            "first few acquisition steps &mdash; by 20,000 genes sampled even a random policy "
            "has found most of the hits."),
        section="Figure 4 / Supplementary Figure 10 &middot; Results",
        paper_pdf="scaling_combined_lines.pdf",
        module="assayloop.scripts.scaling_law_plot",
        # This one script writes 17 panels into its own subdirectory rather
        # than straight into analysis/; the paper uses the combined-lines one.
        stem="scaling_law_v2/scaling_combined_lines",
        needs="the scaling sweep (<code>scaling_law_sweep.py</code> then "
              "<code>scaling_law_eval.py</code>)",
        # The regenerated PDF is byte-identical to the copy the research repo
        # holds, so the released script reproduces the current artifact
        # exactly; it is the paper that embeds an earlier render.
        note="This regenerates bit-for-bit against the current artifact, but "
             "not against the copy embedded in the paper, which is an earlier "
             "render of the same panel.",
    ),
    Figure(
        key="pathway_sunburst",
        title="Biological pathway distribution of each method.",
        caption=(
            "Proposed genes categorised by Reactome pathway across six representative methods. "
            "Inner ring: the eight top-level Reactome categories plus a pooled remainder. "
            "Outer ring: Reactome's second tier. At the centre, the effective number of "
            "pathways &mdash; the count of equally-weighted pathways that would produce the "
            "same spread."),
        section="Figure 5A / Supplementary Figure 7 &middot; Results",
        paper_pdf="pathway_sunburst.pdf",
        # This release renderer consumes the two shipped website JSON files,
        # validates their overlapping metrics, and therefore does not require
        # the private sweep directories used by the original analysis script.
        # It also writes a 2 x 3 pathway_sunburst_appendix companion for the paper.
        module="docs.plot_pathway_sunburst_release",
        args=["--output-stem", str(ANALYSIS / "pathway_sunburst")],
        stem="pathway_sunburst",
        needs="CMU Serif (install locally or pass <code>--font-path</code>)",
    ),
    Figure(
        key="llm_pathway_heatmap",
        title="LLMs converge to similar distributions of biology across families.",
        caption=(
            "Top-level Reactome composition of requested genes, aggregated across the 20 test "
            "screens. Random is the pathway distribution produced by uniform sampling over the "
            "candidate universe. Cells show the ratio of the observed pathway share to Random, "
            "with red indicating over-representation and blue under-representation. LLMs share "
            "a strong emphasis on RNA and protein metabolism and under-represent bulk metabolism "
            "and developmental biology. Differences among LLMs are small compared with their "
            "differences from baseline methods such as Screen-kNN."),
        section="Supplementary Figure 5 &middot; Results",
        paper_pdf="llm_pathway_heatmap.pdf",
        module="assayloop.scripts.plot_llm_pathway_heatmap",
        stem="llm_pathway_heatmap",
        needs="the Reactome GMT (<code>scripts/fetch_gene_sets.sh</code>)",
    ),
    Figure(
        key="rollout_composition",
        title="Qualitative analysis of AssayLoop on two biologically distinct screens.",
        caption=(
            "The figure compares AssayFormer, Gemini-3.1-Pro, and the AssayLoop handoff on an "
            "NF-&kappa;B / TNF signaling screen and an AAV transgene silencing screen. In each "
            "panel, the upper histogram shows the Reactome composition of each proposed batch "
            "and the lower histogram shows the composition of its hits. Gemini and AssayFormer "
            "begin with substantially different pathway compositions. Gemini finds more hits in "
            "the first three rounds but drops off afterward; once AssayFormer takes over, it "
            "proposes different genes with a higher hit yield. AssayFormer also selects more "
            "genes in the Other category, consistent with Gemini favoring well-known genes and "
            "pathways. Gemini does not always use its full acquisition budget."),
        section="Supplementary Figure 1 &middot; Results",
        paper_pdf="paper_handoff_composition_pathway.pdf",
        module="assayloop.scripts.paper_handoff_composition",
        args=["--device", "auto"],
        stem="paper_handoff_composition_pathway",
        needs="the AssayFormer checkpoint and the Reactome GMT",
    ),
    Figure(
        key="influence",
        title="Candidate relationships beyond canonical annotations.",
        # The biologist-written interpretation from the paper, minus its
        # unresolved LaTeX citation keys. The surrounding prose names the four
        # databases used for filtering.
        caption=(
            "After removing gene pairs represented in STRING, CORUM, SIGNOR, and Reactome, we "
            "examine the strongest remaining influences as candidates for hypothesis generation. "
            "Among the positive pairs, adding MYC to the hit history raises the predicted "
            "probability of SMU1 and PRPF4. This association with spliceosome factors agrees with "
            "evidence that oncogenic MYC regulates spliceosome programs and creates dependence on "
            "core splicing machinery. Adding MDM2 raises EMG1 and PNO1, two factors required for "
            "small-subunit ribosome biogenesis. Disrupting ribosome biogenesis can stabilize p53 "
            "through the RPL5/RPL11-5S-RNP-MDM2 nucleolar-surveillance pathway; specifically, "
            "PNO1 depletion has been shown to increase RPL11-MDM2 association and stabilize p53 "
            "in TP53-wild-type colorectal cancer cells.<br><br>Among the negative influences, the "
            "model suggests both novel connections, such as PIK3CA to MIS18BP1, and relationships "
            "with pathway-level precedent. SMAD4 to NDUFS7 and NFU1 connects SMAD4 with "
            "mitochondrial function. SMAD4 loss reduces mitochondrial respiration and confers "
            "resistance to biguanide complex-I inhibitors in pancreatic cancer models, providing "
            "a context-specific rationale for NDUFS7, although these specific pairs remain "
            "experimentally unvalidated. NRAS to AMOTL2 has similarly indirect precedent: "
            "oncogenic NRAS activates Hippo signaling in melanocytes, while parallel BRAF-driven "
            "MAPK activation reduces expression of the YAP/TAZ target AMOTL2; AMOTL2 itself has "
            "not been directly tested under NRAS mutation.<br><br>The positive pairs recover "
            "established biological dependencies, while the negative updates suggest "
            "uncharacterized pathway relationships. Together they provide specific, testable "
            "hypotheses about vulnerabilities and mechanisms of resistance."),
        section="Figure 5B / Supplementary Figure 9 &middot; Results",
        paper_pdf="paper_influence_figure.pdf",
        module="assayloop.scripts.paper_influence_figure",
        # --use-cache reads the influence values the paper's run already wrote
        # to analysis/paper_influence_data.json. Without it the script embeds
        # one fixed synthetic screen description ("A genome-wide CRISPR
        # knockout screen to identify essential genes."), which is not a real
        # screen and so is in neither the shipped embedding cache nor any
        # run's -- it would need a live OPENAI_API_KEY, and it does not fall
        # back to a zero vector. The cached values are the published ones, so
        # the site shows the paper's figure either way.
        args=["--device", "auto", "--use-cache"],
        stem="paper_influence_figure",
        needs="<code>analysis/paper_influence_data.json</code> from the paper's "
              "run, or an <code>OPENAI_API_KEY</code> to recompute it",
    ),
    Figure(
        key="influence_heatmap",
        title="Influence heatmap over canonical drivers.",
        caption=(
            "The same measurement as a dense matrix. The first 12 genes are canonical cancer "
            "drivers; the next 19 are representative functional-module genes."),
        section="Supplementary Figure 8 &middot; Results",
        paper_pdf="gene_matrix_featured_heatmap.pdf",
        module="assayloop.scripts.analyze_gene_matrices",
        stem="gene_matrix_featured_heatmap",
        needs="the AssayFormer checkpoint",
    ),
    # Supplementary Figure 4 -- the static BPMF organisation panel -- is
    # deliberately not here. umap.html plots the same embedding live from
    # gene_umap.json, with the clusters, the hit rates and a gene search, and
    # two versions of one picture on one site is one too many.
    Figure(
        key="recovering_biology",
        title="Recovering textbook biology does not predict task usefulness.",
        caption=(
            "Initial embeddings are evaluated by how well cosine similarity recovers "
            "gene&ndash;gene relationships from STRING, CORUM, SIGNOR, and Reactome. AUROC is "
            "calculated for each database and averaged across all four. Embeddings that most "
            "closely recover these databases, including GenePT and K562, are harder to train "
            "AssayFormer on, whereas matrix-factorization initializations learned from "
            "historical screens are most useful."),
        section="Supplementary Figure 3 &middot; Results",
        paper_pdf="gene_embedding_init_story.pdf",
        module="assayloop.scripts.plot_embedding_init_story",
        stem="gene_embedding_init_story",
        needs="the gene-relationship databases (<code>scripts/fetch_gene_sets.sh</code>)",
    ),
    Figure(
        key="label_ablation",
        title="Performance with and without per-round hit labels.",
        caption=(
            "The same ten-round loop, run once with the previous rounds' hit labels in the "
            "prompt and once without. The gap separates a method's ability to use experimental "
            "feedback from its warm-start prior. Every LLM family benefits from the feedback, "
            "and so do AssayFormer and the AssayLoop system built on it &mdash; the two rows "
            "below the rule, which are blinded by seeing no readout at all rather than by a "
            "prompt with the labels stripped."),
        section="Supplementary Figure 11 &middot; Results",
        paper_pdf="llm_label_ablation.pdf",
        module="assayloop.scripts.plot_label_ablation",
        stem="llm_label_ablation",
        note="Numbers are cached in the script itself; re-run the underlying sweeps to change "
             "them.",
    ),
    Figure(
        key="embedding_drift",
        title="Embedding drift during training for all ablated embedding types.",
        caption=(
            "The effect of training on embedding geometry differs dramatically by embedding "
            "type. BPMF embeddings show near-perfect preservation of gene neighborhoods, while "
            "other initializations undergo substantial reorganization during supervised "
            "pretraining."),
        section="Supplementary Figure 2 &middot; Results",
        paper_pdf="gene_embedding_drift_by_source.pdf",
        module="assayloop.scripts.plot_embedding_drift_by_source",
        stem="gene_embedding_drift_by_source",
        needs="the AssayFormer checkpoints for each embedding source",
    ),
]


class FigureBuildError(RuntimeError):
    """A figure script failed, or produced nothing where it said it would."""


# ------------------------------------------------------------------ PDF diff

_SUBSET_TAG = re.compile(rb"/[A-Z]{6}\+")
_OBJ_HEADER = re.compile(rb"\d+ 0 obj")


def _content_digest(pdf: Path) -> str | None:
    """A hash of a PDF's drawing operators, or None if it can't be read.

    There is no PDF library in this environment, so this walks the raw bytes,
    inflates every Flate stream it finds, and hashes the concatenation with the
    volatile parts filtered out: object numbers, font subset tags, and the
    creation date. Two matplotlib figures drawn from the same data hash the
    same; two drawn from different data do not.

    It is *not* a rasterised comparison. A different matplotlib or font version
    will change the operators without changing the picture, so a mismatch is a
    flag for a human to look at the two files, not proof that the data moved.
    """
    try:
        raw = pdf.read_bytes()
    except OSError:
        return None
    chunks = []
    pos = 0
    while True:
        start = raw.find(b"stream", pos)
        if start < 0:
            break
        start += len(b"stream")
        if raw[start:start + 2] == b"\r\n":
            start += 2
        elif raw[start:start + 1] in (b"\n", b"\r"):
            start += 1
        end = raw.find(b"endstream", start)
        if end < 0:
            break
        try:
            chunks.append(zlib.decompress(raw[start:end]))
        except zlib.error:
            pass  # not Flate-encoded (an embedded image, say) -- skip it
        pos = end + len(b"endstream")
    if not chunks:
        return None
    blob = b"\n".join(chunks)
    blob = _SUBSET_TAG.sub(b"/SUBSET+", blob)
    blob = _OBJ_HEADER.sub(b"N 0 obj", blob)
    import hashlib
    return hashlib.sha256(blob).hexdigest()


_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+"
                       rb"([\d.\-]+)\s+([\d.\-]+)")


def _mediabox(pdf: Path) -> list[float] | None:
    try:
        m = _MEDIABOX.search(pdf.read_bytes())
    except OSError:
        return None
    return [round(float(g), 2) for g in m.groups()] if m else None


def compare_to_paper(built: Path, paper: Path) -> dict:
    """Compare a regenerated PDF against the paper's copy."""
    if not paper.is_file():
        return {"status": "no-reference",
                "detail": f"No paper copy at {paper}."}
    a, b = _content_digest(built), _content_digest(paper)
    box_a, box_b = _mediabox(built), _mediabox(paper)
    same_box = box_a is not None and box_a == box_b
    if a is not None and a == b:
        status = "identical"
        detail = "Drawing operators match the paper's PDF byte for byte."
    elif a is None or b is None:
        status = "not-comparable"
        detail = "One of the PDFs has no inflatable content stream."
    elif same_box:
        status = "differs"
        detail = ("Same page size, different drawing operators. Check the two "
                  "side by side: a matplotlib or font version change produces "
                  "this too, not only a change in the data.")
    else:
        status = "differs"
        detail = (f"Page size changed: paper {box_b}, regenerated {box_a}. "
                  "The figure is not the one in the paper.")
    return {"status": status, "detail": detail,
            "mediabox_built": box_a, "mediabox_paper": box_b,
            "bytes_built": built.stat().st_size,
            "bytes_paper": paper.stat().st_size}


# --------------------------------------------------------------------- build

def run_script(fig: Figure, python: str, dry_run: bool) -> tuple[list[str], str]:
    """Run one figure script.

    Returns the command and the tail of its output. The output is kept even on
    success, because the interesting failure mode for these scripts is exiting
    0 after logging that it skipped everything -- :func:`collect` needs
    something to show when the PDF it expected is not there.
    """
    cmd = [python, "-m", fig.module, *fig.args]
    log.info("[%s] %s", fig.key, " ".join(cmd))
    if dry_run:
        return cmd, ""
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = "\n".join(((proc.stderr or "") + (proc.stdout or "")).splitlines()[-25:])
    if proc.returncode != 0:
        # The reason goes on the *first* line, not the last. A Python
        # traceback puts the exception at the bottom, and the gallery's
        # failure list only has room for one line -- "exited 1" as that line
        # tells a reader nothing about which input was missing.
        reason = next((ln for ln in reversed(tail.splitlines()) if ln.strip()),
                      f"exited {proc.returncode} with no output")
        raise FigureBuildError(
            f"{fig.key}: {reason}\n`{' '.join(cmd)}` exited "
            f"{proc.returncode}.\n{tail}")
    return cmd, tail


def collect(fig: Figure, paper_dir: Path | None, cmd: list[str],
            output: str = "") -> dict:
    """Copy a figure's artifacts into docs/ and build its build record.

    The record is the full picture -- script, command, required inputs, and how
    the regenerated PDF compares to the paper's. :func:`web_entry` cuts it down
    to the handful of fields the site actually renders.
    """
    entry = {
        "key": fig.key,
        "title": fig.title,
        "caption": fig.caption,
        "section": fig.section,
        "note": fig.note,
        "needs": fig.needs,
        "hand_drawn": fig.hand_drawn,
        "paper_pdf": fig.paper_pdf,
    }

    if fig.hand_drawn:
        # Nothing to regenerate: these are vector diagrams drawn by hand, and
        # the paper's PDF is the only source. The web rendering is a PNG, which
        # build_paper_panels.py makes -- shipping the PDF for the browser to
        # embed puts a scrolling PDF viewer in the middle of the page.
        if paper_dir is None:
            raise FigureBuildError(
                f"{fig.key} is a hand-drawn figure with no generating script, so "
                f"its only source is the paper. Pass --paper-dir to include it, "
                f"or --only to skip it.")
        src = paper_dir / "files" / "figures" / fig.paper_pdf
        if not src.is_file():
            raise FigureBuildError(f"{fig.key}: no such file {src}")
        dst = FIG_OUT / f"{fig.key}.pdf"
        shutil.copy2(src, dst)
        png = FIG_OUT / f"{fig.key}.png"
        if not png.is_file():
            # Loud, because the alternative is a card with a broken image on
            # it. The PDF has just been refreshed, so a stale PNG would be
            # wrong in the same way and is not worth accepting either.
            raise FigureBuildError(
                f"{fig.key}: copied {dst.name} but there is no {png.name} to "
                f"show. Rasterise it with `python docs/build_paper_panels.py "
                f"--only {fig.key}` (needs pymupdf).")
        entry.update(script=None, command=None, pdf=_rel(dst),
                     image=_rel(png), image_format="png",
                     reproduction={"status": "hand-drawn",
                                   "detail": "Carried through from the paper; "
                                             "no script produces it."})
        return entry

    stem = fig.stem or fig.key
    produced = {}
    for suffix in ("pdf", "svg", "png"):
        src = ANALYSIS / f"{stem}.{suffix}"
        if not src.is_file():
            continue
        dst = FIG_OUT / f"{fig.key}.{suffix}"
        shutil.copy2(src, dst)
        produced[suffix] = dst
    if "pdf" not in produced:
        if not cmd:
            # --skip-run: nothing was executed, so "returned 0 but wrote no
            # PDF" would blame a script that never ran and print an empty
            # backtick pair as the command.
            raise FigureBuildError(
                f"{fig.key}: --skip-run, and there is no "
                f"{ANALYSIS / (stem + '.pdf')} to collect. Drop --skip-run to "
                f"build it, or check that the stem is right.")
        raise FigureBuildError(
            f"{fig.key}: `{' '.join(cmd)}` returned 0 but wrote no "
            f"{ANALYSIS / (stem + '.pdf')}. Either the stem is wrong or the "
            f"script skipped the panel and exited 0 anyway. Its last words:\n"
            + (output or "(no output)"))

    # Prefer SVG, but not at any size. See SVG_BUDGET.
    if "svg" in produced and produced["svg"].stat().st_size <= SVG_BUDGET:
        image, fmt = produced["svg"], "svg"
    elif "png" in produced:
        image, fmt = produced["png"], "png"
        if "svg" in produced:
            log.info("[%s] SVG is %.1f MB, over the %.1f MB budget -- shipping "
                     "the PNG and dropping it", fig.key,
                     produced["svg"].stat().st_size / 1e6, SVG_BUDGET / 1e6)
            produced["svg"].unlink()
    else:
        raise FigureBuildError(
            f"{fig.key}: no SVG under {SVG_BUDGET} bytes and no PNG either, so "
            f"there is nothing to show on the card.")

    module_path = (fig.module.replace(".", "/") + ".py"
                   if fig.module.startswith("docs.")
                   else f"{SRC_PREFIX}/{fig.module.rsplit('.', 1)[1]}.py")
    entry.update(
        script=module_path,
        script_url=f"{REPO_URL}/blob/main/{module_path}",
        command=" ".join(["python", "-m", fig.module, *fig.args]),
        pdf=_rel(produced["pdf"]),
        image=_rel(image),
        image_format=fmt,
        reproduction=compare_to_paper(
            produced["pdf"], paper_dir / "files" / "figures" / fig.paper_pdf)
        if paper_dir else {"status": "unchecked",
                           "detail": "Run with --paper-dir to compare against "
                                     "the paper's PDF."},
    )
    return entry


def _rel(path: Path) -> str:
    return str(path.relative_to(DOCS))


#: The only keys that reach the browser. Everything else in a build record --
#: script, command, needs, note, reproduction, paper_pdf -- is a note to the
#: people rebuilding the site and stays in the build report.
WEB_FIELDS = ("key", "title", "caption", "section", "image", "pdf")


def web_entry(record: dict) -> dict:
    return {k: record[k] for k in WEB_FIELDS if record.get(k) is not None}


def report(records: list[dict]) -> None:
    """Print the reproduction status of every figure that was built."""
    if not records:
        return
    log.info("")
    log.info("Reproduction against the paper's PDFs:")
    for r in records:
        rep = r["reproduction"]
        log.info("  %-20s %s", r["key"], rep["status"])
        if rep["status"] in ("differs", "not-comparable"):
            log.warning("      %s", rep["detail"])
        if r["note"]:
            log.info("      note: %s", re.sub(r"<[^>]+>", "", r["note"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-dir", type=Path,
                    help="Unzipped paper source, for the reproduction check. "
                         "Required for the three hand-drawn figures.")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="Build just these figure keys.")
    ap.add_argument("--skip-run", action="store_true",
                    help="Collect artifacts already in output/analysis/ "
                         "without re-running the scripts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands and stop.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep-going", action="store_true",
                    help="Record a script that fails and carry on, instead of "
                         "stopping at the first failure.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wanted = FIGURES
    if args.only:
        keys = {f.key for f in FIGURES}
        unknown = sorted(set(args.only) - keys)
        if unknown:
            ap.error(f"unknown figure key(s): {', '.join(unknown)}. "
                     f"Known: {', '.join(sorted(keys))}")
        wanted = [f for f in FIGURES if f.key in set(args.only)]

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for fig in wanted:
        try:
            cmd, output = (([], "") if fig.hand_drawn or args.skip_run
                           else run_script(fig, args.python, args.dry_run))
            if args.dry_run:
                continue
            records.append(collect(fig, args.paper_dir, cmd, output))
        except FigureBuildError as err:
            if not args.keep_going:
                raise
            log.error("%s", err)
            failures.append({"key": fig.key, "error": str(err)})

    if args.dry_run:
        return 0

    report(records)
    entries = [web_entry(r) for r in records]
    out = DATA_OUT / "figures.json"

    # --only rebuilds a subset, so its result has to be merged into the
    # manifest rather than replacing it. Writing `entries` straight out would
    # leave the pages with three figures and no error anywhere saying the other
    # nine were dropped -- exactly the silent-thinning failure the rest of this
    # file refuses to allow. Rebuilt keys win; untouched keys keep their entry
    # and their recorded failure. Order follows FIGURES either way, so nothing
    # reshuffles depending on what was last rebuilt.
    if args.only and out.is_file():
        prev = json.loads(out.read_text())
        rebuilt = {e["key"] for e in entries} | {f["key"] for f in failures}
        entries += [e for e in prev.get("figures", []) if e["key"] not in rebuilt]
        failures += [f for f in prev.get("failures", []) if f["key"] not in rebuilt]
        order = {f.key: i for i, f in enumerate(FIGURES)}
        entries.sort(key=lambda e: order.get(e["key"], len(order)))
        failures.sort(key=lambda f: order.get(f["key"], len(order)))

    payload = {
        "schema": 2,
        "n_figures": len(entries),
        # Recorded, not swallowed. Every page renders this list for the keys it
        # asked for, so a script that stopped working shows up as a stated
        # absence rather than a section that is quietly one figure shorter.
        "failures": failures,
        "figures": entries,
    }
    out.write_text(json.dumps(payload, indent=1, separators=(",", ":")) + "\n")
    log.info("wrote %s (%d figures, %d failures)", out, len(entries), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
