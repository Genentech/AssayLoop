# AssayLoop Pages site

This folder is the GitHub Pages site at
<https://genentech.github.io/AssayLoop>. It is a pure static site — vanilla
HTML + JavaScript + JSON, with Plotly served from a CDN — so it deploys
directly from this directory on every push to `main`. There is no build step
at serve time; the three Python builders below run offline and write JSON and
images into `assets/`.

## Layout

```
docs/
├── .nojekyll                # disables Jekyll processing on GH Pages
├── _config.yml              # title/description fallback
├── index.html               # landing page: abstract, headline numbers, teaser figure
├── method.html              # the pipeline figure panel by panel + what EF measures
├── results.html             # the full-genome results table, plus LOPO / scaling / ablation
├── recovery.html            # recovery-curve explorer (59 methods × 20 screens)
├── diversity.html           # Effective Pathways / Vendi explorer + the sunbursts
├── umap.html                # BPMF gene-embedding scatter + the embedding figures
├── analysis.html            # handoff composition, LLM pathway shares, gene influence
├── cite.html                # BibTeX
├── build_data.py            # → assets/data/{summary,results,recovery_*,diversity}.json
├── build_figures.py         # runs the release figure scripts → assets/figures/ + figures.json
├── build_paper_panels.py    # rasterises the three vector PDFs; cuts Figure 1 into panels
├── build_umap.py            # → assets/data/gene_umap.json
├── ARCHITECTURE.md          # the framework's design doc — predates the site,
│                            #   not part of it, and nothing here links to it
└── assets/
    ├── css/site.css
    ├── js/site.js + one module per page
    ├── data/                # generated JSON
    └── figures/             # generated SVG / PNG / PDF
```

## No silent fallback

Every builder here follows the repo's standing rule: a missing input raises and
names the path it wanted. Nothing is substituted, and in particular nothing is
lifted out of the paper PDF to fill a gap.

That is a deliberate trade. A site that renders a complete-looking leaderboard
with three rows quietly absent is indistinguishable, to a reader, from those
three methods not having been evaluated — so the builders would rather fail
loudly and leave you a missing page than ship a plausible one. Where a value
genuinely is undefined (Effective Pathways below its retention floor) it
renders as an em dash and the page says why.

`build_figures.py` is the exception that proves the rule: it records failures
in `figures.json`, and `figure-strip.js` renders a stated "not available" card
in place of the missing figure, so a script that stopped working is *reported*
rather than either crashing the build or leaving a section quietly one panel
shorter. The build still exits non-zero, so a site with such a card on it is a
site that should not have been deployed.

`build_data.py` has one narrow escape hatch in the same spirit. If a method in
the results table has no result at all, the build stops — unless you name every
such method and say why:

```bash
python docs/build_data.py \
    --allow-unavailable "Qwen3.6-27B" "Qwen3.6-27B (base)" \
    --unavailable-note "The AssayLLM prediction files are not in the public bundle yet."
```

It refuses a name that is not in the table, refuses a name that *does* have a
result (so the flag cannot outlive the gap), and refuses the flag without a
note. The note and the names are written into `results.json`, and
`results.html` puts them in a banner above the table while greying those rows
and tagging them *not evaluated*. `diversity.json` carries them too, so the
retention-floor table does not claim a method fell below the floor when in
truth nobody ran it.

## Regenerate the data

From the repo root, with the package installed (`pip install -e .`):

```bash
# 1. the results table. --json dumps the same rows the LaTeX table prints,
#    so the site and the paper cannot drift.
python -m assayloop.scripts.full_genome_table --min-screen-freq 2 \
    --json output/tables/full_genome_baselines_f2.json

# 2. the recovery curves
python -m assayloop.scripts.export_recovery_curves

# 3. the BPMF gene clusters, for the embedding page
python -m assayloop.scripts.paper_bpmf_k10_export --k 10

# 4. the site JSON
python docs/build_data.py
python docs/build_umap.py
```

`build_data.py` writes:

| File | Source |
| --- | --- |
| `summary.json` | headline numbers for the landing page's stat grid |
| `results.json` | the rows from `full_genome_table.py --json` |
| `recovery_mean.json` | `output/analysis/recovery_curves_mean.csv` |
| `recovery_by_screen.json` | `output/analysis/recovery_curves_by_screen.csv` |
| `diversity.json` | EP-B/EP-S/EP-D, Vendi and EF, sliced out of `results.json` |
| `lopo.json` | the four full-test-set EF values reported in Figure 4B |

`build_umap.py` writes `gene_umap.json` from the BPMF checkpoint plus
`output/analysis/bpmf_k10_gene_clusters.tsv`. It projects with the same helper
and the same seed as the paper figure (`_bpmf_embedding.run_umap_cosine`,
seed 7), so a gene sits where the paper's BPMF organisation figure puts it.
That is why `build_figures.py` does not publish the static version: `umap.html`
already draws the same embedding, live and searchable.

## Rebuild the figures

```bash
python docs/build_figures.py --paper-dir /path/to/unzipped/paper
```

This re-runs each paper figure's own script with `MPLBACKEND=Agg`, copies the
PDF/SVG/PNG into `assets/figures/`, and writes `assets/data/figures.json`.

The pathway sunburst can also be rebuilt by itself from the two release JSON
artifacts; it does not require the original sweep directories or a Reactome
download. Install CMU Serif locally (or pass its OpenType file with
`--font-path`) and run:

```bash
MPLCONFIGDIR=/tmp/assayloop-mpl python docs/plot_pathway_sunburst_release.py
```

The renderer checks every non-random EP-B, EP-S, and EP-D value against
`results.json` before drawing. Its Random panel uses the published f2-universe
reference, 21.7 / 56.4 / 81.6, so stale or incorrectly aggregated data fails
loudly instead of yielding a plausible-looking figure. Each run writes both
the one-row website figure (`pathway_sunburst.*`) and the 2 × 3 paper figure
(`pathway_sunburst_appendix.*`).

That JSON carries only what the browser renders — title, caption, the section
of the paper the figure appears in, and the image. **The site is an
introduction to the paper, not a lab notebook about it**, so the script paths,
the commands, the required inputs, and the paper-vs-repo reproduction verdicts
stay on the build side: they are printed as a report at the end of every run.
If a regenerated figure differs from the paper's, the person building the site
sees it and decides what to do; the reader does not get a discrepancy notice
stapled to the figure.

Pages place generated figures with `<div data-figures="scaling,label_ablation"></div>`, and
`assets/js/figure-strip.js` fills every such slot from `figures.json`. Adding a
figure to a page is one attribute; nothing duplicates the card markup.

Three current-paper panel groups are also shipped directly in both web and
vector form: `paper_dataset_panels` (Figure 2E&ndash;G),
`paper_worked_example` (Figure 3E&ndash;G), and `bio_diversity_pairs_panel`
(Figure 5C). They are displayed directly by `index.html` and `analysis.html`
because they are already-composed excerpts of larger manuscript figures, not
standalone entries in the generated figure gallery. Keep the PNG/PDF pairs
together; Figure 5C additionally includes its source SVG.

`--paper-dir` is optional and drives only the reproduction check. Each
regenerated PDF is compared against `files/figures/<name>.pdf` under that
directory by inflating both files' Flate streams and hashing the drawing
operators, with object numbers, font subset tags and creation dates filtered
out. **This is not a rasterised comparison** — a different matplotlib or font
version changes the operators without changing the picture — so `differs` is a
flag for a human to look at the two files, not proof that the data moved.
`--paper-dir` is required for the three hand-drawn figures, whose only source
is the paper.

Useful flags: `--only KEY [KEY ...]` to rebuild one card, `--skip-run` to
re-collect artifacts already in `output/analysis/`, `--keep-going` to record
failures instead of stopping at the first one, and `--dry-run` to print the
commands.

### Environment

The figure scripts read from `output/analysis/` and write their figures back
there, so point `ASSAYLOOP_OUTPUT` at a scratch tree rather than building into
your results directory:

```bash
export ASSAYLOOP_OUTPUT=/tmp/figure-build   # written to
export ASSAYLOOP_RESULTS=/path/to/results   # read from
export ASSAYLOOP_PUBLISHED=/path/to/sweeps  # scripts/fetch_sweeps.sh bundle
export ASSAYLOOP_GROUND_TRUTH=...           # Figure 9/10 only
```

`full_genome_table.py` writes its LaTeX under `$ASSAYLOOP_OUTPUT/tables/`, so
the same scratch tree keeps the regenerated table out of your results too. The
external-baseline rows (BioBO, Haystacks) come from the downloaded sweep
bundle, which is why `ASSAYLOOP_PUBLISHED` belongs in this list and not only in
the figure section — without it those rows have no runs to read and the table
stops rather than printing them empty.

Two figures cannot be rebuilt from the released artifacts alone: the influence
figure needs an `OPENAI_API_KEY` (it conditions on a synthetic screen
description that is not in the shipped embedding cache), and the
network-recovery figures need the third-party STRING/CORUM/SIGNOR tables.
Without them the build records a failure, exits non-zero, and the page shows a
"not available" card where the figure should be. Fix the inputs before
deploying — do not deploy the card.

## Rasterise the vector figures

Three of the paper's figures — the teaser, the pipeline overview, and the
enrichment-factor explainer — are hand-drawn native-vector PDFs with no
matplotlib script behind them. Handing a PDF to the browser puts a PDF viewer
with its own scrollbar in the middle of the page, so they are rasterised:

```bash
pip install pymupdf            # a build-the-site dependency, not a runtime one
python docs/build_paper_panels.py
```

This writes `figure1.png`, `ef_metrics.png` and `pipeline.png` into
`assets/figures/`, and additionally cuts the pipeline figure into its eight
panels (`pipeline_panel_a.png` … `_h.png`) plus `assets/data/panels.json`,
which `method.html` renders one panel at a time with a caption each. At page
width the composed figure is a wall of 6 pt type; a panel at a time is legible.

The panel boxes are found by looking for the large rectangles in the PDF's own
drawing operators, not by hardcoded coordinates, and the run fails if it does
not find exactly eight — a re-exported figure with a different layout stops the
build instead of silently mis-captioning the panels. `build_figures.py` also
refuses to publish a hand-drawn figure whose PNG is missing, so the two scripts
cannot drift apart.

## Deploy on GitHub Pages

1. Push `docs/` to `main`.
2. Repo settings → **Pages → Build and deployment → Source: Deploy from a
   branch**, then **Branch: `main`**, **Folder: `/docs`**.
3. Pages publishes at `https://genentech.github.io/AssayLoop`.

`.nojekyll` is present, so the files are served as-is.

## Local preview

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

The pages `fetch()` their JSON, which needs an HTTP server — opening
`index.html` over `file://` will not work.

## Conventions

- **Voice.** Press-release register for the landing page's headline and TL;DR;
  paper-faithful wording for the abstract and every figure caption. Figure
  captions are quotes or light edits from `files/figure_files/*.tex`.
- **Colours.** Method families use the `--fam-*` custom properties in
  `site.css`, mirrored in `site.js`'s `FAMILIES`. Keep those two and the
  section grouping in `full_genome_table.py` in sync.
- **EF, formerly NVR.** Paper, code, and the `ef` field of `results.json` all
  agree now. The old name survives only where it is an on-disk contract with an
  already-published artifact: `n_hits_vs_random` in each per-screen
  `result.json`, and the `nvr_*` keys in a checkpoint's `history.json`.
  `results.html` says this in a callout, since anyone diffing the site against
  a downloaded artifact will hit it.
- **Em dashes are load-bearing.** `site.js`'s `fmt()` renders null as `—`.
  Effective Pathways returns null for a scope below its retention floor, and
  the paper prints `--` there. Never substitute a number.
