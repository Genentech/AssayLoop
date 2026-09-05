# AssayLoop — full reference

Every command, flag, path and baseline in the repository. [`README.md`](README.md) is the
short version: install, load the model, run it, score it. This one is written to be read
end to end (by a person or an agent) when the short version stops being enough — it is the
place that says which checkpoint backs which table row, what raises when a download is
missing, and what every environment variable does.

Code for **"Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens."**

A CRISPR screen has ~20,000 candidate genes and a budget for testing a few hundred at a
time. AssayLoop chooses which genes to test next, round after round, so that hits show up
early. It combines two things that usually trade off against each other:

- **ASSAYFORMER** — a transformer trained on 1,349 historical screens to score every gene
  in one forward pass, conditioned on the screen description *and* on the hits/non-hits
  observed so far. It adapts to feedback but knows only what the training screens taught it.
- **An LLM warm start** — GLM-5.1, Gemini-3.1-Pro or GPT-5.6 picks the first rounds from
  its own biological knowledge. It has broad priors but learns poorly in-context.

**ASSAYLOOP** is the handoff: the LLM picks the first *k* rounds (k=2–3), then AssayFormer
takes over with the LLM's observations as context. On 20 held-out screens it recovers
**27.7% of hits after assaying 5% of the library**, ahead of every baseline evaluated.

This repository is the experiment harness: the active-learning loop, every baseline in the
paper, AssayFormer training/RL, and the table and figure scripts. The screen corpus and the
shared metrics live in the companion [`assaybench`](https://pypi.org/project/assaybench/)
package.

**[genentech.github.io/AssayLoop](https://genentech.github.io/AssayLoop)** — the project
site: the results table, an explorer for the recovery curves and the gene embedding, and a
gallery giving the exact command that regenerates every figure in the paper. It is built
from [`docs/`](docs/) by re-running the scripts in this repo, so it doubles as a
reproducibility check on them.

---

## Contents

- [Install](#install)
- [Data you download yourself](#data-you-download-yourself)
- [Quickstart](#quickstart)
- [How the loop is put together](#how-the-loop-is-put-together)
- [Screen sets](#screen-sets)
- [Metrics](#metrics)
- [Models and acquisitions](#models-and-acquisitions)
- [Reproducing the paper's main table](#reproducing-the-papers-main-table)
- [AssayFormer](#assayformer)
- [Outputs and tracing](#outputs-and-tracing)
- [Environment variables](#environment-variables)
- [Extending the framework](#extending-the-framework)
- [License](#license) · [Third-party data](#third-party-data) · [Citation](#citation)

---

## Install

Python 3.11+. The project uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/Genentech/AssayLoop
cd assayloop
uv sync                    # CPU only: the loop, the LLM baselines, the analysis scripts
uv sync --extra torch      # + AssayFormer (train-ranker / eval-ranker) and the Bayesian MLP
```

`torch` is a 500 MB+ install and every module that needs it imports it lazily, which is why
it is opt-in. Nothing else is gated.

**No credentials are required** to run the offline parts, and none are required to reproduce
the AssayFormer numbers: every public screen description ships pre-embedded (see
[Text embeddings](#text-embeddings)). You need an API key only for the LLM baselines, and
only for the provider you actually call.

## Data you download yourself

Three datasets are not redistributed here. Each has a fetch script that downloads from the
original site — which is also how you accept that source's terms.

| What | Command | Needed for |
|---|---|---|
| PRESAGE gene embeddings (3.4 GB packed, ~11 GB unpacked) | `bash scripts/fetch_presage_cache.sh` | `knn`, `rf`, `biobo`, `llmnn`, Vendi diversity |
| MSigDB gene sets (C5 GO:BP, C2 CP) | `bash scripts/fetch_gene_sets.sh` | `bio_ucb`, pathway-overlap metric, sunburst figure |
| STRING / CORUM / SIGNOR interaction tables | `bash scripts/fetch_ground_truth.sh` | network-recovery analysis (Figure 9), CORUM-labelled UMAPs |

One more download is ours rather than a third party's — the cached results of the runs we
cannot ask you to pay to repeat:

| What | Command | Needed for |
|---|---|---|
| Published sweeps + call logs (35 MB) | `bash scripts/fetch_sweeps.sh` | exact raw-response replay for the LLM and external-baseline rows of Table 1 |

**Nothing silently substitutes for a missing download.** A model that needs the PRESAGE
cache raises `MissingPresageCache` (or `MissingPresageSource` for one absent source) and
names what to fetch; the analysis scripts raise `MissingConfiguredPath` naming the
environment variable. There is no degraded-embedding path that would quietly produce a
number that looks like the paper's but isn't.

Re-running a fetch script is cheap — already-downloaded files are skipped unless you pass
`--force`.

<a id="interaction-ground-truth"></a>
### Interaction ground truth, in detail

```bash
scripts/fetch_ground_truth.sh                    # -> data/ground_truth
export ASSAYLOOP_GROUND_TRUTH="$PWD/data/ground_truth"
```

The script downloads STRING v11.5, CORUM 4.1 and SIGNOR 3.0 into `<target>/_raw/`, then runs
`scripts/convert_ground_truth.py` to write `STRING-HUMAN/`, `CORUM-HUMAN/` and
`SIGNOR-HUMAN/`. Only STRING downloads unattended; CORUM and SIGNOR serve their bulk files
through pages that may need a form or may be blocked on a locked-down network. When a fetch
fails the script prints which file to get, from which page, and where to put it, and the
converter exits non-zero listing what is missing rather than building a partial ground truth.

The paper's tables were built from an internal copy of these sources. The public rebuild
reproduces it closely: STRING comes out exact (all 593,222 protein pairs, same score ≥ 400
subset; 42 scores differ by one unit from rounding), and CORUM and SIGNOR reproduce their
complexes, entities and edge sets exactly. Gene-*pair* counts land ~3% **higher** than the
paper's, because public STRING supplies a symbol for every protein while the internal
mapping covered 17,565 of 17,848 — the paper's counts were deflated by missing symbols, not
by a narrower edge set. `convert_ground_truth.py` prints a built-vs-paper table on every run.

All three pinned releases are **CC BY 4.0**, so tables you derive from them are
redistributable with attribution.

## Quickstart

```bash
# Random baseline on the 20 public test screens, 10 rounds of 100 genes. No keys, no data.
uv run assayloop run --model null --acq random --n-steps 10

# What the loop resolved: provider, endpoint, cache paths, trace directory.
uv run assayloop info
```

That writes one `output/runs/<run_id>/result.json` per screen plus an aggregate, and prints
a per-screen and aggregate summary. Everything else in this README is a variation on it.

## How the loop is put together

Five abstractions. They live in the `assaybench` package (`assaybench.core`), not in this
repo, so that someone can benchmark their own policy with `pip install assaybench` alone —
no model zoo, no LLM clients, no paper scripts. `from assayloop import Task, Model, ...`
re-exports them for convenience.

| Abstraction | Role |
|---|---|
| `Task` | Defines the candidate pool, what revealing a gene returns, and the ground truth |
| `Model` | Scores every unacquired candidate (optionally with uncertainty) |
| `AcquisitionFunction` | Picks the next `batch_size` candidates |
| `Metric` | Scores the run |
| `SequentialLoop` | Wires them together and runs the rounds |

Each is an abstract base class; adding a method means implementing one of them. Note the
split between `Model` and `AcquisitionFunction`: some methods do all their work in the model
and pair with `greedy` (AssayFormer, kNN, BPMF), while the LLM acquisitions do all their
work in the acquisition and pair with `--model null`.

The canonical task is `src/assayloop/tasks/gene_batch.py`: one screen, pick `batch_size`
genes per round, reward finding hits early. Which screens you run it on is a separate
concern, in `src/assayloop/tasks/screen_sets.py` — see [Screen sets](#screen-sets) below.
Import both through the package (`from assayloop.tasks import make_task, load_screens`).

### Shortfall is never padded

When an acquisition under-supplies — an LLM returns fewer than `batch_size` usable genes,
repeats an acquired gene, or names a symbol outside the configured candidate universe — the
loop **does not** fill the gap with random picks. The batch is recorded at its true size and
the difference is tracked as `shortfall_count_step` / `shortfall_frac` on every round, so a
method's score reflects only genes it actually chose. A valid f2-universe gene that is absent
from the current screen library is still retained; the paper's SF column records that
separate out-of-library rate.

`--max-shortfall-frac` (default `1.0`, i.e. off) aborts a screen whose running shortfall
exceeds the threshold and counts it as `n_failed` in the aggregate, rather than letting a
broken policy report a diluted number.

## Screen sets

Screens come from [`Genentech/assaybench`](https://huggingface.co/datasets/Genentech/assaybench)
on the Hugging Face Hub — 1,901 CRISPR screens from BioGRID ORCS, split temporally by
`yearfold0` into 1,349 train / 218 validation / 334 test. The two sets the paper reports on
are curated 20-screen subsets of the validation and test folds. `--screen-set` resolves:

| Value | What |
|---|---|
| `public` (default) | the curated **20-screen test set** the paper reports |
| `public_validation` | the curated 20-screen validation set (checkpoint selection) |
| `public_train` | the full 1,349-screen training fold |
| `public_val` | the full 218-screen validation fold |
| `public_test` | the full 334-screen test fold |
| `/path/to.yaml` | your own list |

`public_test` is the whole fold `public` is drawn from, for a broader evaluation than the
paper's. One caveat: the shipped screen-description embeddings cover the curated 20 of that
fold, not all 334, so ASSAYFORMER on `public_test` needs an `OPENAI_API_KEY` — it raises and
says so rather than substituting a different vector. Every other method runs offline.

`--screen <dataset_name,...>` overrides the set and runs exact screens.

The curated sets are restricted to genome-wide libraries (`num_genes ∈ [18000, 22000]`) so
per-screen `frac_acquired` stays within ~12% across screens, use `gemini-3-pro` AnDCG@100
≥ 0.05 as a signal floor, and stratify by coarse phenotype. They do **not** balance
LLM-leading against baseline-leading screens. Every entry carries per-method AnDCG@100,
`gemini_signal_pass`, `best_method` and `llm_lead` so the selection is auditable. The
manifests ship with assaybench as `assayloop-test` and `assayloop-validation`
(`assaybench.data.screen_sets`); regenerate them with:

```bash
uv run assayloop select-public-screens --split test --n-target 20 \
    --per-screen-tsv output/analysis/public_andcg.tsv
```

That writes to `$ASSAYLOOP_OUTPUT/screen_sets/`; copy the result over the shipped
manifest to adopt it. Also shipped: `lopo-*-{train,test}`, the
leave-one-phenotype-out splits, usable directly as `--screen-set lopo-drug-test`.

## Metrics

The paper's columns and the keys they come from in `result.json`:

| Paper | Key | Meaning |
|---|---|---|
| **EF** | `n_hits_vs_random` | hits found ÷ hits a uniform-random policy would find. 1.0 = random. |
| **nAUC** | `hits_auc / hits_auc_best` | area under cumulative-hits-vs-budget, as a fraction of the best possible ordering |
| **FH** | `frac_hits` | fraction of the screen's hits recovered within the budget |
| **SF** | table export | fraction of accepted picks outside the current screen's measured library |
| **%ess** | — | share of picks that are DepMap common-essential genes (a "cheating" proxy: essentials are hits in most screens) |
| **VS** | — | Vendi diversity of the acquired batches |
| **PO** | — | intra-batch pathway overlap vs. random |

`adjusted_ndcg@{10,50,100}` of the current model over the unacquired pool is also recorded.
Sweep aggregates prefix each metric with `mean_` / `min_` / `max_` over screens.

## Models and acquisitions

`--model` (see `make_model` in `src/assayloop/experiment/runner.py`):

| Name | What it does | Needs |
|---|---|---|
| `null` | uniform scores; the "model" when the acquisition does all the work | — |
| `screen_knn` | nearest screens by hit overlap; `prior_only=True` gives the prior-hit baseline | — |
| `knn` | k-NN over PRESAGE/GenePT gene embeddings, distance-weighted vote from acquired neighbours | PRESAGE |
| `rf` | RandomForest on the same embeddings; per-tree variance feeds `ucb` | PRESAGE |
| `bayesian_mlp` (`bmlp`) | Bayesian MLP surrogate | PRESAGE |
| `biobo` | **BioBO** (Li et al., ICLR 2026). Bayesian MLP over concatenated PRESAGE sources (`genept,depmap`). Pairs with `bio_ucb`. | PRESAGE |
| `bpmf` | Bayesian probabilistic matrix factorisation over the screen × gene hit matrix | a trained checkpoint |
| `maml` | MAML meta-learned ranker over BPMF embeddings | checkpoint |
| `amortized_ranker` (`ranker`) | **AssayFormer**. Pairs with `greedy`. | checkpoint, torch |
| `llm_ranker` | in-context LLM ranker: shows the LLM the acquired hits/non-hits, asks for a ranked candidate list | LLM |
| `hypothesis_ranker` | **ICBR-EF**. LLM proposes mechanisms from the description and history, then ranks genes against them. | LLM |
| `llmnn` | **LLMNN** (Gupta et al., arXiv 2509.21403). LLM proposes `n_centers` cluster centres; NN expansion in a PRESAGE space fills the batch. Scores only genes that have an embedding. | LLM + PRESAGE |
| `agent_ranker` | **Haiku-4.5 Agent**. An LLM agent that analyses the training screens with real code execution in a local sandbox. | LLM + Apptainer |

`--acq`:

| Name | What it does | Uses model? |
|---|---|---|
| `random` | uniform from the remaining pool | no |
| `greedy` | top-K by score, with ε-exploration and random tie-breaks | yes |
| `ucb` | `score + β·√uncertainty` | yes (needs uncertainty) |
| `bio_ucb` | BioBO's acquisition: UCB reweighted by a decaying πBO prior (Hvarfner et al., 2022) from a hypergeometric pathway-enrichment test over observed hits. Converges to plain `ucb` as evidence accumulates. Loads MSigDB Hallmark at construction. | yes (needs uncertainty) |
| `llm_single` | **open-vocabulary**: the LLM is *not* shown a candidate list, only the description and AL history, and returns symbols from its own knowledge. Its output is filtered to the unrevealed shared f2 universe afterwards; valid genes outside the current screen library are retained, while repeats and symbols outside that universe become unfilled slots. | no |
| `llm_single_blind` | ablation: same, but hit/non-hit labels are hidden. The LLM still sees which genes were sampled (so it won't repeat them) but cannot learn from outcomes. If this matches `llm_single`, the feedback loop is doing nothing. | no |

The LLM → AssayFormer handoff is not an `--acq`; it is its own command,
[`eval-ranker-handoff`](#the-handoff-assayloop-itself).

In the retrospective full-universe evaluation, a valid f2 gene can be absent from one
screen's measured library. The task keeps the acquisition, marks it with
`metadata.in_library=false`, and supplies a false hit label because that screen has no
positive observation for it; the labeled LLM history therefore groups it with non-hits.
This is the convention used by the published trajectories. Changing it to an explicit
"unmeasured" state would define a different feedback protocol and requires a fresh
evaluation rather than a presentation-only update.

### Choosing the LLM

Every LLM component reads its defaults from `.env`. Override per invocation:

```bash
ASSAYLOOP_LLM_PROVIDER=anthropic ANTHROPIC_MODEL_NAME=claude-opus-4-5 \
    uv run assayloop run --model null --acq llm_single --n-steps 3
```

Or point at one of the committed AssayBench configs in `configs/lm/`, which carry the exact
model id and sampling preset used for each row of the paper's table:

```bash
uv run assayloop run --model null --acq llm_single --screen-set public \
    --lm-config configs/lm/collect-gemini-3.1-pro.yaml
```

Credentials always come from the environment, never from these files. The default vLLM
client uses `temperature=1.0, max_tokens=32000, timeout=300s, enable_thinking=True` to match
AssayBench's `collect-predictions.yaml`. When a thinking trace is truncated and
`message.content` comes back empty, the client falls back to `message.reasoning` /
`reasoning_content` and records the case in the local trace, so a truncated response
degrades to a logged partial rather than a silent zero-gene round.

### The Haiku-4.5 Agent baseline

`agent_ranker` gives the model a Python interpreter over the public training screens, so it
needs somewhere safe to run that code. Build the sandbox locally — it is a stock
`python:3.11-slim` plus numpy/scipy/pandas/scikit-learn and the exported screens, so nothing
is downloaded from us:

```bash
uv run python -m assayloop.scripts.export_screen_dataset   # description, phenotype, hits, non-hits
bash scripts/build_agent_sandbox.sh                        # bakes the export in at /data/screens.json
uv run assayloop run --model agent_ranker --acq greedy --screen-set public --n-steps 10
```

Requires [Apptainer](https://apptainer.org) and `ANTHROPIC_API_KEY`; the model checks for
both at construction and refuses to run without them. A missing sandbox does not quietly
turn the agent into a plain LLM ranker, which would be a different baseline reported under
this one's name. `ASSAYLOOP_AGENT_SANDBOX_SIF` relocates the image.

### Sweeps and parallelism

`assayloop sweep` takes comma-separated `--models`, `--acqs`, `--seeds` and runs their
Cartesian product, one sweep per cell:

```bash
uv run assayloop sweep --models null,knn,rf --acqs random,greedy,ucb --seeds 0,1,2 --n-steps 5
```

`--parallel N` / `-j N` fans screens onto a thread pool on either command. It helps where
the bottleneck releases the GIL — LLM and agent acquisitions (network-bound, near-linear)
and the sklearn-backed `knn`/`rf` once real embeddings are loaded. `null`/`random` are
pure-Python per step and won't speed up; that's expected.

## Reproducing the paper's main table

Table 1 compares every method on the 20 public test screens with a **full-genome candidate
pool** (`--full-genome` for classical methods; the default for open-vocabulary LLMs).
Classical models normally pick from the screen's ~18k library while LLMs pick from the whole
genome, so the pool is unified to the union across screens and every method faces the same
choice set.

That pool is the **f2 universe** — the union kept to genes measured in at least two of the
twenty libraries, 21,147 genes. The one-screen tail is mostly pseudogenes and per-library
assembly artefacts; the unfiltered union is 22,174. One function builds it,
`assayloop.tasks.gene_universe(screens, min_screen_freq=2)`. The open-vocabulary default,
`--full-genome`, and `full_genome_table.py --min-screen-freq 2` all go through it. Pass
`--min-screen-freq 0` for the unfiltered union.

Half the table you regenerate, half you download. The AssayFormer, BPMF, MAML and random
rows are deterministic given a checkpoint, so the table generator just re-runs them — minutes
on a GPU. The LLM and external-baseline rows are not reproducible in the same sense: they are
~400 paid API runs spread over nine vendors, several of whose models have since been retired.
Those we publish as the exact `sweep.json`, `result.json`, and lossless LLM call logs the
paper's numbers were computed from. The call logs matter because current evaluation reparses
the original answer against the shared f2 universe; `result.json` abbreviates long trace text.

```bash
bash scripts/fetch_sweeps.sh                      # -> output/published (or $ASSAYLOOP_PUBLISHED)

# Rebuild the LaTeX table. Reads $ASSAYLOOP_RESULTS, then $ASSAYLOOP_SHARED_PATH, then
# the downloaded bundle -- a sweep you re-ran yourself always wins over the shipped copy.
uv run assayloop figure main-table --min-screen-freq 2
```

Skip the fetch and the table does not quietly print `--` for those rows: an unresolved sweep
raises `MissingConfiguredPath`, lists the directories it searched, and names the fetch script.
A dash in a results table reads as "we measured nothing here", not "you did not download the
data".

The bundle's runs executed behind an internal gateway, so API keys, internal hostnames and
internal filesystem paths were replaced with placeholders before publication. No metric,
acquired-gene list, prompt or model output was altered. `scripts/build_sweep_bundle.py` does
the redaction and refuses to write an archive that still trips its own secret scan;
`MANIFEST.json` inside the bundle maps every sweep to the table row it backs.

`full_genome_table.py` is the source of truth for the row → configuration mapping; its
`METHODS`, `HANDOFF_METHODS` and `JSONL_HANDOFF_METHODS` lists name the exact checkpoint and
sweep tag behind every line of the table. It re-runs anything it cannot find cached. The
individual runs:

```bash
# --- Base LLMs (rows: GLM-5.1, Gemini-3.1-pro, GPT-5.6 Sol) ---
uv run assayloop run --model null --acq llm_single --screen-set public \
    --lm-config configs/lm/collect-GLM-5.1.yaml       # swap in any configs/lm/collect-*.yaml

# --- Classical / heuristic ---
uv run assayloop run --model screen_knn --acq greedy --screen-set public --full-genome \
    --model-param prior_only=true                     # Prior hit baseline
uv run assayloop run --model screen_knn --acq greedy --screen-set public --full-genome
uv run assayloop run --model bpmf --acq greedy --screen-set public --full-genome
uv run assayloop run --model biobo --acq bio_ucb --screen-set public --seed 0 --full-genome
uv run assayloop run --model biobo --acq greedy  --screen-set public --seed 0 --full-genome

# --- Agent, tuned, meta-learning ---
uv run assayloop run --model agent_ranker --acq greedy --screen-set public --n-steps 10
uv run assayloop run --model llmnn --acq greedy --screen-set public --n-steps 10 \
    --model-param n_centers=3 --model-param embedding_source=depmap
uv run assayloop run --model hypothesis_ranker --acq greedy --screen-set public --n-steps 10

# --- AssayFormer and AssayLoop: see the next section ---
```

The **AssayLLM** rows (Qwen3.6-27B + SFT + GRPO) are LLM prediction dumps rather than loop
runs. Point `ASSAYLOOP_LLM_PREDICTIONS` at a directory of prediction JSONLs and the table
generator adapts them into the same shape. To score any externally produced ranking:

```bash
uv run assayloop score-existing predictions.jsonl --out-path output/baselines/scores.json
```

> The paper's trained checkpoints are not committed to this repository. Rows that depend on
> one (AssayFormer, its ablations, MAML, BPMF, and every AssayLoop handoff) need either the
> released checkpoints or a local training run — see below. Nothing falls back to an
> untrained model: `bpmf` raises rather than scoring on a missing `ASSAYLOOP_BPMF_CHECKPOINT`.

### Paper figures

The paper's data figures each have a name under `assayloop figure`. `--list` prints the
table — name, paper reference, what it draws, and which runs have to exist first:

```bash
uv run assayloop figure --list
uv run assayloop figure lopo         # Figure 7
uv run assayloop figure influence    # Figures 12-13
```

Each entry is a thin wrapper over `python -m assayloop.scripts.<module>` and forwards
unknown options through, so the underlying script's own `--help` remains the reference for
its flags. The wrapper only draws: it reads persisted results and never launches a run or a
training job for you, so produce the sweeps and checkpoints in the "needs" column first.
Figures 1–3 are hand-drawn schematics and have no script.

## AssayFormer

An offline-trained, context-conditioned ranker that amortizes the loop: instead of an LLM
call per round, a small transformer encoder scores every gene in one forward pass.

- A **DESC** token = projection of the screen-description text embedding.
- One token per observed gene = `gene_emb ⊕ hit_emb` — the AL context.
- The contextualized DESC output is scored against a **tied, learned gene-embedding table**:
  `scores = desc_out @ E^T + bias`.
- Trained with MSE against `unmasked_relevance_scores` (continuous, for every gene — not
  just hits), on `public_train`, checkpoint-selected on `public_validation`, evaluated on
  `public`.

The learned table is saved as `gene_embeddings.npy` for post-hoc study — it is what the
embedding-drift (`src/assayloop/scripts/plot_embedding_drift.py`) and network-recovery
(`src/assayloop/scripts/eval_gene_networks.py`) figures read.

### Encoders (`--encoder`)

The scoring head is shared, so eval, analysis and `gene_embeddings.npy` behave identically
across encoders:

| `--encoder` | Input | Notes |
|---|---|---|
| `transformer` (default) | embedding tokens `[DESC, obs…]` | small `nn.TransformerEncoder`, CPU-friendly |
| `modernbert-embed` | the same tokens via `inputs_embeds` | pretrained [ModernBERT](https://huggingface.co/blog/modernbert) backbone; needs text embeddings + a GPU |
| `modernbert-text` | tokenized text: description + `Observed hit genes: …` / `Observed non-hit genes: …` | no text-embedding cache needed; CLS/mean-pooled; needs a GPU |

Both `modernbert-*` variants full-fine-tune the backbone with discriminative learning rates
(`--bert-lr 2e-5` backbone, `--lr 3e-4` heads). `--freeze-bert` freezes it for CPU smoke
tests. The rendered text context is right-truncated **hits first**, so non-hits drop before
hits. The first run downloads the backbone (~600 MB) to `HF_HOME`.

<a id="text-embeddings"></a>
### Text embeddings

AssayFormer conditions on a 1536-d OpenAI `text-embedding-3-small` embedding of the screen
description. **Every screen in every public set ships pre-embedded**, in
`src/assayloop/data/text_embeddings/screen_descriptions.npz` (1,356 vectors, 8.8 MB), so
training and evaluation on the public sets make zero API calls and need no key. See that
directory's `PROVENANCE.md`.

`OPENAI_API_KEY` is needed only to embed a description we never saw — one of your own. On a
cache miss with no key the embedder **raises** and names the variable. It does not
substitute a different model: an earlier version fell back to a 384-d sentence-transformer,
which silently invalidates any number computed against a released checkpoint. New vectors go
to `output/rankers/text_emb_cache/cache.npz`, a separate layer consulted first; the shipped
file is never rewritten. `--text-backend local` selects the sentence-transformer explicitly
as an ablation — it is incompatible with the released checkpoints and nothing picks it
implicitly.

```bash
uv run assayloop warm-text-cache      # pre-populate for every screen set, once
```

### BPMF: the gene-embedding init

The gene-embedding table is initialised from a Bayesian probabilistic matrix factorisation of
the `public_train` screen × gene hit matrix — the same factorisation that backs the `bpmf`
baseline row. No checkpoint ships, so fit it once:

```bash
uv run assayloop train-bpmf --target-set public_train --K 10
```

The paper's settings are the defaults: `K=10`, 2000 Gibbs iterations, 1000 discarded as
burn-in, thinned by 2, gene factors set to the posterior mean of the retained samples. The
GPU sampler is the default and its `--K` takes a comma list, so one call can sweep the latent
dimension; `--cpu` selects the reference CPU implementation instead.

Each fit writes `bpmf_result.pkl` into its own tagged directory under
`$ASSAYLOOP_OUTPUT/bpmf/`, so set `ASSAYLOOP_BPMF_CHECKPOINT` to the pkl you want — the
default (`output/bpmf/bpmf_result.pkl`) is the un-tagged path and will not exist until you
point it somewhere or copy a fit there.

### Train, evaluate, analyze

```bash
# Full training on public_train, selection on public_validation, then eval on public test.
uv run assayloop train-ranker

# Common knobs.
uv run assayloop train-ranker \
    --epochs 30 --batch-size 32 --lr 1e-3 \
    --d-model 256 --d-gene 256 --num-layers 2 --nhead 4 \
    --dagger-rounds 1 --dagger-screens 64 \
    --val-al-screens 20 --al-batch-size 100 --al-n-steps 10 \
    --device auto

# Evaluate a checkpoint in the AL loop, persisted as a sweep like any other run.
uv run assayloop eval-ranker --checkpoint <checkpoint-dir>
```

`eval-ranker` runs two diagnostics by default, both aimed at the question *is this
model actually using the AL context, or is it doing static retrieval?* —
`--ablate-context` re-scores with the observation tokens removed, and `--context-value`
measures the marginal value of each additional observed round. `--no-ablate-context` /
`--no-context-value` skip them if you only want the AL sweep.

Training and validation metrics stream to Weights & Biases. `--wandb-entity` defaults to
your account default (or `$WANDB_ENTITY`); `--wandb-mode offline` or `disabled` skips the
network entirely. Note that a failed `wandb.init` is logged as a warning and training
continues unlogged, so set the entity deliberately if you care about the record.

### RL fine-tuning

`train-ranker-rl` GRPO-fine-tunes a warm-started ranker to optimize the AL objective
directly (the `+ GRPO` rows of the table — this is what turns AssayFormer from EF 3.83 to
4.83):

```bash
uv run assayloop train-ranker-rl --init-checkpoint <supervised-checkpoint-dir>
```

Multi-GPU data-parallel RL is a flag, not a separate launcher: `--gpus 4` re-execs the
command under `torchrun` itself.

### The handoff: AssayLoop itself

```bash
uv run assayloop eval-ranker-handoff \
    --checkpoint <rl-checkpoint-dir> \
    --ckpt-file model_last.pt \
    --warm-dir <dir-of-llm-warm-start-runs> \
    --warm-prefix sweep-<id>- \
    --n 3
```

`--ckpt-file` matters for reproduction. A ranker directory holds two sets of weights:
`model.pt` is the best-on-validation epoch and `model_last.pt` is the final epoch, and
for the released RL checkpoint they are genuinely different models (epoch 92 vs epoch 99).
**Every number in the paper comes from `model_last.pt`** — `full_genome_table.py` sets
`HANDOFF_CKPT_FILE = "model_last.pt"` and the figure scripts follow it. The CLI still
defaults to `model.pt`, which is the right default for a checkpoint you just trained
yourself, so pass `--ckpt-file model_last.pt` explicitly when you mean to reproduce a
published row. The same flag exists on `eval-ranker`.

The LLM's first `--n` rounds are replayed from an existing `llm_single` sweep (matched by
`--warm-prefix`), then AssayFormer takes over with those observations as its context. The
paper uses `n=2` for GLM-5.1 and AssayLLM and `n=3` for Gemini-3.1-Pro and GPT-5.6 Sol;
`full_genome_table.py`'s `HANDOFF_METHODS` records which trace sweep pairs with which row.
`--warm-start-n` / `--warm-start-prob` / `--warm-start-glm-train` on `train-ranker` train the
ranker to expect a warm start in the first place, and `train-ranker-rl` carries the matching
`--handoff-*` flags for RL under the handoff objective.

## Outputs and tracing

```
output/
├── runs/<run_id>/
│   ├── result.json        # full StepRecord history + final metrics
│   ├── config.json        # the RunConfig used
│   └── llm_calls.jsonl    # every LLM call made during this run
└── sweeps/<sweep_id>/
    └── sweep.json         # per-screen rows + aggregate
```

Every LLM call inside a run is appended to `llm_calls.jsonl`: the full prompt, response text,
any reasoning text, finish_reason, token usage and latency. Nothing is uploaded anywhere —
it is a plain file you can read, diff and replay offline, and it is what you look at when a
run produced fewer genes than it was asked for.

```bash
uv run assayloop trace <run_id>           # last 5 calls, truncated
uv run assayloop trace <run_id> -n 20 --full
```

Runs and sweeps are ordinary directories, so housekeeping is `rm -rf` on the ones you no
longer want; nothing outside `output/` refers to them.

`collect-dataset` builds the teacher-trace dataset behind the AssayLLM rows: it replays LLM
acquisitions and writes one SFT record per round. It resumes, so re-running the same command
after an interruption fills in only the `(screen, trace)` pairs that are still missing.

## Environment variables

Read from `.env` (via `python-dotenv`) or the environment. Every path is resolved in exactly
one place, `src/assayloop/config.py`, and each one documents its default and what raises when
it is missing.

| Variable | Default | Notes |
|---|---|---|
| `ASSAYLOOP_LLM_PROVIDER` | `vllm` | `vllm` \| `anthropic` \| `dspy` |
| `VLLM_BASE_URL` / `VLLM_MODEL_NAME` / `VLLM_API_KEY` | `http://localhost:8000/v1`, `glm-5`, `not-needed` | OpenAI-compatible vLLM endpoint. `VLLM_TOP_P`, `VLLM_TOP_K`, `VLLM_MIN_P`, `VLLM_REPETITION_PENALTY`, `VLLM_PRESENCE_PENALTY` override the sampling preset. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL_NAME` | — | the `anthropic` provider, and `agent_ranker` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | only to embed a description outside the [shipped cache](#text-embeddings) |
| `ASSAYLOOP_PRESAGE_CACHE` | `src/assayloop/data/presage_cache` | unpacked PRESAGE cache |
| `ASSAYLOOP_GENE_SETS` / `ASSAYLOOP_GENE_SETS_SOURCE` | bundled dir, `c5.go.bp.v2023.2.Hs.symbols` | MSigDB gene sets |
| `ASSAYLOOP_GROUND_TRUTH` | — | STRING/CORUM/SIGNOR tables. No default; scripts raise when unset. |
| `ASSAYLOOP_BPMF_CHECKPOINT` | `output/bpmf/bpmf_result.pkl` | trained BPMF factorisation. No checkpoint ships; `bpmf` raises rather than scoring on an untrained one. |
| `ASSAYLOOP_OUTPUT` | `output` | everything this repo *writes*: runs, sweeps, and the `analysis/` directory the figure scripts read caches from and write figures back into. `ASSAYLOOP_RESULTS`, `ASSAYLOOP_SHARED_PATH` and `ASSAYLOOP_PUBLISHED` default relative to it, so pointing this one variable at a downloaded bundle points the read side at it too. Set it to a scratch tree when rebuilding figures, so `savefig` does not write into your results. |
| `ASSAYLOOP_RESULTS` | `output` | where the table and figure scripts look for finished runs |
| `ASSAYLOOP_SHARED_PATH` | `output/shared` | optional second results location, searched after the local one |
| `ASSAYLOOP_PUBLISHED` | `output/published` | where `scripts/fetch_sweeps.sh` unpacks the published bundle; searched last, so your own re-run wins |
| `ASSAYLOOP_BUNDLE_URL` | GitHub release `data-v1` | override the bundle download host (mirror or local copy) |
| `ASSAYLOOP_LLM_PREDICTIONS` | `output/llm_predictions` | AssayLLM prediction JSONLs read by the results tables |
| `ASSAYLOOP_AGENT_SANDBOX_SIF` | built in place | relocates the Apptainer image |
| `SCRATCH_PATH` | `scratch` | scratch space for training jobs |
| `ASSAYLOOP_LOG` | `INFO` | log level |
| `WANDB_ENTITY` / `WANDB_RUN_GROUP` | — | W&B account default is used when unset |

## Extending the framework

Implement any one of the four abstractions and pass it to `SequentialLoop`:

```python
from assayloop import (
    AcquisitionFunction, Metric, Model, ModelPrediction,
    Observation, SequentialLoop, StepRecord, Task,
)

class MyModel(Model):
    def predict(self, observations, candidates, task_context=None):
        return ModelPrediction(scores={c: my_score(c, observations) for c in candidates})

class MyAcq(AcquisitionFunction):
    def suggest(self, history, candidates, batch_size,
                model_prediction=None, task_context=None):
        ...
        return picked_batch

loop = SequentialLoop(
    task=my_task, model=MyModel(), acquisition=MyAcq(),
    metrics=[my_metric], batch_size=100,
)
result = loop.run(n_steps=10)
```

`result.history` is a list of
`StepRecord(step, acquired_batch, new_observations, model_prediction, metrics, acquisition_trace)`.
To expose a new component to the CLI, add a branch to `make_model` / `make_acquisition` in
`src/assayloop/experiment/runner.py`.

`src/assayloop/examples/assaybench/` is a complete worked example of a *different* task
shape — selecting which historical screens to put in a target screen's few-shot context,
rather than which genes to test:

```bash
uv run python -m assayloop.examples.assaybench.run --acquisition random --n-steps 5
```

Architecture notes for the internals live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Tests: `uv run pytest tests/`.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Genentech, Inc.

The companion [`assaybench`](https://pypi.org/project/assaybench/) package (screen corpus
loader and the shared EF / nAUC / shortfall / %essential metrics) is MIT under the same
copyright.

## Third-party data

No third-party dataset is redistributed by this repository except where noted below. The
fetch scripts (`scripts/fetch_*.sh`) download each source from its own site, which is also
how you accept its terms; each script's header records the pinned version and license. If a
source is missing, the code that needs it raises and names it — it does not fall back to a
partial or substituted input.

| Source | Used for | Terms |
|---|---|---|
| **MSigDB Hallmark** (`h.all.v2023.2.Hs.symbols.gmt`, downloaded — `assaybench download msigdb-hallmark` or `scripts/fetch_gene_sets.sh`) | BioBO's enrichment prior; pathway-diversity metric | CC BY 4.0. Attribution: Broad Institute, Inc., Massachusetts Institute of Technology, and Regents of the University of California. |
| **MSigDB C5 GO:BP / C2 CP** (fetched) | pathway diversity, sunburst figure | Broad terms. C2 CP is **not** redistributable as a single file — it mixes KEGG legacy and BioCarta sets held under qualified permission with CC BY-SA 4.0 KEGG_MEDICUS sets, which is why you download it yourself. |
| **Reactome** pathway hierarchy + human interactors (fetched) | pathway hierarchy, co-pathway counts | CC BY 4.0. `reactome_two_level.json` is a small derived hierarchy and is committed. |
| **STRING v11.5, CORUM 4.1, SIGNOR 3.0** (fetched) | [network recovery, Figure 9](#interaction-ground-truth) | All three CC BY 4.0 at these pinned versions. |
| **DepMap** common-essential genes | %essential | Bundled with `assaybench`; CC BY 4.0, Broad Institute. |
| **BioGRID ORCS** screens | the screen corpus itself | Loaded from the `Genentech/assaybench` Hugging Face dataset; see that dataset card. |
| **PRESAGE** gene embeddings (fetched, 3.4 GB) | `knn`, `rf`, `biobo`, `llmnn`, Vendi diversity | See the PRESAGE release for its terms; nothing is redistributed here. |
| **Screen-description embeddings** (`src/assayloop/data/text_embeddings/`, committed) | the DESC token | Derived from public screen descriptions via OpenAI `text-embedding-3-small`; provenance in that directory's `PROVENANCE.md`. |

## Citation

If you use this code, please cite the AssayLoop paper. <!-- TODO: replace with the final
BibTeX entry once the paper has a citable reference. -->
