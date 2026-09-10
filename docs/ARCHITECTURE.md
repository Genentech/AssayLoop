# assayloop architecture

This is the design document for the framework. The [README](../README.md) is the
user-facing guide — how to install it, what the commands do, how to reproduce the
paper. This file explains *why the pieces are shaped the way they are*, which is
what you need if you are adding a model, an acquisition, or a task.

## Why this exists

We wanted to prototype sequential-design strategies on real CRISPR screens and get
an honest, comparable signal across very different kinds of "ranker" — random
sampling, classical embedding-similarity models, matrix factorisation, in-context
LLM rankers, a trained transformer, and a tool-using agent — without rebuilding the
harness for each new idea.

The task is:

> Given an unannotated CRISPR screen of ~18–22k genes, repeatedly choose a batch of
> 100 genes to assay (reveal their label), and find as many hits as possible as
> early as possible. Score by the trapezoidal area under the cumulative-hits-found
> vs. fraction-of-library-acquired curve.

We score with two normalized variants of `hits_auc` because the raw value isn't
comparable across screens of very different sizes.

**Primary: `hits_auc_vs_random` (and `n_hits_vs_random`).** "How many times better
than a uniform-random pick was this run?"

```
hits_auc_vs_random = hits_auc / hits_auc_random
hits_auc_random    = x² / 2                          where x = frac_acquired
n_eff              = N_L + N̄_G + N_miss
h_rand             = total_hits / library_size
n_hits_vs_random   = n_hits / (n_eff · h_rand)
```

Random acquisition has expected curve `y(x) = x`, so its expected AUC up to
`x = frac_acquired` is `x²/2`. The ratio is **1.0 = no better than random,
>1.0 = above random, <1.0 = below random**. The best achievable ratio (perfect
ordering at small `x`) is roughly `L/H`. It has the most intuitive units — a number
of "× times better" — which is why it is the paper's EF column. In the endpoint
EF, `N_L` counts in-library acquisitions, `N̄_G` counts hallucinated acquisitions,
and `N_miss` counts unfilled acquisition slots. Valid genes outside the screen
library are forgiven. For nAUC, unfilled slots do not advance `x`; an
under-supplying policy ends its curve earlier.

**Secondary: `hits_auc_normalized`.** "What fraction of the *best possible* ordering
did we attain?"

```
hits_auc_normalized = hits_auc / hits_auc_best
```

where `hits_auc_best` is the AUC of the optimal-ordering curve clipped to the
current `frac_acquired`. The optimal curve climbs at slope `L/H` until it has
captured every hit, then is flat at 1. So **perfect ordering → 1.0**, **random
acquisition → H/L in expectation**, **all-misses-first → 0.0**. The exact formula is
in `metrics/hits_auc.py::best_possible_auc`. This is the paper's nAUC column: the
right framing for "how close to perfect could the agent get", where
`hits_auc_vs_random` is the right framing for "how much did it help over baseline".

## The loop

```
    SequentialLoop on one screen:
        candidates ─► model.predict ─► acq.suggest ─►
        task.reveal  ─► metrics ─► history ─► next step

    run_sweep:  fans the loop out across N screens (optionally parallel)
```

## Modular core (`assaybench.core`)

The whole point is that you can swap any one of these without touching the others.
Each is a small abstract base class with one or two methods.

They are not defined in this repo. The ABCs, the shared dataclasses and
`SequentialLoop` live in the `assaybench` PyPI package, so that a third party can
implement against the benchmark without inheriting this repo's model zoo, LLM
clients and paper scripts. `assayloop` depends on it and re-exports the names
(`from assayloop import Task, Model, ...`). The table below lists this repo's
implementations of each.

| ABC                    | Method                          | Implementations today |
|------------------------|---------------------------------|-----------------------|
| `Task`                 | `candidates() / reveal() / context() / ground_truth() / task_id()` | `AssayBenchGeneBatchTask` |
| `Model`                | `predict(observations, candidates, ctx) → ModelPrediction` | `NullModel`, `KNNGeneEmbedding`, `RFGeneEmbedding`, `BayesianMLPModel`, `ScreenKNNModel`, `BPMFModel`, `AmortizedRankerModel`, `MAMLRankerModel`, `LLMInContextRanker`, `AgentRankerModel`, `HypothesisRankerModel`, `LLMNNModel` |
| `AcquisitionFunction`  | `suggest(history, candidates, batch_size, pred, ctx) → batch` | `RandomAcquisition`, `GreedyFromModel`, `UCBFromModel`, `BioUCBFromModel`, `LLMSingleAcquisition`, `GlmHandoffAcquisition` |
| `Metric`               | `score(observations, prediction, ground_truth, remaining) → dict` | `HitsAUC`, `AnDCGAtK`, `BatchHits`, `BatchDiversity` |

The deliberate separation of **`Model`** (produces a ranking / scoring) from
**`AcquisitionFunction`** (picks the next batch given that ranking plus history) is
the most important design decision. It lets us pair, say, a Bayesian-MLP surrogate
with a UCB acquisition, or pair a "null" model with a single-shot LLM acquisition
that does its own ranking internally. Without that split we'd have a combinatorial
explosion of one-off classes.

`SequentialLoop` (`assaybench/core/loop.py`) orchestrates them. Per step it asks the
task for candidates, asks the model for predictions, asks the acquisition for a batch,
asks the task to reveal labels, runs every metric, and appends a `StepRecord` to
history. It does no I/O of its own; `experiment/runner.py` passes it this repo's
`run_trace_scope` as `trace_scope=`, which is what puts every LLM call the model or
acquisition makes into `output/runs/<run_id>/llm_calls.jsonl`.

### Shortfall is never padded

An acquisition that returns fewer than `batch_size` genes is recorded as a
shortfall, not topped up with random picks — padding would credit a method for
genes it never chose. `RunConfig.max_shortfall_frac` (default `1.0`) aborts a
degenerate screen rather than reporting a diluted number, and `shortfall_frac` is
carried into `result.json` so the paper's SF column is auditable.

## Where everything came from

| Source | How we use it |
|---|---|
| `assaybench` (PyPI) | The screen corpus and ranking metrics: `AssayBenchDataset`, `RankingMetrics`, and the prompt loaders. Screens come from the public `Genentech/assaybench` dataset on the Hub. |
| PRESAGE | Pre-computed multi-source gene embeddings, downloaded separately (see the README). Optional: only the embedding-based models need it. |

Every screen set the paper reports on is public, and assayloop installs from PyPI
alone.

## Concrete wiring

`experiment/runner.py::make_model` and `make_acquisition` are the two name → object
factories. Adding a model means adding one branch to `make_model`; there is no
plugin registry to register with and no config schema to extend.

### Models

- **`null`** — uniform scores; lets acquisitions that do their own selection
  (random, single-call LLM) be evaluated cleanly.
- **`knn`** — kNN over a PRESAGE gene-embedding space (GenePT by default). Mouse
  genes not covered by PRESAGE go through MGI→HGNC ortholog mapping.
- **`rf`** — random-forest classifier over the same embeddings, `predict_proba` as
  the score and per-tree variance as uncertainty (so UCB can consume it).
- **`bayesian_mlp` / `bmlp`** — Bayesian MLP surrogate with calibrated uncertainty.
- **`biobo`** — BioBO (Li et al., ICLR 2026): the same Bayesian MLP over
  *concatenated* PRESAGE sources (`genept,depmap` by default). Pairs with `bio_ucb`.
- **`screen_knn`** — nearest-neighbour retrieval over *screens* rather than genes;
  `prior_only=true` degenerates it to the "prior hit frequency" baseline.
- **`bpmf`** — Bayesian probabilistic matrix factorisation of the screen × gene hit
  matrix. Needs a checkpoint; raises rather than scoring untrained.
- **`amortized_ranker` / `ranker`** — AssayFormer. See below.
- **`maml`** — MAML meta-learned ranker over BPMF embeddings.
- **`agent_ranker`** — the Haiku-4.5 Agent: an LLM agent that analyses the public
  training screens with real code execution, sandboxed in an Apptainer container.
  Raises at construction if the container is missing rather than silently becoming a
  plain ranker.
- **`hypothesis_ranker`** — ICBR-EF: the LLM proposes mechanisms, then ranks genes
  against them.
- **`llmnn`** — LLMNN (Gupta et al., arXiv 2509.21403): the LLM proposes cluster
  centers and nearest-neighbour expansion in a PRESAGE space fills the batch.
- **`llm_ranker` / `llm_incontext`** — gives the LLM the current observations plus
  the candidate pool and asks for a ranked list; parsed by `llm/parse_genes.py`,
  which handles structured JSON, XML-like tags, and plain text.

### Acquisitions

- **`random`** — seeded random sample. The baseline every ratio is measured against.
- **`greedy`** — top-k by `ModelPrediction.scores`.
- **`ucb`** — `score + β · uncertainty`; requires a model that reports uncertainty.
- **`bio_ucb`** — the BioBO acquisition: UCB augmented with an enrichment-analysis
  prior over the candidate pool.
- **`llm_single`** — AssayBench-style open-vocabulary acquisition. The LLM is **not**
  shown a candidate list; it produces gene symbols from its own knowledge of the
  genome (matching AssayBench's `biogrid_ranking_prompt`). Output is parsed, deduped,
  then filtered to genes still in the unrevealed pool.
- **`llm_single_blind`** — the label ablation: the same loop with the per-round hit
  labels stripped from the history, so the model sees what it asked for but not what
  came back.
- **`GlmHandoffAcquisition`** — replays an LLM's recorded first *n* rounds, then hands
  off to the ranker. It is constructed directly by `eval-ranker-handoff` rather than
  resolved by name, because it needs the warm-start traces as an argument.

### LLM backend

`llm/client.py` is a unified factory returning one interface across three providers:
**vLLM** (a locally served open-weights model), **Anthropic**, and **DSPy** (for
programmatic prompting with signatures/optimizers). `LLMClientConfig` is what every
model and acquisition that uses an LLM takes as its config argument, so switching
backends is one field. Per-model YAML lives in `configs/lm/`.

Every LLM call inside a run is appended to `output/runs/<id>/llm_calls.jsonl` —
prompt, response, reasoning text, finish_reason, tokens, latency. Nothing is
uploaded anywhere.

### Gene embeddings

`data/gene_embeddings/presage.py` loads the pre-computed multi-source PRESAGE cache.
If the cache is absent it raises `MissingPresageCache`; if a requested source within
it is absent, `MissingPresageSource`. `OnehotGeneEmbedding` and
`RandomGeneEmbedding` exist for unit tests and smoke tests, and **nothing selects
them automatically** — a missing cache is an error, not a degraded run, because
metrics computed against orthogonal one-hot vectors are meaningless but look fine.

### Screen sets

The canonical target is the **20 curated public screens** in the
`assayloop-test` manifest, with a matching 20-screen validation set in
`assayloop-validation`. Both are drawn from the public AssayBench `yearfold0`
splits. The manifests themselves ship with assaybench, in
`assaybench.data.screen_sets` — which screens a reported EF was averaged over
is part of the benchmark definition, not of this repo's plumbing, and
`load_manifest(name).summary()` states the split so two results are not
silently computed on different subsets. `tasks/screen_sets.py` is the adapter
from a manifest to loaded `ScreenRecord`s and exposes the public names
(`paper_test`, `paper_validation`, `train`, `validation`, `test`). The older
`public*` names remain aliases for configs and checkpoints already written
against them. Override with `--screen-set paper_validation`, `--screen-set
lopo-drug-test` (any shipped manifest name),
`--screen-set /path/to.yaml`, or explicit `--screen dataset_name,...`.

`src/assayloop/scripts/select_default_public_screens.py` regenerates them into
`$ASSAYLOOP_OUTPUT/screen_sets/` (adopting a regenerated set means copying it over
the shipped one; the script will not write into an installed package): it loads the requested
split from `Genentech/assaybench`, scores each screen with AnDCG@100 from
`gemini-3-pro` and from three trained / retrieval baselines (Oracle kNN, Embedding
kNN, `phenotype-hit-freq`), and picks 20 screens that

- are **genome-wide** libraries (`num_genes ∈ [18k, 22k]`) so every screen's
  `frac_acquired` ends within ~12% of every other, keeping the cross-screen mean
  curve honest end-to-end. Together with `num_hits ≥ 50` and `hit_rate ≤ 15%` this
  keeps the frontier non-trivial without saturating in 2 batches,
- have `gemini-3-pro` AnDCG@100 ≥ 0.05. Baseline scores are audit metrics only
  and are not eligibility floors,
- are stratified across the 5 coarse phenotypes with a soft cap of 6 per phenotype,
  relaxed only when needed to reach 20 screens, and
- do not balance LLM-leading and baseline-leading winners.

Within each bucket we use a quality-weighted greedy max-min on TF-IDF over phenotype
+ condition + cell-line text, to avoid picking three near-duplicate HT-29
drug-response screens. The committed YAML persists the per-method AnDCG@100 scores,
`gemini_signal_pass`, `best_method`, and `llm_lead` for each screen, so the rationale
is auditable.

The Gemini floor is a judgement call and worth stating plainly: screens where the
strongest available prior-knowledge signal gets AnDCG@100 < 0.05 are dropped, because
an active-learning curve on them measures noise rather than method quality. It is a
*selection* criterion, applied before any method is run and identically to all of
them, but it does mean the reported set is conditioned on one model's scores. Each
manifest entry keeps `gemini_signal_pass` and the per-method AnDCG@100 so the
consequence is inspectable. The five leave-one-phenotype-out splits are the
`lopo-*-{train,test}` manifests alongside them.

## What runs at the top: `experiment/runner.py`

`RunConfig` is the single source of truth for one loop invocation: which task, which
model + acquisition and their overrides, batch size, `n_steps`, warm-start size,
seed, which metrics to compute, `max_shortfall_frac`, and `parallel` for fanning out
screens. It's a plain dataclass so it serialises cleanly into
`output/runs/<id>/config.json`.

`run_one_screen` does exactly one loop: builds task/model/acq/metrics from
`RunConfig`, runs `SequentialLoop.run`, writes `result.json` and `config.json` under
`output/runs/<run_id>/`.

`run_sweep` is the per-sweep driver:

1. Resolve screens (default set vs. all vs. explicit list).
2. For each screen, build a deterministic
   `run_id = "<sweep_id>-<idx>-<truncated_dataset_name>"`.
3. Execute serially or via `ThreadPoolExecutor(max_workers=parallel)`.
4. Aggregate per-screen final metrics (`mean_`/`min_`/`max_` of each) and persist
   `output/sweeps/<sweep_id>/sweep.json`.

Parallel execution shines for **I/O-bound** runs (LLM and agent — near-linear 8×);
classical models that hit pure-Python paths stay GIL-bound, which is fine and
expected.

## AssayFormer (`amortized/`)

The trained ranker that amortizes the loop: instead of an LLM call per round, one
transformer forward pass scores every gene.

| Module | What it holds |
|---|---|
| `model.py` | the encoder, the tied gene-embedding table, the scoring head |
| `data.py` | training-example construction from screen histories |
| `gene_factors.py` | embedding initialisation, including the BPMF init |
| `text_embed.py` | screen-description embeddings and their cache |
| `train.py` | supervised training + DAgger rounds |
| `rl.py`, `rl_ddp.py` | GRPO fine-tuning, single- and multi-GPU |
| `warmstart.py` | replaying recorded LLM rounds as context — the handoff |
| `context_value.py` | the marginal-value-of-context diagnostic |
| `analysis/` | embedding-geometry analyses (learned table vs. PRESAGE sources) |

## CLI surface

`cli.py` is a thin Typer layer: every command builds a `RunConfig` (or a training
config) and calls into the library. Nothing in `experiment/`, `amortized/`, or
`models/` imports the CLI, so the whole framework is usable as a library.

```
uv run assayloop run                   # single (model, acq) over a screen set
uv run assayloop sweep                 # cross-product of --models, --acqs, --seeds
uv run assayloop train-bpmf            # the gene-embedding init
uv run assayloop train-ranker          # supervised AssayFormer
uv run assayloop train-ranker-rl       # GRPO fine-tuning
uv run assayloop eval-ranker-handoff   # AssayLoop itself
uv run assayloop figure --list         # regenerate a paper figure or table
```

`--parallel N` (or `-j N`) on `run` and `sweep` fans screens out onto a thread pool.
Run `assayloop --help` for the full list.

## What this gives us

- **Plug in a new model** ⇒ subclass `Model`, add a branch to `make_model`. No other
  file changes.
- **Plug in a new acquisition** ⇒ subclass `AcquisitionFunction`, add a branch to
  `make_acquisition`. Done.
- **Plug in a new task** ⇒ subclass `Task`, write a `make_*` helper. The sweep driver
  works unchanged because everything keys off `run_id` and `final_metrics`.
- **Run more screens in parallel** ⇒ `--parallel N`.

## Directory map

```
src/assayloop/
│                          # (the ABCs and SequentialLoop are in `assaybench.core`)
├── tasks/                 # gene_batch.py (the Task) + screen_sets.py (which screens)
├── models/                # null, knn, rf, bayesian_mlp, screen_knn, bpmf,
│                          #   amortized_ranker, maml, llm_incontext, agent_ranker,
│                          #   hypothesis_ranker, llm_nn
├── acquisitions/          # random, greedy, ucb, bio_ucb, llm_single, glm_handoff
├── metrics/               # hits_auc, andcg_at_k, batch_hits, batch_diversity
├── llm/                   # unified client factory + gene-list parsing
├── data/                  # screen-set YAMLs, text_embeddings/, gene_embeddings/
├── amortized/             # AssayFormer: model, training, RL, warm start, analysis
├── experiment/            # runner.py (run_one_screen, run_sweep, parallel)
├── baselines/             # scoring for externally produced predictions
├── examples/assaybench/   # "Option 1": a second Task/Model/Acq/Metric quartet
├── scripts/               # paper figures, tables, and offline analyses
└── cli.py                 # typer entrypoint
```
