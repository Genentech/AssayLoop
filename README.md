# AssayLoop

Code for **"Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens."**

[:globe_with_meridians: Project site](https://genentech.github.io/assayloop) | [:hugs: AssayFormer checkpoint](https://huggingface.co/collections/Genentech/assaybench) | [:page_with_curl: Full reference](README_DETAILED.md) | [![PyPI](https://img.shields.io/pypi/v/assaybench)](https://pypi.org/project/assaybench/)

![The task: a screen is a library of genes, a phenotype and a hit set. Each round, a method
sees the phenotype and everything it has already assayed, and chooses the next hundred
genes.](docs/assets/figures/figure1.png)

A CRISPR screen has ~20,000 candidate genes and a budget for testing a few hundred at a
time. AssayLoop chooses which genes to test next, round after round, so hits show up early.
An LLM picks the first two or three rounds from its biological priors, then **AssayFormer**,
a transformer trained on 1,349 historical screens, takes over with those observations as its
context. On 20 held-out screens it recovers **27.7% of hits after assaying 5% of the
library**, ahead of every baseline we evaluated.

## Install

```bash
git clone https://github.com/genentech/assayloop
cd assayloop
uv sync --extra torch     # drop --extra torch for the CPU-only baselines and analysis
```

Python 3.11+. Screens download from Hugging Face on first use. No API key is needed to run
AssayFormer or to reproduce its numbers; you need one only for the LLM baselines, and only
for the provider you call (set `ASSAYLOOP_LLM_PROVIDER` and that provider's key in `.env`).

## Quickstart

```bash
uv run assayloop run --model null --acq random --n-steps 10   # random baseline, 20 test screens
uv run assayloop info                                         # what resolved: provider, caches, paths
```

## AssayFormer

Load the released checkpoint and score it on a held-out screen:

```python
from huggingface_hub import snapshot_download
from assaybench import SequentialLoop, enrichment_factor
from assayloop.acquisitions.greedy_from_model import GreedyFromModel
from assayloop.models.amortized_ranker import AmortizedRankerModel
from assayloop.tasks import load_screens, make_task

ckpt = snapshot_download("Genentech/assayformer")
model = AmortizedRankerModel(checkpoint=ckpt, ckpt_file="model_last.pt")

screen = load_screens(target_set="public")[0]
run = SequentialLoop(make_task(screen), model, GreedyFromModel(),
                     metrics=[], batch_size=100).run(n_steps=10)

picked = [g for step in run.history for g in step.acquired_batch]
hits = [g for g, h in zip(screen.genes, screen.hits) if h]
print(enrichment_factor(picked, screen.genes, hits, budget=1000))   # 6.96
```

Training your own:

```bash
uv run assayloop train-bpmf --target-set public_train --K 10        # gene-embedding init
uv run assayloop train-ranker                                       # supervised
uv run assayloop train-ranker-rl --init-checkpoint <ckpt-dir>       # + GRPO: EF 3.83 -> 4.83
uv run assayloop eval-ranker --checkpoint <ckpt-dir> --ckpt-file model_last.pt
```

## AssayLoop: the handoff

Collect the LLM warm start, then hand off to the ranker:

```bash
uv run assayloop run --model null --acq llm_single --screen-set public --full-genome \
    --lm-config configs/lm/collect-gemini-3.1-pro.yaml

uv run assayloop eval-ranker-handoff \
    --checkpoint <rl-ckpt-dir> --ckpt-file model_last.pt \
    --warm-dir output/runs --warm-prefix sweep-<id>- --n 3
```

The first `--n` rounds are replayed from that sweep, then AssayFormer continues with them as
context.

## Screen sets

```python
from assayloop.tasks import load_screens

load_screens(target_set="public")              # the paper's curated 20-screen test set
load_screens(target_set="public_test")         # the full 334-screen test fold
load_screens(dataset_names=["U_1733_merged"])  # exact screens
```

| `--screen-set` | What |
|---|---|
| `public` (default) | the curated **20-screen test set** the paper reports |
| `public_validation` | the curated 20-screen validation set (checkpoint selection) |
| `public_train` | the full 1,349-screen training fold |
| `public_val` | the full 218-screen validation fold |
| `public_test` | the full 334-screen test fold |
| `lopo-*-{train,test}` | the leave-one-phenotype-out splits |
| `/path/to.yaml` | your own list |

```bash
# Same random baseline as the Quickstart, on the validation set instead of the test set.
uv run assayloop run --model null --acq random --screen-set public_validation
```

## Metrics

Every metric takes the genes a method picked and the screen's ground truth, nothing else:

```python
import numpy as np
from assaybench import (adjusted_nauc, enrichment_factor, fraction_of_hits,
                        percent_essential, shortfall)
from assayloop.tasks import load_screens

screen = load_screens(target_set="public")[0]
hits = [g for g, h in zip(screen.genes, screen.hits) if h]

picked = list(np.random.default_rng(0).choice(screen.genes, 1000, replace=False))
rounds = [picked[i * 100:(i + 1) * 100] for i in range(10)]   # your method's ten batches

enrichment_factor(picked, screen.genes, hits, budget=1000)  # EF     0.979  vs. uniform random
adjusted_nauc(rounds, screen.genes, hits)                   # nAUC   0.031  how early hits came in
fraction_of_hits(picked, screen.genes, hits)                # FH     0.053  share of the screen's hits
shortfall(picked, screen.genes)                             # SF     0.0    budget slots left unfilled
percent_essential(picked, screen.genes, hits)               # %ess   0.444  DepMap common-essential
```

A uniform-random policy scores EF 1.0. Pass `universe=` to score against the full-genome
pool the paper's table uses rather than one screen's library.

## Your own method

Implement `Model` (score the candidates) or `AcquisitionFunction` (choose from the scores)
and hand it to the same loop:

```python
from assaybench import Model, ModelPrediction, SequentialLoop
from assayloop.acquisitions.greedy_from_model import GreedyFromModel
from assayloop.tasks import load_screens, make_task

class MyRanker(Model):
    """Score a gene 1.0 if a hit found so far shares its first three letters.

    Gene symbols are loosely paralogous by prefix -- NDUFA1/NDUFA2, RPL3/RPL4 --
    so this is a crude "more of whatever is working". It is here to show the
    interface, not because it is a good policy.
    """

    def predict(self, observations, candidates, task_context=None):
        # observations is what has been assayed so far. o.candidate is the gene
        # symbol; o.label is {"hit": bool, "relevance_score": float}. candidates
        # is what is left to score.
        hit_prefixes = {o.candidate[:3] for o in observations if o.label["hit"]}
        return ModelPrediction(
            scores={g: float(g[:3] in hit_prefixes) for g in candidates})

screen = load_screens(target_set="public")[0]
run = SequentialLoop(make_task(screen), MyRanker(), GreedyFromModel(),
                     metrics=[], batch_size=100).run(n_steps=10)

for step in run.history:            # one record per round
    found = sum(1 for o in step.new_observations if o.label["hit"])
    print(f"round {step.step}: assayed {len(step.acquired_batch)}, {found} hits")
```

To reach your method from the CLI, add a branch to `make_model` in
`src/assayloop/experiment/runner.py`. The five abstractions (`Task`, `Model`,
`AcquisitionFunction`, `Metric`, `SequentialLoop`) live in `assaybench`, so
`pip install assaybench` is enough to benchmark a policy without this repo.

## Downloads

| Command | Size | Needed for |
|---|---|---|
| `bash scripts/fetch_sweeps.sh` | 5.5 MB | the published LLM and external-baseline runs behind the paper's table |
| `bash scripts/fetch_presage_cache.sh` | 3.4 GB | `knn`, `rf`, `biobo`, `llmnn`, Vendi diversity |
| `bash scripts/fetch_gene_sets.sh` | MSigDB | `bio_ucb`, pathway metrics, the sunburst figure |
| `bash scripts/fetch_ground_truth.sh` | STRING / CORUM / SIGNOR | network-recovery analysis |

Nothing silently substitutes for a missing download. The code raises and names the fetch
script.

## Reproducing the paper's table

```bash
bash scripts/fetch_sweeps.sh
uv run assayloop figure main-table --min-screen-freq 2
uv run assayloop figure --list             # every paper figure, and what has to exist first
```

The AssayFormer, BPMF, MAML and random rows re-run from a checkpoint. The LLM and
external-baseline rows are ~400 paid API runs across nine vendors, published as the sweep
bundle above. Nothing prints a dash for data you did not download; it raises and names the
fetch script. [`README_DETAILED.md`](README_DETAILED.md) has the per-row commands, every
baseline and acquisition, and the figure scripts.

## Citation

If you use this code, please cite the AssayLoop paper. <!-- TODO: replace with the final
BibTeX entry once the paper has a citable reference. -->
