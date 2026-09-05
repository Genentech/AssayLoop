# AssayLoop

Code for **"Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens."**

[:globe_with_meridians: Website](https://genentech.github.io/AssayLoop) | [:hugs: Checkpoints](https://huggingface.co/collections/Genentech/assaybench) | [:page_with_curl: Paper](README_DETAILED.md) | [![PyPI](https://img.shields.io/pypi/v/assaybench)](https://pypi.org/project/assaybench/)
<!-- TODO: the Paper badge points at README_DETAILED.md until the arXiv ID exists.
     Repoint it at the arXiv abstract then, alongside the Citation BibTeX below. -->

[![Try it yourself](https://img.shields.io/badge/Try_it_yourself!-Run_AssayFormer_in_your_browser-176b87?style=for-the-badge)](https://genentech.github.io/AssayLoop/try.html)

![The task: a screen is a library of genes, a phenotype and a hit set. Each round, a method
sees the phenotype and everything it has already assayed, and chooses the next hundred
genes.](docs/assets/figures/figure1.png?v=2d3c8d2f)

A CRISPR screen has ~20,000 candidate genes and a budget for testing a few hundred at a
time. AssayLoop chooses which genes to test next, round after round, so hits show up early.
An LLM picks the first two or three rounds from its biological priors, then **AssayFormer**,
a transformer trained on 1,349 historical screens, takes over with those observations as its
context. On 20 held-out screens it recovers **27.7% of hits after assaying 5% of the
library**, ahead of every baseline we evaluated.

## Install

```bash
git clone https://github.com/Genentech/AssayLoop
cd assayloop
uv sync --extra torch     # drop --extra torch for the CPU-only baselines and analysis
```

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
from assayloop.tasks import gene_universe, load_screens, make_task

ckpt = snapshot_download("Genentech/assayformer")
model = AmortizedRankerModel(checkpoint=ckpt, ckpt_file="model_last.pt")

screens = load_screens(target_set="paper_test")
universe = gene_universe(screens)          # the f2 pool the paper scores against
screen = screens[0]

task = make_task(screen, universe_genes=universe)
run = SequentialLoop(task, model, GreedyFromModel(),
                     metrics=[], batch_size=100).run(n_steps=10)

picked = [g for step in run.history for g in step.acquired_batch]
hits = [g for g, h in zip(screen.genes, screen.hits) if h]
print(enrichment_factor(picked, screen.genes, hits, universe, budget=1000))   # 7.61
```

No API key is needed to run AssayFormer or to reproduce its numbers.

Training your own:

```bash
uv run assayloop train-bpmf --target-set train --K 10               # gene-embedding init
uv run assayloop train-ranker                                       # supervised
uv run assayloop train-ranker-rl --init-checkpoint <ckpt-dir>       # + GRPO: EF 3.83 -> 4.83
uv run assayloop eval-ranker --checkpoint <ckpt-dir> --ckpt-file model_last.pt
```

## AssayLoop: the handoff

Collect the LLM warm start, then hand off to the ranker:

```bash
uv run assayloop run --model null --acq llm_single --screen-set paper_test --full-genome \
    --lm-config configs/lm/collect-gemini-3.1-pro.yaml

uv run assayloop eval-ranker-handoff \
    --checkpoint <rl-ckpt-dir> --ckpt-file model_last.pt \
    --warm-dir output/runs --warm-prefix sweep-<id>- --n 3
```
Open-vocabulary LLM acquisitions use the paper's shared f2 gene universe by default; pass
`--screen-library` only when you explicitly want per-screen filtering. The first `--n` rounds
are replayed from that sweep, then AssayFormer continues with them as context. For the llm
provider you set `ASSAYLOOP_LLM_PROVIDER` and that provider's key in `.env`.

## Screen sets

```python
from assayloop.tasks import load_screens

load_screens(target_set="paper_test")          # the paper's curated 20-screen test set
load_screens(target_set="test")                # the full 334-screen test fold
load_screens(dataset_names=["U_1733_merged"])  # exact screens
```

| `--screen-set` | What |
|---|---|
| `paper_test` (default) | the curated **20-screen test set** the paper reports |
| `paper_validation` | the curated 20-screen validation set (checkpoint selection) |
| `train` | the full 1,349-screen training fold |
| `validation` | the full 218-screen validation fold |
| `test` | the full 334-screen test fold |
| `lopo-*-{train,test}` | the leave-one-phenotype-out splits |
| `/path/to.yaml` | your own list |

The former `public`, `public_validation`, `public_train`, `public_val`, and
`public_test` names remain accepted as compatibility aliases for released
checkpoints and older commands.

```bash
# Same random baseline as the Quickstart, on the validation set instead of the test set.
uv run assayloop run --model null --acq random --screen-set paper_validation
```

## Metrics

Every metric takes the genes a method picked and the screen's ground truth. Two of them also
take the candidate pool, which is how a pick that is a real gene but not in this screen's
library gets charged:

```python
import numpy as np
from assaybench import (adjusted_nauc, enrichment_factor, fraction_of_hits,
                        percent_essential, shortfall)
from assayloop.tasks import gene_universe, load_screens

screens = load_screens(target_set="paper_test")
universe = gene_universe(screens)             # the f2 pool, 21,147 genes
screen = screens[0]
hits = [g for g, h in zip(screen.genes, screen.hits) if h]

picked = list(np.random.default_rng(0).choice(universe, 1000, replace=False))
rounds = [picked[i * 100:(i + 1) * 100] for i in range(10)]   # your method's ten batches

enrichment_factor(picked, screen.genes, hits, universe, budget=1000)  # EF    0.993
adjusted_nauc(rounds, screen.genes, hits, universe)                   # nAUC  0.015
fraction_of_hits(picked, screen.genes, hits)                          # FH    0.047
shortfall(picked, screen.genes)                                       # SF    0.124
percent_essential(picked, screen.genes, hits)                         # %ess  0.250
```

## Your own method

Implement `Model` (score the candidates) or `AcquisitionFunction` (choose from the scores)
and hand it to the same loop:

```python
from assaybench import Model, ModelPrediction, SequentialLoop
from assayloop.acquisitions.greedy_from_model import GreedyFromModel
from assayloop.tasks import gene_universe, load_screens, make_task

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

screens = load_screens(target_set="paper_test")
task = make_task(screens[0], universe_genes=gene_universe(screens))   # the f2 pool
run = SequentialLoop(task, MyRanker(), GreedyFromModel(),
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
| `bash scripts/fetch_sweeps.sh` | 35 MB | the published LLM/external-baseline runs and full call logs needed for exact raw-response replay |
| `bash scripts/fetch_presage_cache.sh` | 3.4 GB | `knn`, `rf`, `biobo`, `llmnn`, Vendi diversity |
| `bash scripts/fetch_gene_sets.sh` | MSigDB | `bio_ucb`, pathway metrics, the sunburst figure |
| `bash scripts/fetch_ground_truth.sh` | STRING / CORUM / SIGNOR | network-recovery analysis |


## Reproducing the paper's table

```bash
bash scripts/fetch_sweeps.sh
uv run assayloop figure main-table --min-screen-freq 2
uv run assayloop figure --list             # every paper figure, and what has to exist first
```

[`README_DETAILED.md`](README_DETAILED.md) has the per-row commands, every baseline and acquisition, and the figure scripts.

## Citation

If you use this code, please cite the AssayLoop paper. <!-- TODO: replace with the final
BibTeX entry once the paper has a citable reference. -->
