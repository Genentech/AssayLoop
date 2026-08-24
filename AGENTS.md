# AGENTS.md

Read [`README_DETAILED.md`](README_DETAILED.md). It is the full reference: every command,
flag, path and baseline, written to be read end to end. [`README.md`](README.md) is the
short human version and deliberately omits most of it.

```bash
uv sync --extra torch          # torch is opt-in; everything else runs without it
uv run pytest                  # tests/
uv run ruff check --select F   # src tests docs
```

Two things that are easy to get wrong and that nothing will catch for you:

- **EF in the paper is `n_hits_vs_random` in the code**, and `nvr` is the legacy name for it
  throughout `history.json`, the W&B metrics, the `nvr_terminal` reward mode and the sweep
  ids. Those strings are a persistence boundary. Do not rename them.
- **A ranker checkpoint directory holds two models.** `model.pt` is best-on-validation,
  `model_last.pt` is the final epoch, and every number in the paper comes from
  `model_last.pt`. The loaders default to `model.pt`, so pass `ckpt_file="model_last.pt"`
  explicitly when reproducing a published row.
