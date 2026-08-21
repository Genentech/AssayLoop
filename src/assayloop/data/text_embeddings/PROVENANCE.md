# Shipped screen-description embeddings

`screen_descriptions.npz` -- 1356 embeddings of `text-embedding-3-small`, 1536-d float32,
keyed by `sha1(model + "\x00" + text)` (see
`assayloop.amortized.text_embed._text_key`).

| | |
|---|---|
| Model | `text-embedding-3-small` |
| Dimensionality | 1536 |
| Entries | 1356 |
| Bytes | 8,754,358 |
| sha256 | `ff3c1f1a6ab7d27f31e12030dc7af0a36b3d324623dd7d30242115ea3ebd7eae` |
| Built by | `python -m assayloop.scripts.build_text_embedding_cache` (re-keyed from cache.npz) |

## Coverage

One entry per distinct screen description across the public screen sets:

| Screen set | Screens |
|---|---|
| `public` | 20 |
| `public_validation` | 20 |
| `public_train` | 1349 |
| `public_val` | 218 |

Distinct texts total fewer than the sum of the rows because the smaller
evaluation sets are subsets of the training fold.

Not covered: the other 314 screens of the biogrid test fold (`public_test`).
The paper reports on the curated 20, so those are what ships; evaluating
ASSAYFORMER on the full test fold needs an `OPENAI_API_KEY`.

## Why this ships

ASSAYFORMER conditions on a 1536-d embedding of the screen description.
Shipping the vectors means reproducing every number in the paper needs no
API key and costs nothing. Screens outside these sets are not cached; the
embedder raises and names the credential to set rather than substituting a
different model.

## Provenance of the vectors

The research runs called `text-embedding-3-small` through an Azure OpenAI
deployment. These are those exact vectors, re-keyed from the
backend-qualified cache key to the bare model id -- not re-embedded -- so
they are bit-identical to what the published results used. The public
OpenAI API serves the same model. To confirm the two endpoints agree
before relying on that, with your own key:

```
OPENAI_API_KEY=sk-... python -m assayloop.scripts.build_text_embedding_cache \
    --from-cache <research cache>.npz --verify 25 --dry-run
```
