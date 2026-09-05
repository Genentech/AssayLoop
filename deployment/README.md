# AssayFormer website demo

This directory is a clean deployment for the website's **Try It** page. It
loads only `Genentech/assayformer`'s `model_last.pt`: the renamed public artifact
for the paper's s19 GRPO checkpoint. Startup validation rejects a model
whose config does not record seed 19, `public_train`, the 24,217-gene vocabulary,
and `text-embedding-3-small`. It also verifies the weights against SHA-256
`c5a58d77c43916cd1261410cc0085fc45877dd7888db09aed9022002738d39f4`, which is
identical for the paper's local s19 file and the pinned Hub revision.

There is intentionally no BPMF checkpoint, LLM inference, model handoff,
internal annotation file, or internal screen here. The five examples are
public BioGRID ORCS screens distributed through AssayBench. Their exact screen
descriptions use the released embedding cache, so they do not need an embedding
API key.

The service ranks the released 21,147-gene f2 candidate universe. Alongside the
raw acquisition score, it estimates each gene's marginal probability of being
selected in the requested batch using 4,000 Monte Carlo draws from the same
Gumbel-top-k (Plackett-Luce) policy used during training. This is a probability
of **batch inclusion**, not a calibrated probability that the gene is a hit.

The five built-in examples can also run as a completely static website. The
browser downloads a verified ONNX copy of the same s19 model and runs it with
ONNX Runtime Web, then estimates inclusion probabilities with 500 client-side
draws. A server remains necessary for genuinely new descriptions because those
need an embedding API endpoint.

## Run

From this directory:

```bash
cp env.example .env
# Fill HF_TOKEN while the Hub repository remains private.
uv run --with-requirements requirements.txt \
  uvicorn app:app --env-file .env --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/try.html>. The app serves `../docs` at `/` and its
API at `/api`, so no API URL needs to be embedded in the public website.

For a separately hosted API, set the `assayloop-api-base` meta tag in
`docs/try.html` to its HTTPS origin and add the GitHub Pages origin to
`ASSAYLOOP_ALLOWED_ORIGINS`.

## Build the browser model

From the repository root, export and verify the public checkpoint, then build
the static bundle:

```bash
uv run --extra torch --extra onnx \
  python deployment/export_onnx.py /path/to/assayformer-checkpoint --verify
uv run --extra torch --extra onnx \
  python deployment/build_browser_bundle.py /path/to/assayformer-checkpoint
```

AssayFormer is a custom architecture rather than a model class registered in
Transformers.js, so the site uses ONNX Runtime Web directly—the browser runtime
that powers Transformers.js—while preserving AssayFormer's own forward pass.
The generated `docs/assets/model/assayformer.onnx` contains the complete model
weights. Publish it only after confirming the source checkpoint is the approved
public release. Both build steps reject any checkpoint that does not match the
paper s19 SHA-256 above, and the bundle builder also rejects any ONNX file that
does not match the separately reviewed public-graph checksum. If an exporter
upgrade changes the graph bytes, re-run parity checks before deliberately
updating `PUBLIC_BROWSER_GRAPH_SHA256`.

## Credentials and data handling

- `HF_TOKEN` is sent only by `huggingface_hub` while downloading the private
  model artifact. It is never returned by the API.
- `OPENAI_API_KEY` is optional for the built-in public examples and required
  for embedding a genuinely new screen description. It performs embedding
  only; the demo does not host or call an LLM.
- The application does not write requests or results to disk and does not log
  request bodies. Configure the surrounding proxy and platform consistently if
  submitted screen descriptions are sensitive.
- Pinning `ASSAYFORMER_MODEL_REVISION` is possible, but the default is the
  verified release revision. A local `ASSAYFORMER_CHECKPOINT` must pass the
  same s19 validation.
- `ASSAYFORMER_PROBABILITY_SAMPLES` controls the inclusion-probability Monte
  Carlo estimate (default 4,000; clamped to 500–20,000).
