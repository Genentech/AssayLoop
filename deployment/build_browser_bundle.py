"""Build the static assets used for in-browser AssayFormer inference."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from assayloop.amortized.text_embed import (
    EmbeddingCache,
    OpenAITextEmbedder,
    embed_texts,
    screen_description_text,
)

from export_onnx import (
    MODEL_SHA256,
    PUBLIC_BROWSER_GRAPH_SHA256,
    _validate_release_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "deployment" / "examples"
F2_GENES = ROOT / "deployment" / "data" / "f2_genes.json"
OUTPUT_DATA = ROOT / "docs" / "assets" / "data" / "assayformer_browser.json"
OUTPUT_MODEL = ROOT / "docs" / "assets" / "model" / "assayformer.onnx"


def _sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def build(checkpoint: Path, onnx_model: Path) -> None:
    _validate_release_checkpoint(checkpoint)
    if not onnx_model.is_file():
        raise RuntimeError(f"ONNX model does not exist: {onnx_model}")
    model_sha = _sha256(onnx_model)
    if model_sha != PUBLIC_BROWSER_GRAPH_SHA256:
        raise RuntimeError(
            "Refusing to publish an ONNX graph that is not the reviewed public "
            "s19 browser artifact. Verify the new graph and deliberately update "
            "PUBLIC_BROWSER_GRAPH_SHA256 before releasing it."
        )
    examples = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in EXAMPLES_DIR.glob("*.json")
    ]
    examples.sort(key=lambda row: row["order"])
    texts = [screen_description_text(row["screen"]) for row in examples]
    vectors = embed_texts(
        texts,
        OpenAITextEmbedder(),
        cache=EmbeddingCache(path=Path("/dev/null")),
        save=False,
    )
    if vectors.shape != (len(examples), 1536):
        raise RuntimeError(f"Unexpected description embedding shape: {vectors.shape}")

    vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))["itos"]
    genes = json.loads(F2_GENES.read_text(encoding="utf-8"))
    payload = {
        "schema": 1,
        "model": {
            "path": f"assets/model/assayformer.onnx?v={model_sha[:12]}",
            "sha256": model_sha,
            "checkpoint_sha256": MODEL_SHA256,
            "runtime": "onnxruntime-web@1.29.0",
            "opset": 17,
        },
        "description_embedding": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "encoding": "base64-float32-le",
        },
        "probability_samples": 500,
        "vocab": vocab,
        "candidate_genes": genes,
        "example_embeddings": {
            row["screen_id"]: base64.b64encode(
                np.asarray(vectors[index], dtype="<f4").tobytes()
            ).decode("ascii")
            for index, row in enumerate(examples)
        },
    }

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_model, OUTPUT_MODEL)
    print(
        f"Wrote {OUTPUT_DATA} ({OUTPUT_DATA.stat().st_size / 1024:.1f} KiB) and "
        f"{OUTPUT_MODEL} ({OUTPUT_MODEL.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path(__file__).resolve().parent
        / ".model-cache"
        / "assayformer.onnx",
    )
    args = parser.parse_args()
    build(args.checkpoint.expanduser().resolve(), args.onnx_model.expanduser().resolve())


if __name__ == "__main__":
    main()
