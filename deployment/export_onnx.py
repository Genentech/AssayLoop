"""Export the released AssayFormer checkpoint as a browser-compatible ONNX graph.

The graph accepts an already-computed 1,536-dimensional screen-description
embedding, observed gene ids and hit labels, and candidate gene ids. Text
embedding stays outside the graph so released examples can use cached vectors
while genuinely new descriptions can use a separately configured embedding
endpoint.

The default destination is under ``deployment/.model-cache`` (gitignored).
Publishing the graph is a separate, deliberate release step because an ONNX
file contains the complete model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from assayloop.amortized.model import RankerConfig, RankerNet

MODEL_FILE = "model_last.pt"
MODEL_SHA256 = "c5a58d77c43916cd1261410cc0085fc45877dd7888db09aed9022002738d39f4"
# Exact graph reviewed and shipped with the public website. Exporter/library
# upgrades can legitimately change the serialized graph; review and reverify
# such a graph before deliberately updating this release checksum.
PUBLIC_BROWSER_GRAPH_SHA256 = (
    "0c4a78861513181262f5afb3b7473b7ef2d0325a2b7dde3b140760ac2bd7326c"
)


def _validate_release_checkpoint(checkpoint: Path) -> None:
    required = {"config.json", "vocab.json", MODEL_FILE}
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise RuntimeError(f"AssayFormer release is missing: {', '.join(missing)}")
    digest = hashlib.file_digest((checkpoint / MODEL_FILE).open("rb"), "sha256")
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError("Checkpoint weights do not match the paper s19 artifact.")


class BrowserAssayFormer(nn.Module):
    """Export-friendly equivalent of the paper model's inference path.

    PyTorch's legacy ONNX lowering for ``nn.MultiheadAttention`` bakes the
    traced sequence length into an internal reshape. Writing the same
    scaled-dot-product attention explicitly keeps the context axis dynamic,
    which the browser needs for zero through 1,024 observations.
    """

    def __init__(self, net: RankerNet):
        super().__init__()
        self.net = net

    def forward(
        self,
        description_embedding: torch.Tensor,
        observed_gene_ids: torch.Tensor,
        observed_hit_labels: torch.Tensor,
        candidate_gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = description_embedding.shape[0]
        description_token = (
            self.net.desc_proj(description_embedding) + self.net.type_emb.weight[0]
        ).unsqueeze(1)
        observed_genes = self.net.gene_emb(observed_gene_ids)
        observed_hits = self.net.hit_emb(observed_hit_labels.clamp(0, 1))
        observed_tokens = self.net.ctx_proj(
            torch.cat([observed_genes, observed_hits], dim=-1)
        ) + self.net.type_emb.weight[1]
        hidden = torch.cat([description_token, observed_tokens], dim=1)

        for layer in self.net.encoder.layers:
            attention = layer.self_attn
            embed_dim = attention.embed_dim
            num_heads = attention.num_heads
            head_dim = embed_dim // num_heads
            sequence_length = hidden.shape[1]
            qkv = F.linear(hidden, attention.in_proj_weight, attention.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.reshape(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2)
            key = key.reshape(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2)
            value = value.reshape(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2)
            weights = torch.softmax(
                torch.matmul(query, key.transpose(-2, -1)) / head_dim**0.5,
                dim=-1,
            )
            attended = torch.matmul(weights, value).transpose(1, 2).reshape(
                batch_size, sequence_length, embed_dim
            )
            hidden = layer.norm1(hidden + attention.out_proj(attended))
            feed_forward = layer.linear2(layer.activation(layer.linear1(hidden)))
            hidden = layer.norm2(hidden + feed_forward)

        description_repr = self.net.out_proj(self.net.norm(hidden[:, 0, :]))
        return self.net.score_ids(description_repr, candidate_gene_ids)


def _load_net(checkpoint: Path) -> RankerNet:
    _validate_release_checkpoint(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    net = RankerNet(RankerConfig(**config["arch"]))
    state = torch.load(checkpoint / MODEL_FILE, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def _load_model(checkpoint: Path) -> BrowserAssayFormer:
    return BrowserAssayFormer(_load_net(checkpoint)).eval()


def export(checkpoint: Path, output: Path) -> None:
    model = _load_model(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = (
        torch.zeros(1, 1536, dtype=torch.float32),
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[3, 4, 5]], dtype=torch.int64),
    )
    torch.onnx.export(
        model,
        inputs,
        output,
        input_names=[
            "description_embedding",
            "observed_gene_ids",
            "observed_hit_labels",
            "candidate_gene_ids",
        ],
        output_names=["acquisition_scores"],
        dynamic_axes={
            "observed_gene_ids": {1: "context_length"},
            "observed_hit_labels": {1: "context_length"},
            "candidate_gene_ids": {1: "candidate_count"},
            "acquisition_scores": {1: "candidate_count"},
        },
        opset_version=17,
        dynamo=False,
        external_data=False,
    )

    import onnx

    graph = onnx.load(output)
    graph.metadata_props.add(
        key="assayloop.checkpoint_sha256", value=MODEL_SHA256
    )
    graph.metadata_props.add(key="assayloop.candidate_universe", value="f2:21147")
    graph.metadata_props.add(key="assayloop.text_embedding", value="1536")
    onnx.checker.check_model(graph)
    onnx.save(graph, output)


def verify(checkpoint: Path, output: Path) -> None:
    import onnxruntime as ort

    torch.manual_seed(7)
    model = _load_model(checkpoint)
    source = _load_net(checkpoint)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for context_length in (0, 7, 200):
        candidate_count = 257
        description = torch.randn(1, 1536, dtype=torch.float32)
        observed_ids = torch.randint(1, 24217, (1, context_length))
        observed_hits = torch.randint(0, 2, (1, context_length))
        candidate_ids = torch.randint(1, 24217, (1, candidate_count))
        with torch.no_grad():
            expected = model(
                description, observed_ids, observed_hits, candidate_ids
            ).numpy()
            source_scores = source(
                description,
                observed_ids,
                observed_hits,
                torch.zeros_like(observed_ids, dtype=torch.bool),
                candidate_ids,
            ).numpy()
        np.testing.assert_allclose(expected, source_scores, rtol=2e-4, atol=2e-5)
        actual = session.run(
            ["acquisition_scores"],
            {
                "description_embedding": description.numpy(),
                "observed_gene_ids": observed_ids.numpy(),
                "observed_hit_labels": observed_hits.numpy(),
                "candidate_gene_ids": candidate_ids.numpy(),
            },
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        if not np.array_equal(np.argsort(actual[0]), np.argsort(expected[0])):
            raise AssertionError(
                f"ONNX rank order differs from PyTorch at context length {context_length}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / ".model-cache" / "assayformer.onnx",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    export(checkpoint, output)
    if args.verify:
        verify(checkpoint, output)
    print(f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
