"""Compute gene × gene cosine similarity matrix from embeddings.

Supports three sources:
  1. Trained checkpoint: gene_embeddings.npy or model.pt gene_emb weights
  2. Raw init: bpmf, mf, mf:raw, svd, genept, k562, random

Usage::

    # From a trained checkpoint
    uv run python -m assayloop.scripts.compute_embedding_matrix \
        --checkpoint gf-bpmf-train-hits --ckpt-file model.pt

    # From raw init (no training)
    uv run python -m assayloop.scripts.compute_embedding_matrix \
        --raw-source bpmf

    # Multiple raw sources at once
    uv run python -m assayloop.scripts.compute_embedding_matrix \
        --raw-source bpmf,mf,mf:raw,svd,genept,k562,random
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter

import numpy as np

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_embedding_matrix")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"


def _cosine_matrix(V, gene_ids):
    """Compute cosine similarity for selected gene IDs."""
    E = V[gene_ids].astype(np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    E_norm = E / norms
    return (E_norm @ E_norm.T).astype(np.float32)


def _save_matrix(sim, gene_names, gene_ids, label, out_dir):
    tag = label.replace("/", "_").replace(":", "_").replace(" ", "_")
    out_path = out_dir / ("embedding_matrix_%s.npz" % tag)
    np.savez_compressed(out_path,
                        similarity=sim,
                        gene_names=np.array(gene_names),
                        gene_ids=np.array(gene_ids),
                        label=label)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Summary
    triu = np.triu_indices(len(gene_names), k=1)
    vals = sim[triu]
    print("  %s: mean=%.4f std=%.4f range=[%.4f, %.4f]" % (
        label, vals.mean(), vals.std(), vals.min(), vals.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="Comma-sep checkpoint names (optionally name:ckpt_file)")
    ap.add_argument("--raw-source", default=None,
                    help="Comma-sep raw sources: bpmf,mf,mf:raw,svd,genept,k562,random")
    ap.add_argument("--min-screen-freq", type=int, default=2)
    ap.add_argument("--d-gene", type=int, default=10,
                    help="Dimension for raw sources")
    args = ap.parse_args()

    from assayloop.tasks import load_screens
    from assayloop.amortized.data import GeneVocab

    # Build universe
    all_screens = load_screens(target_set="public")
    gene_freq = Counter()
    for s in all_screens:
        for g in set(s.genes):
            gene_freq[g] += 1
    universe = {g for g in gene_freq if gene_freq[g] >= args.min_screen_freq}

    out_dir = config.OUTPUT_PATH / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process trained checkpoints
    if args.checkpoint:
        for entry in args.checkpoint.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                name, ckpt_file = entry.split(":", 1)
            else:
                name, ckpt_file = entry, "model.pt"

            ckpt_dir = RANKERS_DIR / name
            if not ckpt_dir.exists():
                log.warning("Checkpoint %s not found, skipping", name)
                continue

            log.info("Loading %s (%s)...", name, ckpt_file)
            cfg = json.loads((ckpt_dir / "config.json").read_text())
            vocab = GeneVocab.load(ckpt_dir / "vocab.json")

            # Load gene embeddings
            if "arch" in cfg:
                emb_path = ckpt_dir / "gene_embeddings.npy"
                if emb_path.exists() and ckpt_file == "model.pt":
                    V = np.load(emb_path)
                else:
                    import torch
                    state = torch.load(ckpt_dir / ckpt_file,
                                      map_location="cpu", weights_only=False)
                    if "model_state_dict" in state:
                        state = state["model_state_dict"]
                    V = state["gene_emb.weight"].numpy()
            else:
                import torch
                state = torch.load(ckpt_dir / ckpt_file,
                                  map_location="cpu", weights_only=False)
                V = state["V"]

            # Map universe to vocab IDs
            gene_ids, gene_names = [], []
            for g in sorted(universe):
                vid = vocab.to_idx(g)
                if vid == 0:
                    vid = vocab.to_idx(g.upper())
                if vid != 0 and vid < V.shape[0]:
                    gene_ids.append(vid)
                    gene_names.append(g)

            log.info("  V=%s, %d universe genes mapped", V.shape, len(gene_ids))
            sim = _cosine_matrix(V, gene_ids)

            label = "%s_%s" % (name, ckpt_file.replace(".pt", ""))
            _save_matrix(sim, gene_names, gene_ids, label, out_dir)

    # Process raw sources
    if args.raw_source:
        from assayloop.scripts.eval_gene_networks import _load_raw_embedding

        for src in args.raw_source.split(","):
            src = src.strip()
            if not src:
                continue

            log.info("Loading raw source: %s ...", src)
            t0 = time.time()
            try:
                V, vocab = _load_raw_embedding(src, d_gene=args.d_gene)
            except Exception as e:
                log.warning("Failed to load %s: %s", src, e)
                continue
            log.info("  Loaded in %.1fs, V=%s", time.time() - t0, V.shape)

            gene_ids, gene_names = [], []
            for g in sorted(universe):
                vid = vocab.to_idx(g)
                if vid == 0:
                    vid = vocab.to_idx(g.upper())
                if vid != 0 and vid < V.shape[0]:
                    gene_ids.append(vid)
                    gene_names.append(g)

            log.info("  %d universe genes mapped", len(gene_ids))
            sim = _cosine_matrix(V, gene_ids)

            label = "raw_%s" % src.replace(":", "_")
            _save_matrix(sim, gene_names, gene_ids, label, out_dir)


if __name__ == "__main__":
    main()
