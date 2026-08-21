"""Compute the full gene × gene influence matrix.

For each probe gene X, measures how observing X as a hit changes the
model's predicted score for every other gene Y. The result is an
asymmetric matrix I[X, Y] = E[score(Y | context ∪ {X=hit}) - score(Y | context)].

Usage::

    # Full universe (~21k genes, ~15 min on GPU)
    uv run python -m assayloop.scripts.compute_influence_matrix \
        --checkpoint gf-bpmf-train-hits-rl-fg-s19 --ckpt-file model_last.pt \
        --device cuda

    # Top 5k genes only (~4 min)
    uv run python -m assayloop.scripts.compute_influence_matrix \
        --checkpoint gf-bpmf-train-hits-rl-fg-s19 --ckpt-file model_last.pt \
        --device cuda --max-genes 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_influence_matrix")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gf-bpmf-train-hits-rl-fg-s19")
    ap.add_argument("--ckpt-file", default="model_last.pt")
    ap.add_argument("--n-samples", type=int, default=10,
                    help="Background context samples per probe gene")
    ap.add_argument("--n-background", type=int, default=50,
                    help="Random genes in each background context")
    ap.add_argument("--max-genes", type=int, default=0,
                    help="Limit to top N genes by training frequency (0 = full universe)")
    ap.add_argument("--min-screen-freq", type=int, default=2,
                    help="Universe filter: genes in >= N test screens")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Probe genes per batch (memory vs speed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None,
                    help="Output path (default: output/analysis/influence_matrix.npz)")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model
    from assayloop.amortized.data import GeneVocab
    from assayloop.amortized.model import RankerConfig, RankerNet
    from assayloop.amortized.text_embed import get_text_embedder, embed_texts

    ckpt_dir = RANKERS_DIR / args.checkpoint
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    vocab = GeneVocab.load(ckpt_dir / "vocab.json")

    net = RankerNet(RankerConfig(**cfg["arch"]))
    state = torch.load(ckpt_dir / args.ckpt_file, map_location=device, weights_only=False)
    net.load_state_dict(state)
    net.to(device).eval()

    # Description embedding
    embedder = get_text_embedder(cfg.get("text_backend", "auto"))
    text = "A genome-wide CRISPR knockout screen to identify essential genes."
    emb = embed_texts([text], embedder)[0]
    desc_emb = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)

    V = net.gene_emb.weight.detach()  # (vocab_size, d_gene)
    d_gene = V.shape[1]

    # Build gene list (universe-filtered)
    from assayloop.tasks import load_screens
    all_screens = load_screens(target_set="public")
    gene_freq = Counter()
    for s in all_screens:
        for g in set(s.genes):
            gene_freq[g] += 1
    universe = {g for g in gene_freq if gene_freq[g] >= args.min_screen_freq}

    # Map universe genes to vocab IDs
    gene_ids = []
    gene_names = []
    for g in sorted(universe):
        vid = vocab.to_idx(g)
        if vid == 0:
            vid = vocab.to_idx(g.upper())
        if vid != 0:
            gene_ids.append(vid)
            gene_names.append(g)

    if args.max_genes > 0 and len(gene_ids) > args.max_genes:
        # Keep genes with highest training frequency
        freq_order = sorted(range(len(gene_names)),
                           key=lambda i: -gene_freq.get(gene_names[i], 0))
        keep = freq_order[:args.max_genes]
        keep.sort()  # restore sorted order
        gene_ids = [gene_ids[i] for i in keep]
        gene_names = [gene_names[i] for i in keep]

    N = len(gene_ids)
    log.info("Computing %d × %d influence matrix (%d samples, %d background)...",
             N, N, args.n_samples, args.n_background)

    gene_ids_t = torch.tensor(gene_ids, device=device)
    all_vocab_ids = list(range(1, len(vocab)))
    rng = np.random.default_rng(args.seed)

    # Pre-compute V for target genes
    V_targets = V[gene_ids_t]  # (N, d_gene)

    # Compute influence matrix
    influence = np.zeros((N, N), dtype=np.float32)
    t0 = time.time()

    for probe_idx in range(N):
        probe_id = gene_ids[probe_idx]

        delta_repr_acc = torch.zeros(d_gene, device=device)

        for _ in range(args.n_samples):
            bg_ids = rng.choice(all_vocab_ids,
                                size=min(args.n_background, len(all_vocab_ids)),
                                replace=False).tolist()
            bg_ids = [x for x in bg_ids if x != probe_id]
            bg_hits = rng.integers(0, 2, size=len(bg_ids)).tolist()

            ctx_ids = torch.tensor([bg_ids], device=device)
            ctx_hits = torch.tensor([bg_hits], device=device)
            ctx_pad = torch.zeros(1, len(bg_ids), device=device)

            with torch.no_grad():
                repr_base = net.encode(desc_emb, ctx_ids, ctx_hits, ctx_pad)

            ctx_ids_p = torch.cat([ctx_ids,
                                   torch.tensor([[probe_id]], device=device)], dim=1)
            ctx_hits_p = torch.cat([ctx_hits,
                                    torch.tensor([[1]], device=device)], dim=1)
            ctx_pad_p = torch.zeros(1, ctx_ids_p.shape[1], device=device)

            with torch.no_grad():
                repr_probe = net.encode(desc_emb, ctx_ids_p, ctx_hits_p, ctx_pad_p)

            delta_repr_acc += (repr_probe - repr_base).squeeze(0)

        # Average delta, project onto all target genes
        delta_repr_avg = delta_repr_acc / args.n_samples  # (d_gene,)
        with torch.no_grad():
            scores = (V_targets @ delta_repr_avg).cpu().numpy()  # (N,)
        influence[probe_idx] = scores

        if (probe_idx + 1) % 500 == 0 or probe_idx == 0:
            elapsed = time.time() - t0
            rate = (probe_idx + 1) / elapsed
            eta = (N - probe_idx - 1) / rate
            log.info("  [%d/%d] %.1f genes/s, ETA %.0fs",
                     probe_idx + 1, N, rate, eta)

    elapsed = time.time() - t0
    log.info("Done in %.1fs (%.1f genes/s)", elapsed, N / elapsed)

    # Save
    out_dir = config.OUTPUT_PATH / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = args.checkpoint.replace("/", "_")
    file_tag = args.ckpt_file.replace(".pt", "")
    out_name = "influence_matrix_%s_%s.npz" % (ckpt_tag, file_tag)
    out_path = Path(args.out) if args.out else out_dir / out_name

    np.savez_compressed(out_path,
                        influence=influence,
                        gene_names=np.array(gene_names),
                        gene_ids=np.array(gene_ids),
                        checkpoint=args.checkpoint,
                        ckpt_file=args.ckpt_file,
                        n_samples=args.n_samples,
                        n_background=args.n_background)
    log.info("Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Summary stats
    sym = (influence + influence.T) / 2
    np.fill_diagonal(sym, 0)
    print()
    print("Influence matrix: %d × %d" % (N, N))
    print("  Range: [%.4f, %.4f]" % (influence.min(), influence.max()))
    print("  Mean abs: %.4f" % np.abs(influence).mean())
    print("  Symmetry (corr of I and I^T): %.4f" % np.corrcoef(
        influence.ravel(), influence.T.ravel())[0, 1])
    print()

    # Top 10 strongest pairs
    triu = np.triu_indices(N, k=1)
    sym_flat = sym[triu]
    top_boost = np.argsort(-sym_flat)[:10]
    top_suppress = np.argsort(sym_flat)[:10]
    print("Top 10 boosted pairs (symmetric):")
    for idx in top_boost:
        i, j = triu[0][idx], triu[1][idx]
        print("  %s ↔ %s: %.4f" % (gene_names[i], gene_names[j], sym_flat[idx]))
    print()
    print("Top 10 suppressed pairs (symmetric):")
    for idx in top_suppress:
        i, j = triu[0][idx], triu[1][idx]
        print("  %s ↔ %s: %.4f" % (gene_names[i], gene_names[j], sym_flat[idx]))

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
