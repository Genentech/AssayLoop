"""Generate the paper figure for context-conditional gene influence.

Two panels:
  Left:  Boosted examples (synthetic lethality / co-essentiality)
  Right: Suppressed examples (epistatic masking / lineage exclusion)

Each panel shows a heatmap of selected probe genes (rows) vs their
top influenced targets (columns), with pathway annotations.

Usage::

    uv run python -m assayloop.scripts.paper_influence_figure \
        --checkpoint gf-bpmf-train-hits-rl-fg-s19 --ckpt-file model_last.pt \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assayloop.scripts._figure_io import save_figure
from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_influence_figure")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"

# Curated examples from the deep research analysis
FEATURED_PAIRS = {
    "boosted": [
        # (probe, target, annotation)
        ("MYC", "SMU1", "Spliceosome\nbottleneck"),
        ("MYC", "PRPF4", "Spliceosome\nbottleneck"),
        ("MDM2", "EMG1", "Nucleolar\nstress"),
        ("MDM2", "PNO1", "Nucleolar\nstress"),
        ("MDM2", "VARS1", "Translational\nfidelity"),
        ("ERBB2", "PFDN4", "Proteotoxic\nstress"),
        ("APC", "RAD18", "DNA repair"),
        ("APC", "NBN", "DNA repair"),
        ("PIK3CA", "STIP1", "Co-chaperone"),
        ("PIK3CA", "ATM", "DNA damage"),
    ],
    "suppressed": [
        ("PIK3CA", "MIS18BP1", "Mitotic masking\n(cell cycle arrest)"),
        ("PIK3CA", "MPHOSPH10", "Mitotic masking\n(cell cycle arrest)"),
        ("SMAD4", "NDUFS7", "OXPHOS bypass\n(EMT glycolysis)"),
        ("SMAD4", "NFU1", "Fe-S cluster\n(EMT glycolysis)"),
        ("EGFR", "PPDPF", "Lineage\nexclusion"),
        ("KRAS", "DCP1A", "TGF-β bypass"),
        ("NRAS", "AMOTL2", "Mechano-\ntransduction"),
        ("BRAF", "B3GNT7", "Dediff.\n(keratan sulfate)"),
        ("BRCA1", "PRSS50", "Lineage\n(testis-specific)"),
        ("ATM", "IBA57", "Epistatic\nmasking"),
    ],
}


def _compute_influence(net, vocab, desc_emb, gene_symbol, target_genes,
                       device, n_samples=30, n_background=50):
    """Compute influence of probe gene on specific target genes."""
    import torch

    gene_id = vocab.to_idx(gene_symbol)
    if gene_id == 0:
        gene_id = vocab.to_idx(gene_symbol.upper())
    if gene_id == 0:
        return {}

    target_ids = []
    target_syms = []
    for t in target_genes:
        tid = vocab.to_idx(t)
        if tid == 0:
            tid = vocab.to_idx(t.upper())
        if tid != 0:
            target_ids.append(tid)
            target_syms.append(t)

    if not target_ids:
        return {}

    n_vocab = len(vocab)
    all_ids = list(range(1, n_vocab))
    rng = np.random.default_rng(42)
    V = net.gene_emb.weight.detach()

    target_ids_t = torch.tensor(target_ids, device=device)
    delta_acc = torch.zeros(len(target_ids), device=device)

    for _ in range(n_samples):
        bg_ids = rng.choice(all_ids, size=min(n_background, len(all_ids)),
                            replace=False).tolist()
        bg_ids = [x for x in bg_ids if x != gene_id]
        bg_hits = rng.integers(0, 2, size=len(bg_ids)).tolist()

        ctx_ids = torch.tensor([bg_ids], device=device)
        ctx_hits = torch.tensor([bg_hits], device=device)
        ctx_pad = torch.zeros(1, len(bg_ids), device=device)

        with torch.no_grad():
            repr_base = net.encode(desc_emb, ctx_ids, ctx_hits, ctx_pad)

        ctx_ids_p = torch.cat([ctx_ids, torch.tensor([[gene_id]], device=device)], dim=1)
        ctx_hits_p = torch.cat([ctx_hits, torch.tensor([[1]], device=device)], dim=1)
        ctx_pad_p = torch.zeros(1, ctx_ids_p.shape[1], device=device)

        with torch.no_grad():
            repr_probe = net.encode(desc_emb, ctx_ids_p, ctx_hits_p, ctx_pad_p)

        delta_repr = repr_probe - repr_base
        delta_scores = V[target_ids_t] @ delta_repr.squeeze(0)
        delta_acc += delta_scores

    deltas = (delta_acc / n_samples).cpu().numpy()
    return {sym: float(d) for sym, d in zip(target_syms, deltas)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gf-bpmf-train-hits-rl-fg-s19")
    ap.add_argument("--ckpt-file", default="model_last.pt")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-cache", action="store_true",
                    help="Skip model inference and reuse the influence values "
                         "cached in paper_influence_data.json (for fast layout "
                         "iteration).")
    args = ap.parse_args()

    out_dir = config.OUTPUT_PATH / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "paper_influence_data.json"

    if args.use_cache and cache_path.exists():
        log.info("Loading cached influence values from %s", cache_path)
        influence_data = json.loads(cache_path.read_text())["influence"]
    else:
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

        embedder = get_text_embedder(cfg.get("text_backend", "auto"))
        text = "A genome-wide CRISPR knockout screen to identify essential genes."
        emb = embed_texts([text], embedder)[0]
        desc_emb = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)

        # Compute influence for all featured pairs
        all_targets = set()
        for pairs in FEATURED_PAIRS.values():
            for _, target, _ in pairs:
                all_targets.add(target)
        all_probes = set()
        for pairs in FEATURED_PAIRS.values():
            for probe, _, _ in pairs:
                all_probes.add(probe)

        log.info("Computing influence for %d probes × %d targets...",
                 len(all_probes), len(all_targets))

        influence_data = {}
        for probe in sorted(all_probes):
            log.info("  Probing %s...", probe)
            inf = _compute_influence(net, vocab, desc_emb, probe,
                                     list(all_targets), device,
                                     n_samples=args.n_samples)
            influence_data[probe] = inf

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)

    for panel_idx, (direction, title, cmap) in enumerate([
        ("boosted", "Boosted (Synthetic Lethality / Co-essentiality)", "Reds"),
        ("suppressed", "Suppressed (Epistatic Masking / Lineage Exclusion)", "Blues_r"),
    ]):
        ax = fig.add_subplot(gs[panel_idx])
        pairs = FEATURED_PAIRS[direction]

        # Each row = one (probe, target) pair labeled as "PROBE → TARGET"
        row_labels = []
        values = []
        annotations_list = []
        for probe, target, ann in pairs:
            row_labels.append("%s → %s" % (probe, target))
            val = influence_data.get(probe, {}).get(target, 0)
            values.append(val)
            annotations_list.append(ann.replace("\n", " "))

        n_rows = len(row_labels)

        # Horizontal bar chart — color by magnitude (darker = stronger)
        abs_vals = np.abs(values)
        norm_vals = abs_vals / max(np.max(abs_vals), 1e-6)
        if direction == "boosted":
            colors_arr = plt.get_cmap("Reds")(0.3 + 0.6 * norm_vals)
        else:
            colors_arr = plt.get_cmap("Blues")(0.35 + 0.55 * norm_vals)
        ax.barh(range(n_rows), values, color=colors_arr, edgecolor="white",
                linewidth=0.5)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=11, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("Influence score", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.axvline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
        ax.grid(True, axis="x", alpha=0.15)

        # Annotation text — place inside bar to avoid y-axis overlap
        for i, (val, ann) in enumerate(zip(values, annotations_list)):
            if direction == "boosted":
                if abs(val) > 0.3:
                    # hug the zero (left) end — mirrors the suppressed side
                    ax.text(0.015, i, ann, va="center", ha="left",
                           fontsize=8.5, color="white", style="italic")
                else:
                    ax.text(val + 0.02, i, ann, va="center", ha="left",
                           fontsize=8.5, color="#555555", style="italic")
            else:
                # Hug the zero (right) end so the mechanism text stays clear of
                # the value label sitting outside the left tip.
                ax.text(-0.015, i, ann, va="center", ha="right",
                       fontsize=8.5, color="white", style="italic")

        # Value labels: inside the tip (boosted) / just outside the tip in dark
        # ink (suppressed) so they never collide with the centered mechanism text.
        for i, val in enumerate(values):
            if direction == "boosted":
                ax.text(val - 0.01, i, "%.2f" % val, va="center", ha="right",
                       fontsize=9.5, color="white", fontweight="bold")
            else:
                ax.text(val - 0.015, i, "%.2f" % val, va="center", ha="right",
                       fontsize=9.5, color="#222222", fontweight="bold")

        if direction == "suppressed":
            # back-to-back layout: row labels on the OUTER (right) edge so the
            # long "PROBE -> TARGET" labels don't collide with the boosted panel.
            ax.yaxis.tick_right()
            # room to the left of the longest bar for the outside value labels
            ax.set_xlim(left=min(values) - 0.13, right=0)

        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)

    save_figure(fig, out_dir / "paper_influence_figure.png",
                dpi=150, vector_dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also save the computed values
    results = {"influence": influence_data, "featured_pairs": FEATURED_PAIRS}
    (out_dir / "paper_influence_data.json").write_text(json.dumps(results, indent=2))

    # Print summary
    print()
    print("Featured pairs and computed influence:")
    for direction in ["boosted", "suppressed"]:
        print("\n--- %s ---" % direction.upper())
        for probe, target, ann in FEATURED_PAIRS[direction]:
            val = influence_data.get(probe, {}).get(target, 0)
            print("  %s → %s: %.4f  (%s)" % (probe, target, val,
                                                ann.replace("\n", " ")))


if __name__ == "__main__":
    main()
