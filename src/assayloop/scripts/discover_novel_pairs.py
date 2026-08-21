"""Discover novel gene associations from model influence that aren't in standard databases.

For a set of probe genes, computes context-conditional influence, then filters
out pairs found in STRING, CORUM, SIGNOR, or Reactome. The remaining high-influence
pairs are candidates for novel biological associations learned from CRISPR screens.

Outputs a ranked list + a deep research prompt for validation.

Usage::

    uv run python -m assayloop.scripts.discover_novel_pairs \
        --checkpoint gf-bpmf-train-hits-rl-fg-s19 --ckpt-file model_last.pt \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

import numpy as np

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("discover_novel_pairs")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
# STRING / CORUM / SIGNOR parquet tables. Not redistributed with this repo:
# set ASSAYLOOP_GROUND_TRUTH to a directory holding STRING-HUMAN/,
# CORUM-HUMAN/ and SIGNOR-HUMAN/. Every source is required: this script
# reports pairs that are *absent* from the known set, so a source that
# quietly failed to load would show up as a longer list of "novel" pairs.


PROBE_GENES = [
    "TP53", "KRAS", "BRCA1", "EGFR", "MYC", "PTEN", "BRAF", "PIK3CA",
    "RB1", "APC", "SMAD4", "ATM", "CDKN2A", "ERBB2", "NRAS",
    "MDM2", "CDK4", "CTNNB1", "NOTCH1", "MTOR",
]


def _load_known_pairs():
    """Load gene-gene pairs from STRING, CORUM, and SIGNOR."""
    import pandas as pd

    known = set()

    # STRING
    string_dir = config.ground_truth_dir("STRING-HUMAN")
    genes_df = pd.read_parquet(string_dir / "dimension_GeneSymbol.parquet")
    edges_df = pd.read_parquet(string_dir / "dimension_Interactions.parquet")
    id_to_gene = {k: v for k, v in zip(genes_df["string_human_id"], genes_df["gene_symbol"])
                  if isinstance(v, str)}
    for _, row in edges_df.iterrows():
        g1 = id_to_gene.get(row["string_human_id_1"])
        g2 = id_to_gene.get(row["string_human_id_2"])
        if g1 and g2 and isinstance(g1, str) and isinstance(g2, str):
            known.add((min(g1, g2), max(g1, g2)))
    log.info("STRING: %d edges", len(known))

    # CORUM
    n_before = len(known)
    corum_dir = config.ground_truth_dir("CORUM-HUMAN")
    subunits = pd.read_parquet(corum_dir / "dimension_Subunit.parquet")
    complexes = {}
    for _, row in subunits.iterrows():
        gene = row.get("gene_name")
        if not isinstance(gene, str):
            continue
        gene = gene.strip().upper()
        if gene:
            complexes.setdefault(row["complex_id"], set()).add(gene)
    for members in complexes.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                known.add((members[i], members[j]))
    log.info("CORUM: +%d edges", len(known) - n_before)

    # SIGNOR
    n_before = len(known)
    signor_dir = config.ground_truth_dir("SIGNOR-HUMAN")
    ents = pd.read_parquet(signor_dir / "Dimension_Entities.parquet")
    facts = pd.read_parquet(signor_dir / "Fact_Protein_Interactions.parquet")
    key_to_gene = {}
    for _, row in ents.iterrows():
        if row.get("TYPE") == "protein":
            name = row.get("ENTITY_NAME")
            if isinstance(name, str):
                key_to_gene[row["ENTITY_KEY"]] = name.upper()
    for _, row in facts.iterrows():
        g1 = key_to_gene.get(row["ENTITYA_KEY"])
        g2 = key_to_gene.get(row["ENTITYB_KEY"])
        if g1 and g2 and g1 != g2:
            known.add((min(g1, g2), max(g1, g2)))
    log.info("SIGNOR: +%d edges", len(known) - n_before)

    # Pathway co-membership (MSigDB c2.cp includes Reactome, KEGG, BioCarta)
    n_before = len(known)
    gmt_dir = config.require_path(
        config.GENE_SETS_PATH,
        env_var="ASSAYLOOP_GENE_SETS",
        what="the gene-set (.gmt) directory",
        hint="Run scripts/fetch_gene_sets.sh.",
    )
    gmt_paths = sorted(gmt_dir.glob("*.gmt"))
    if not gmt_paths:
        raise config.MissingConfiguredPath(
            f"No .gmt files in {gmt_dir}. Run scripts/fetch_gene_sets.sh. "
            "Skipping them would inflate the novel-pair list."
        )
    for gmt_path in gmt_paths:
        with open(gmt_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                members = [g.strip() for g in parts[2:] if g.strip()]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        known.add((min(members[i], members[j]),
                                  max(members[i], members[j])))
    log.info("Pathway co-membership (GMT): +%d edges", len(known) - n_before)

    log.info("Total known pairs: %d", len(known))
    return known


def _compute_influence(net, vocab, desc_emb, gene_symbol, device,
                       universe_set, n_samples=20, n_background=50):
    """Compute influence of observing gene as hit on all other genes."""
    import torch

    gene_id = vocab.to_idx(gene_symbol)
    if gene_id == 0:
        gene_id = vocab.to_idx(gene_symbol.upper())
    if gene_id == 0:
        return {}

    n_vocab = len(vocab)
    all_ids = [i for i in range(1, n_vocab)]
    rng = np.random.default_rng(42)

    delta_acc = torch.zeros(n_vocab, device=device)
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

        V = net.gene_emb.weight.detach()
        delta_repr = repr_probe - repr_base
        delta_scores = V @ delta_repr.squeeze(0)
        delta_acc += delta_scores

    deltas = (delta_acc / n_samples).cpu().numpy()

    result = {}
    for vid in range(1, n_vocab):
        sym = vocab.itos[vid] if vid < len(vocab.itos) else None
        if sym and sym != gene_symbol and sym in universe_set:
            result[sym] = float(deltas[vid])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gf-bpmf-train-hits-rl-fg-s19")
    ap.add_argument("--ckpt-file", default="model_last.pt")
    ap.add_argument("--genes", default=None,
                    help="Comma-sep probe genes (default: curated cancer genes)")
    ap.add_argument("--n-top", type=int, default=20,
                    help="Top novel pairs per probe gene")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    probe_genes = args.genes.split(",") if args.genes else PROBE_GENES

    # Build universe filter
    from assayloop.tasks import load_screens
    all_screens = load_screens(target_set="public")
    gene_freq = Counter()
    for s in all_screens:
        for g in set(s.genes):
            gene_freq[g] += 1
    universe_set = {g for g in gene_freq if gene_freq[g] >= 2}

    # Load known interactions
    known = _load_known_pairs()

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

    log.info("Loaded model, %d known pairs, %d probe genes",
             len(known), len(probe_genes))

    # Compute influence and find novel pairs
    all_novel = []
    for gene in probe_genes:
        log.info("Probing %s...", gene)
        influence = _compute_influence(net, vocab, desc_emb, gene, device,
                                       universe_set, n_samples=args.n_samples)
        if not influence:
            log.warning("  %s not in vocab, skipping", gene)
            continue

        # Sort by absolute influence (both boosted and suppressed)
        sorted_genes = sorted(influence.items(), key=lambda x: abs(x[1]), reverse=True)

        novel_count = 0
        for target, score in sorted_genes:
            pair = (min(gene, target), max(gene, target))
            is_known = pair in known or (gene.upper(), target.upper()) in known
            if not is_known:
                all_novel.append({
                    "probe": gene, "target": target,
                    "influence": score, "direction": "boosted" if score > 0 else "suppressed",
                    "abs_influence": abs(score),
                })
                novel_count += 1
                if novel_count >= args.n_top:
                    break

        log.info("  %s: %d novel pairs (top influence: %s %.3f)",
                 gene, novel_count,
                 all_novel[-1]["target"] if all_novel else "?",
                 all_novel[-1]["influence"] if all_novel else 0)

    # Take top N per probe gene for balanced coverage
    per_gene_top = 3
    seen_pairs = set()
    balanced_novel = []
    for gene in probe_genes:
        gene_pairs = [p for p in all_novel if p["probe"] == gene]
        gene_pairs.sort(key=lambda x: x["abs_influence"], reverse=True)
        count = 0
        for p in gene_pairs:
            pair_key = (min(p["probe"], p["target"]), max(p["probe"], p["target"]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            balanced_novel.append(p)
            count += 1
            if count >= per_gene_top:
                break

    # Save all novel pairs
    out_dir = config.OUTPUT_PATH / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "novel_pairs.json"
    out_file.write_text(json.dumps(all_novel, indent=2))
    log.info("Wrote %d novel pairs to %s", len(all_novel), out_file)

    # Print balanced top pairs (diverse coverage)
    print()
    print("Top novel gene associations (balanced across probe genes):")
    print("%-8s %-12s %10s  %s" % ("Probe", "Target", "Influence", "Direction"))
    print("-" * 50)
    for p in balanced_novel:
        print("%-8s %-12s %10.4f  %s" % (
            p["probe"], p["target"], p["influence"], p["direction"]))

    # Generate deep research prompt from balanced pairs
    top_pairs = balanced_novel
    pair_list = "\n".join([
        "- %s → %s (influence=%.3f, %s): When %s is knocked out and found to be a hit, "
        "the model predicts %s is %s likely to be a hit." % (
            p["probe"], p["target"], p["influence"], p["direction"],
            p["probe"], p["target"],
            "more" if p["direction"] == "boosted" else "less")
        for p in top_pairs
    ])

    prompt = f"""I have a transformer model trained on 1,349 CRISPR knockout screens that has
learned gene-gene associations through in-context learning. The model observes which genes
are hits in a screen and uses that to predict which other genes will be hits.

The following gene pairs show strong context-conditional influence in the model but are
NOT found in standard interaction databases (STRING, CORUM, SIGNOR). I need to find
supporting evidence in the scientific literature for these associations.

For each pair, please search for:
1. Direct functional evidence (e.g., one gene regulates the other, they're in the same pathway)
2. Genetic interaction evidence (synthetic lethality, epistasis, suppression)
3. Co-essential evidence (both required for the same cellular process)
4. Recent papers (2020+) that might explain the association
5. Whether the association makes biological sense given known gene functions

Gene pairs to investigate:
{pair_list}

For each pair, provide:
- A one-sentence summary of the strongest evidence found
- The most relevant paper citation
- A confidence rating (strong/moderate/weak/no evidence)
- Whether this association was known before our model predicted it

Focus especially on pairs where the model's prediction is surprising or novel."""

    prompt_file = out_dir / "novel_pairs_research_prompt.txt"
    prompt_file.write_text(prompt)
    log.info("Wrote research prompt to %s", prompt_file)
    print("\nDeep research prompt saved to: %s" % prompt_file)
    print("Run with: /deep-research < %s" % prompt_file)


if __name__ == "__main__":
    main()
