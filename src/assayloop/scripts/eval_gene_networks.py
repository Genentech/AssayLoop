"""Evaluate learned gene embeddings against known biological networks.

Computes AP and AUROC of recovering STRING interactions and CORUM complex
co-membership from pairwise cosine similarity of gene factor vectors.

Compares multiple checkpoints (random init, BPMF, RL-trained, GenePT, K562)
to show whether training improves biological coherence.

Usage::

    uv run python -m assayloop.scripts.eval_gene_networks \
        --checkpoints gf-random-train-d10,gf-bpmf-train-hits,gf-bpmf-train-hits-rl-fg-s19

    # With specific ckpt files
    uv run python -m assayloop.scripts.eval_gene_networks \
        --checkpoints gf-bpmf-train-hits-rl-fg-s19:model_last.pt
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_gene_networks")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
# STRING / CORUM / SIGNOR parquet tables. Not redistributed with this repo:
# set ASSAYLOOP_GROUND_TRUTH to a directory holding STRING-HUMAN/,
# CORUM-HUMAN/ and SIGNOR-HUMAN/. A missing source raises -- it does not
# reduce to an empty edge set, which would look like a legitimate result.


def _load_gene_factors(ckpt_name: str, ckpt_file: str = "model.pt"):
    """Load gene factors V and vocab from a checkpoint."""
    ckpt_dir = RANKERS_DIR / ckpt_name
    cfg_path = ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    from assayloop.amortized.data import GeneVocab
    vocab = GeneVocab.load(ckpt_dir / "vocab.json")

    if "arch" in cfg:
        # gene_embeddings.npy is a cached copy of gene_emb.weight, but training
        # writes it *after* reloading the best-on-val state, so it belongs to
        # model.pt and to nothing else. Taking it for any other ckpt_file
        # silently returned the wrong epoch's vectors while the results row
        # below went on reporting the ckpt_file that was asked for -- for
        # gf-bpmf-train-hits-rl-fg-s19 the two tables differ by up to 0.0095.
        # Same guard compute_embedding_matrix.py already had.
        emb_path = ckpt_dir / "gene_embeddings.npy"
        if emb_path.exists() and ckpt_file == "model.pt":
            V = np.load(emb_path)
        else:
            import torch
            state = torch.load(ckpt_dir / ckpt_file, map_location="cpu",
                              weights_only=False)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            V = state["gene_emb.weight"].numpy()
    else:
        import torch
        state = torch.load(ckpt_dir / ckpt_file, map_location="cpu",
                          weights_only=False)
        V = state["V"]

    return V, vocab


def _load_raw_embedding(source: str, d_gene: int = 10):
    """Load raw (untrained) gene factors and a matching vocab.

    Supports:
      bpmf:<pkl_path>   — BPMF posterior-mean V
      mf                — ALS matrix factorization
      mf:raw            — ALS without sphere projection
      svd               — truncated SVD (sphere-projected)
      genept            — GenePT PCA
      k562              — K562 Perturb-seq PCA
    """
    from assayloop.tasks import load_screens
    from assayloop.amortized.data import GeneVocab

    screens = load_screens(target_set="public_train")
    all_genes = set()
    for s in screens:
        all_genes.update(s.genes)
    # Also add test genes for coverage
    test_screens = load_screens(target_set="public")
    for s in test_screens:
        all_genes.update(s.genes)
    vocab = GeneVocab(sorted(all_genes))

    if source.startswith("bpmf:"):
        import pickle
        pkl_path = source.split(":", 1)[1]
        import glob
        matches = glob.glob(pkl_path)
        if matches:
            pkl_path = matches[-1]
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        bpmf_genes = result.gene_names
        bpmf_V = result.V_samples.mean(axis=0)  # (n_genes, K)
        gene_to_row = {g: i for i, g in enumerate(bpmf_genes)}
        V = np.zeros((len(vocab), d_gene), dtype=np.float32)
        for sym, vid in vocab.stoi.items():
            if vid == 0:
                continue
            row = gene_to_row.get(sym) or gene_to_row.get(sym.upper())
            if row is not None:
                V[vid] = bpmf_V[row, :d_gene].astype(np.float32)
        log.info("Raw BPMF: V=%s", V.shape)
    elif source.startswith("mf"):
        from assayloop.amortized.data import build_examples
        from assayloop.amortized.text_embed import get_text_embedder, embed_screens
        from assayloop.amortized.gene_factors import marginal_hit_freq, mf_vocab_factors
        embedder = get_text_embedder("auto")
        desc = embed_screens(screens, embedder)
        examples = build_examples(screens, desc, vocab)
        marginal = marginal_hit_freq(examples, vocab)
        normalize = ":raw" not in source
        V_np, bias = mf_vocab_factors(examples, vocab, d_gene, marginal,
                                       reg=0.1, normalize=normalize)
        V = V_np
        log.info("Raw MF%s: V=%s", " (sphere)" if normalize else " (raw)", V.shape)
    elif source == "svd":
        from assayloop.amortized.data import build_examples
        from assayloop.amortized.text_embed import get_text_embedder, embed_screens
        from assayloop.amortized.gene_factors import marginal_hit_freq, svd_vocab_factors
        embedder = get_text_embedder("auto")
        desc = embed_screens(screens, embedder)
        examples = build_examples(screens, desc, vocab)
        marginal = marginal_hit_freq(examples, vocab)
        V_np, bias = svd_vocab_factors(examples, vocab, d_gene, marginal)
        V = V_np
        log.info("Raw SVD: V=%s", V.shape)
    elif source == "genept":
        from assayloop.data.gene_embeddings.presage import genept_vocab_factors
        V_np, bias = genept_vocab_factors(vocab, d_gene, source="genept")
        V = V_np
        log.info("Raw GenePT PCA: V=%s", V.shape)
    elif source == "k562":
        from assayloop.data.gene_embeddings.presage import k562_perturbseq_vocab_factors
        V_np, bias = k562_perturbseq_vocab_factors(vocab, d_gene)
        V = V_np
        log.info("Raw K562 PCA: V=%s", V.shape)
    elif source == "random":
        rng = np.random.default_rng(42)
        V = rng.normal(0, 1, (len(vocab), d_gene)).astype(np.float32)
        V[0] = 0
        log.info("Random: V=%s", V.shape)
    else:
        raise ValueError("Unknown raw source: %s" % source)

    return V, vocab


def _load_string_edges(min_score: int = 400):
    """Load STRING experimental interactions as gene-gene pairs."""
    string_dir = config.ground_truth_dir("STRING-HUMAN")

    genes_df = pd.read_parquet(string_dir / "dimension_GeneSymbol.parquet")
    edges_df = pd.read_parquet(string_dir / "dimension_Interactions.parquet")

    id_to_gene = {k: v for k, v in zip(genes_df["string_human_id"], genes_df["gene_symbol"])
                  if isinstance(v, str)}
    gene_set = set(id_to_gene.values())

    pairs = set()
    for _, row in edges_df.iterrows():
        if row["experiment_score"] < min_score:
            continue
        g1 = id_to_gene.get(row["string_human_id_1"])
        g2 = id_to_gene.get(row["string_human_id_2"])
        if g1 and g2 and isinstance(g1, str) and isinstance(g2, str) and g1 != g2:
            pairs.add((min(g1, g2), max(g1, g2)))

    log.info("STRING: %d edges (score >= %d) over %d genes",
             len(pairs), min_score, len(gene_set))
    return pairs, gene_set


def _load_corum_edges():
    """Load CORUM complex co-membership as gene-gene pairs."""
    corum_dir = config.ground_truth_dir("CORUM-HUMAN")

    subunits = pd.read_parquet(corum_dir / "dimension_Subunit.parquet")
    gene_set = set()
    complexes = {}
    for _, row in subunits.iterrows():
        cid = row["complex_id"]
        gene = row.get("gene_name")
        if gene is None or not isinstance(gene, str) or not gene.strip():
            continue
        gene = gene.strip().upper()
        if gene:
            complexes.setdefault(cid, set()).add(gene)
            gene_set.add(gene)

    pairs = set()
    for members in complexes.values():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))

    log.info("CORUM: %d edges from %d complexes over %d genes",
             len(pairs), len(complexes), len(gene_set))
    return pairs, gene_set


def _eval_recovery(V, vocab, positive_pairs, gene_universe,
                   max_genes=5000, seed=42):
    """Compute AP and AUROC for recovering positive pairs from cosine similarity."""
    # Restrict to genes in both vocab and universe
    common = []
    for g in gene_universe:
        if not isinstance(g, str):
            continue
        idx = vocab.to_idx(g)
        if idx == 0:
            idx = vocab.to_idx(g.upper())
        if idx != 0:
            common.append((g, idx))

    if len(common) > max_genes:
        rng = np.random.default_rng(seed)
        # Keep genes that appear in positive pairs, sample rest
        pair_genes = set()
        for a, b in positive_pairs:
            pair_genes.add(a)
            pair_genes.add(b)
        keep = [(g, i) for g, i in common if g in pair_genes or g.upper() in pair_genes]
        rest = [(g, i) for g, i in common if g not in pair_genes and g.upper() not in pair_genes]
        n_sample = max_genes - len(keep)
        if n_sample > 0 and rest:
            sampled = rng.choice(len(rest), min(n_sample, len(rest)), replace=False)
            keep.extend([rest[i] for i in sampled])
        common = keep

    genes = [g for g, _ in common]
    indices = [i for _, i in common]
    gene_to_pos = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    if n < 10:
        return {"ap": 0, "auroc": 0, "n_genes": n, "n_pos": 0, "n_pairs": 0}

    # Get embeddings and compute cosine similarity
    E = V[indices].astype(np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    E_norm = E / norms
    sim = E_norm @ E_norm.T

    # Build labels: 1 for positive pairs, 0 for negatives
    labels = np.zeros((n, n), dtype=np.int32)
    n_pos = 0
    for a, b in positive_pairs:
        ia = gene_to_pos.get(a) or gene_to_pos.get(a.upper())
        ib = gene_to_pos.get(b) or gene_to_pos.get(b.upper())
        if ia is not None and ib is not None:
            labels[ia, ib] = 1
            labels[ib, ia] = 1
            n_pos += 1

    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(n, k=1)
    y_true = labels[triu_idx]
    y_score = sim[triu_idx]

    n_pairs = len(y_true)
    n_pos_triu = y_true.sum()

    if n_pos_triu < 5:
        return {"ap": 0, "auroc": 0, "n_genes": n, "n_pos": int(n_pos_triu),
                "n_pairs": n_pairs}

    ap = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)

    return {"ap": float(ap), "auroc": float(auroc), "n_genes": n,
            "n_pos": int(n_pos_triu), "n_pairs": n_pairs}


def _eval_influence_recovery(ckpt_name, ckpt_file, vocab, positive_pairs,
                             gene_universe, device, max_genes=500,
                             n_samples=10, n_background=30, seed=42):
    """Compute AP/AUROC from context-conditional influence similarity.

    For each gene pair (A, B): influence = mean score change for B when A
    is observed as a hit. Symmetrize as (inf_AB + inf_BA) / 2.
    """
    import torch
    from assayloop.amortized.model import RankerConfig, RankerNet
    from assayloop.amortized.text_embed import get_text_embedder, embed_texts

    ckpt_dir = RANKERS_DIR / ckpt_name
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    if "arch" not in cfg:
        return None

    net = RankerNet(RankerConfig(**cfg["arch"]))
    state = torch.load(ckpt_dir / ckpt_file, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    net.load_state_dict(state)
    net.to(device).eval()

    text_backend = cfg.get("text_backend", "auto")
    embedder = get_text_embedder(text_backend)
    text = "A genome-wide CRISPR knockout screen to identify essential genes."
    emb = embed_texts([text], embedder)[0]
    desc_emb = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)

    V = net.gene_emb.weight.detach()

    # Restrict to genes in both vocab and universe
    rng = np.random.default_rng(seed)
    common = []
    for g in gene_universe:
        if not isinstance(g, str):
            continue
        idx = vocab.to_idx(g)
        if idx == 0:
            idx = vocab.to_idx(g.upper())
        if idx != 0:
            common.append((g, idx))

    # Prioritize genes in positive pairs, sample the rest
    pair_genes = set()
    for a, b in positive_pairs:
        pair_genes.add(a)
        pair_genes.add(b)
    keep = [(g, i) for g, i in common if g in pair_genes or g.upper() in pair_genes]
    rest = [(g, i) for g, i in common if g not in pair_genes and g.upper() not in pair_genes]
    n_sample = max_genes - len(keep)
    if n_sample > 0 and rest:
        sampled = rng.choice(len(rest), min(n_sample, len(rest)), replace=False)
        keep.extend([rest[i] for i in sampled])
    common = keep

    genes = [g for g, _ in common]
    indices = [i for _, i in common]
    gene_to_pos = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    n_vocab = len(vocab)

    if n < 10:
        return {"ap": 0, "auroc": 0, "n_genes": n, "n_pos": 0}

    log.info("    Computing influence matrix for %d genes (%d samples)...", n, n_samples)

    # Compute influence matrix: inf[i, j] = score change for gene j when gene i is hit
    all_ids = [idx for idx in range(1, n_vocab)]
    inf_matrix = np.zeros((n, n), dtype=np.float32)

    for i, (g_probe, id_probe) in enumerate(zip(genes, indices)):
        if (i + 1) % 50 == 0:
            log.info("      %d/%d genes probed", i + 1, n)

        delta_acc = torch.zeros(n, device=device)
        for _ in range(n_samples):
            bg_ids = rng.choice(all_ids, size=min(n_background, len(all_ids)),
                                replace=False).tolist()
            bg_ids = [x for x in bg_ids if x != id_probe]
            bg_hits = rng.integers(0, 2, size=len(bg_ids)).tolist()

            ctx_ids = torch.tensor([bg_ids], device=device)
            ctx_hits = torch.tensor([bg_hits], device=device)
            ctx_pad = torch.zeros(1, len(bg_ids), device=device)

            with torch.no_grad():
                repr_base = net.encode(desc_emb, ctx_ids, ctx_hits, ctx_pad)

            ctx_ids_p = torch.cat([ctx_ids, torch.tensor([[id_probe]], device=device)], dim=1)
            ctx_hits_p = torch.cat([ctx_hits, torch.tensor([[1]], device=device)], dim=1)
            ctx_pad_p = torch.zeros(1, ctx_ids_p.shape[1], device=device)

            with torch.no_grad():
                repr_probe = net.encode(desc_emb, ctx_ids_p, ctx_hits_p, ctx_pad_p)

            delta_repr = repr_probe - repr_base  # (1, d_gene)
            target_ids = torch.tensor(indices, device=device)
            delta_scores = V[target_ids] @ delta_repr.squeeze(0)  # (n,)
            delta_acc += delta_scores

        inf_matrix[i] = (delta_acc / n_samples).cpu().numpy()

    # Symmetrize
    sym_inf = (inf_matrix + inf_matrix.T) / 2

    # Build labels
    labels = np.zeros((n, n), dtype=np.int32)
    n_pos = 0
    for a, b in positive_pairs:
        ia = gene_to_pos.get(a) or gene_to_pos.get(a.upper())
        ib = gene_to_pos.get(b) or gene_to_pos.get(b.upper())
        if ia is not None and ib is not None:
            labels[ia, ib] = 1
            labels[ib, ia] = 1
            n_pos += 1

    triu_idx = np.triu_indices(n, k=1)
    y_true = labels[triu_idx]
    y_score = sym_inf[triu_idx]
    n_pos_triu = y_true.sum()

    if n_pos_triu < 5:
        return {"ap": 0, "auroc": 0, "n_genes": n, "n_pos": int(n_pos_triu)}

    ap_val = average_precision_score(y_true, y_score)
    auroc_val = roc_auc_score(y_true, y_score)

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"ap": float(ap_val), "auroc": float(auroc_val),
            "n_genes": n, "n_pos": int(n_pos_triu)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="",
                    help="Comma-sep checkpoint names (optionally name:ckpt_file)")
    ap.add_argument("--string-min-score", type=int, default=400,
                    help="Minimum STRING experimental score")
    ap.add_argument("--max-genes", type=int, default=5000,
                    help="Max genes for pairwise similarity (memory)")
    ap.add_argument("--influence", action="store_true",
                    help="Also compute influence-based recovery (slow, needs GPU)")
    ap.add_argument("--influence-max-genes", type=int, default=500,
                    help="Max genes for influence matrix (O(n²) forward passes)")
    ap.add_argument("--raw-sources", default=None,
                    help="Comma-sep raw embedding sources to compare "
                         "(e.g., random,bpmf:output/bpmf/bpmf_*.pkl,mf,mf:raw,svd,genept,k562)")
    ap.add_argument("--d-gene", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Parse checkpoints
    ckpts = []
    for entry in args.checkpoints.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, ckpt_file = entry.split(":", 1)
        else:
            name, ckpt_file = entry, "model.pt"
        ckpts.append((name, ckpt_file))

    # Load ground truth
    string_pairs, string_genes = _load_string_edges(args.string_min_score)
    corum_pairs, corum_genes = _load_corum_edges()

    ground_truths = []
    if string_pairs:
        ground_truths.append(("STRING", string_pairs, string_genes))
    if corum_pairs:
        ground_truths.append(("CORUM", corum_pairs, corum_genes))

    if not ground_truths:
        log.error("No ground truth loaded")
        return

    # Evaluate each checkpoint
    results = []
    for name, ckpt_file in ckpts:
        ckpt_path = RANKERS_DIR / name
        if not ckpt_path.exists():
            log.warning("Checkpoint %s not found, skipping", name)
            continue

        log.info("=== %s (%s) ===", name, ckpt_file)
        V, vocab = _load_gene_factors(name, ckpt_file)
        log.info("  V shape: %s, vocab: %d", V.shape, len(vocab))

        row = {"checkpoint": name, "ckpt_file": ckpt_file,
               "d_gene": V.shape[1]}

        for gt_name, pairs, gene_set in ground_truths:
            log.info("  Evaluating vs %s (%d pairs, %d genes)...",
                     gt_name, len(pairs), len(gene_set))
            r = _eval_recovery(V, vocab, pairs, gene_set,
                               max_genes=args.max_genes)
            row["%s_ap" % gt_name] = r["ap"]
            row["%s_auroc" % gt_name] = r["auroc"]
            row["%s_n_genes" % gt_name] = r["n_genes"]
            row["%s_n_pos" % gt_name] = r["n_pos"]
            log.info("    Embedding: AP=%.4f  AUROC=%.4f  (%d genes, %d pos pairs)",
                     r["ap"], r["auroc"], r["n_genes"], r["n_pos"])

            # Influence-based recovery
            if args.influence:
                import torch
                dev = args.device
                if dev == "auto":
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("    Computing influence-based recovery...")
                inf_r = _eval_influence_recovery(
                    name, ckpt_file, vocab, pairs, gene_set,
                    device=torch.device(dev),
                    max_genes=args.influence_max_genes)
                if inf_r is not None:
                    row["%s_inf_ap" % gt_name] = inf_r["ap"]
                    row["%s_inf_auroc" % gt_name] = inf_r["auroc"]
                    log.info("    Influence: AP=%.4f  AUROC=%.4f  (%d genes, %d pos pairs)",
                             inf_r["ap"], inf_r["auroc"], inf_r["n_genes"], inf_r["n_pos"])
                else:
                    log.info("    Influence: skipped (no encoder arch)")

        results.append(row)

    # Evaluate raw embedding sources (no trained model, embedding-only)
    if args.raw_sources:
        for src in args.raw_sources.split(","):
            src = src.strip()
            if not src:
                continue
            log.info("=== raw:%s ===", src)
            try:
                V_raw, vocab_raw = _load_raw_embedding(src, d_gene=args.d_gene)
            except Exception as e:
                log.warning("  Failed to load raw source %s: %s", src, e)
                continue

            row = {"checkpoint": "raw:%s" % src, "ckpt_file": "-",
                   "d_gene": V_raw.shape[1]}

            for gt_name, pairs, gene_set in ground_truths:
                log.info("  Evaluating vs %s...", gt_name)
                r = _eval_recovery(V_raw, vocab_raw, pairs, gene_set,
                                   max_genes=args.max_genes)
                row["%s_ap" % gt_name] = r["ap"]
                row["%s_auroc" % gt_name] = r["auroc"]
                row["%s_n_genes" % gt_name] = r["n_genes"]
                row["%s_n_pos" % gt_name] = r["n_pos"]
                log.info("    AP=%.4f  AUROC=%.4f  (%d genes, %d pos pairs)",
                         r["ap"], r["auroc"], r["n_genes"], r["n_pos"])

            results.append(row)

    # Save results
    out_dir = config.OUTPUT_PATH / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "gene_network_recovery.json"
    results_file.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", results_file)

    # Print table
    print()
    gt_names = [gt[0] for gt in ground_truths]
    header = "%-40s" % "Checkpoint"
    for gt in gt_names:
        header += "  %s_emb_AP  %s_emb_AUC" % (gt, gt)
        if args.influence:
            header += "  %s_inf_AP  %s_inf_AUC" % (gt, gt)
    print(header)
    print("-" * len(header))
    for row in results:
        line = "%-40s" % row["checkpoint"]
        for gt in gt_names:
            line += "  %9.4f  %9.4f" % (row.get("%s_ap" % gt, 0),
                                          row.get("%s_auroc" % gt, 0))
            if args.influence:
                line += "  %9.4f  %9.4f" % (row.get("%s_inf_ap" % gt, 0),
                                              row.get("%s_inf_auroc" % gt, 0))
        print(line)

    # Plot
    fig, axes = plt.subplots(1, len(gt_names), figsize=(5 * len(gt_names), 5),
                              squeeze=False)
    ckpt_labels = [r["checkpoint"].replace("gf-", "").replace("-train-", "\n")
                   for r in results]
    x = np.arange(len(results))

    for col, gt in enumerate(gt_names):
        ax = axes[0][col]
        aps = [r.get("%s_ap" % gt, 0) for r in results]
        aurocs = [r.get("%s_auroc" % gt, 0) for r in results]
        w = 0.35
        ax.bar(x - w/2, aps, w, label="AP", color="C0")
        ax.bar(x + w/2, aurocs, w, label="AUROC", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels(ckpt_labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(gt, fontsize=12)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Gene Network Recovery from Learned Embeddings", fontsize=13)
    fig.tight_layout()
    fname = out_dir / "gene_network_recovery.png"
    fig.savefig(fname, dpi=150)
    log.info("Wrote %s", fname)


if __name__ == "__main__":
    main()
