"""Visualize leave-one-phenotype-out (LOPO) cross-validation results.

Reads summary.json and history.json from each LOPO fold's supervised and RL
output dirs, then produces:

  1. Grouped bar chart of NVR per fold (kNN / supervised / RL split-half).
  2. RL training curves (eval NVR over epochs, per fold).
  3. Printed + CSV summary table.

Usage::

    uv run python -m assayloop.scripts.plot_lopo_results
    uv run python -m assayloop.scripts.plot_lopo_results --rankers-dir output/rankers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from assayloop import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_lopo")

FOLDS = ["fitness", "drug", "infection", "molecular", "trafficking"]
FOLD_LABELS = {
    "fitness": "Fitness",
    "drug": "Drug",
    "infection": "Infection",
    "molecular": "Molecular",
    "trafficking": "Trafficking",
}


def _load_fold_test_screens(fold):
    """Load the held-out test screens for a LOPO fold from its split manifest.

    Matches the LOPO protocol: each fold's test set is ALL screens of the
    held-out phenotype from the full 1901-screen public pool (train +
    validation + test yearfold0 partitions), NOT the 20-screen public set.

    The manifests ship in :mod:`assaybench.data.screen_sets`. An unknown fold
    name raises there; this used to return ``[]`` for a missing file, which
    dropped the fold from the figure without saying so.
    """
    from assayloop.tasks import load_screens
    return load_screens(target_set="lopo-%s-test" % fold)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _eval_checkpoint_universe(ckpt_name, ckpt_file, eval_screens, universe, device="cpu"):
    """Re-evaluate a checkpoint on screens with universe candidates + domain-adjusted NVR."""
    from assayloop.models.amortized_ranker import AmortizedRankerModel
    from assayloop.experiment.runner import RunConfig, run_one_screen
    from assayloop.scripts.full_genome_table import _adj_nvr_from_run
    import torch

    ckpt_dir = Path(ckpt_name) if Path(ckpt_name).is_absolute() else config.OUTPUT_PATH / "rankers" / ckpt_name
    if not (ckpt_dir / ckpt_file).exists():
        return None

    model = AmortizedRankerModel(checkpoint=str(ckpt_dir), ckpt_file=ckpt_file,
                                 device=device)
    cfg = RunConfig(
        screen_set="custom", model="null", acq="greedy",
        batch_size=100, n_steps=10, persist=True,
        metrics=["hits_auc"], max_shortfall_frac=1.0,
        universe_genes=universe,
    )

    runs_dir = config.OUTPUT_PATH / "runs"
    universe_set = set(universe)
    nvrs = []
    for i, screen in enumerate(eval_screens):
        sweep_id = "lopo-eval-%s" % ckpt_dir.name
        run_id = "%s-%02d-%s" % (sweep_id, i, screen.dataset_name)
        cached = runs_dir / run_id / "result.json"
        if cached.is_file():
            adj = _adj_nvr_from_run(cached, set(screen.genes), universe_set,
                                    sum(screen.hits), len(screen.genes), 1000)
            nvrs.append(adj)
        else:
            res = run_one_screen(screen, cfg, model_obj=model, verbose=False,
                                run_id=run_id, sweep_id=sweep_id)
            fm = res.final_metrics or {}
            # Recompute adjusted NVR
            fp = runs_dir / run_id / "result.json"
            if fp.exists():
                adj = _adj_nvr_from_run(fp, set(screen.genes), universe_set,
                                        sum(screen.hits), len(screen.genes), 1000)
                nvrs.append(adj)
            else:
                nvrs.append(fm.get("n_hits_vs_random", 0))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(np.mean(nvrs)) if nvrs else None


def _fold_train_screens(fold):
    """The fold's TRAINING screens (all other phenotypes) — the pool the
    nearest-screen kNN retrieves from, matching what the fold model saw."""
    from assayloop.tasks import load_screens
    return load_screens(target_set="lopo-%s-train" % fold)


# ---- kNN baselines evaluated on the f2 universe (domain-adjusted NVR) ----
# Both run the same 100x10 greedy AL rollout over the universe candidates as the
# transformer, then score with adjusted_nvr_value (the full-genome-table metric).
_KNN_TAU, _KNN_KAPPA = 0.1, 100.0


def _build_screen_knn_matrices(train_screens, universe):
    """Vectorized ScreenKNN state: hit / measured matrices (S x G) + hit prior.

    Verified to reproduce ``ScreenKNNModel`` exactly (diff 0) at ~800x its speed.
    """
    gi = {g: i for i, g in enumerate(universe)}
    S, G = len(train_screens), len(universe)
    H = np.zeros((S, G), np.float32)
    M = np.zeros((S, G), np.float32)
    hc = np.zeros(G)
    tc = np.zeros(G)
    for si, s in enumerate(train_screens):
        for g, h in zip(s.genes, s.hits):
            j = gi.get(g)
            if j is None:
                continue
            M[si, j] = 1.0
            tc[j] += 1
            if h:
                H[si, j] = 1.0
                hc[j] += 1
    prior = np.where(tc > 0, hc / np.maximum(tc, 1), 0.0).astype(np.float32)
    return H, M, prior


def _screen_knn_rollout(screen, universe, H, M, prior):
    """Nearest-SCREEN kNN (the table's baseline) rollout -> adjusted NVR."""
    from assayloop.metrics.hits_auc import adjusted_nvr_value
    G = len(universe)
    lib = set(screen.genes)
    hit_of = {g: bool(h) for g, h in zip(screen.genes, screen.hits)}
    picked = np.zeros(G, bool)
    rev_idx, rev_val = [], []
    n1 = n2 = hits = 0
    for _ in range(10):
        nrev = len(rev_idx)
        lam = nrev / (nrev + _KNN_KAPPA) if nrev > 0 else 0.0
        if nrev == 0:
            w = np.full(H.shape[0], np.exp(-1.0 / _KNN_TAU), np.float32)
        else:
            R = np.array(rev_idx)
            r = np.array(rev_val, np.float32)
            MR = M[:, R]
            cnt = MR.sum(1)
            sq = ((r[None, :] - H[:, R]) ** 2 * MR).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.where(cnt > 0, sq / np.maximum(cnt, 1), np.nan)
            w = np.where(cnt > 0, np.exp(-dist / _KNN_TAU),
                         np.exp(-1.0 / _KNN_TAU)).astype(np.float32)
        wM = w @ M
        wH = w @ H
        with np.errstate(divide="ignore", invalid="ignore"):
            eviden = np.where(wM > 0, wH / np.maximum(wM, 1e-12), 0.0)
        score = np.where(wM > 0, (1 - lam) * prior + lam * eviden, prior).astype(np.float64)
        score[picked] = -np.inf
        top = np.argpartition(-score, 100 - 1)[:100]
        for idx in top:
            picked[idx] = True
            name = universe[idx]
            if name in lib:
                n1 += 1
                is_hit = hit_of[name]
                hits += int(is_hit)
                rev_idx.append(idx); rev_val.append(1.0 if is_hit else 0.0)
            else:
                n2 += 1
                rev_idx.append(idx); rev_val.append(0.0)
    return adjusted_nvr_value(hits, n1, n2, len(lib), sum(screen.hits), 1000)


def _knn_gene_rollout(screen, cand_names, emb, bias):
    """Nearest-GENE kNN over the model's learned embeddings (max cosine sim to
    observed hits; cold start = per-gene bias) rollout -> adjusted NVR."""
    from assayloop.metrics.hits_auc import adjusted_nvr_value
    G = len(cand_names)
    lib = set(screen.genes)
    hit_of = {g: bool(h) for g, h in zip(screen.genes, screen.hits)}
    picked = np.zeros(G, bool)
    hit_embs = []
    n1 = n2 = hits = 0
    for _ in range(10):
        if hit_embs:
            score = (emb @ np.stack(hit_embs).T).max(1).astype(np.float64)
        else:
            score = bias.astype(np.float64).copy()
        score[picked] = -np.inf
        top = np.argpartition(-score, 100 - 1)[:100]
        for idx in top:
            picked[idx] = True
            name = cand_names[idx]
            if name in lib:
                n1 += 1
                if hit_of[name]:
                    hits += 1
                    hit_embs.append(emb[idx])
            else:
                n2 += 1
    return adjusted_nvr_value(hits, n1, n2, len(screen.genes), sum(screen.hits), 1000)


def _eval_knn_screen_universe(fold, eval_screens, universe):
    train = _fold_train_screens(fold)
    if not train:
        log.warning("Fold %s: no train YAML for nearest-screen kNN", fold)
        return None
    H, M, prior = _build_screen_knn_matrices(train, universe)
    return [_screen_knn_rollout(s, universe, H, M, prior) for s in eval_screens]


def _eval_knn_gene_universe(sup_dir, eval_screens, universe, device="cpu"):
    import torch
    from assayloop.amortized.data import GeneVocab
    from assayloop.amortized.model import RankerConfig, RankerNet
    ckpt_dir = Path(sup_dir)
    if not (ckpt_dir / "model.pt").exists():
        return None
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    vocab = GeneVocab.load(ckpt_dir / "vocab.json")
    net = RankerNet(RankerConfig(**cfg["arch"]))
    net.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu",
                                   weights_only=False))
    net.eval()
    V = net.gene_emb.weight.detach().cpu().numpy()
    bias_all = net.gene_bias.detach().cpu().numpy()
    cand_names, cand_ids = [], []
    for g in universe:
        vid = vocab.to_idx(g) or vocab.to_idx(g.upper())
        if vid:
            cand_names.append(g)
            cand_ids.append(vid)
    emb = V[cand_ids]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    bias = bias_all[cand_ids]
    return [_knn_gene_rollout(s, cand_names, emb, bias) for s in eval_screens]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankers-dir", default=str(config.OUTPUT_PATH / "rankers"))
    ap.add_argument("--out-dir", default=str(config.OUTPUT_PATH / "analysis"))
    ap.add_argument("--prefix", default="lopo", help="Run-name prefix for the LOPO folds.")
    ap.add_argument("--universe", action="store_true",
                    help="Re-evaluate with filtered universe + domain-adjusted NVR")
    ap.add_argument("--min-screen-freq", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recompute-knn", action="store_true",
                    help="Ignore the cached f2-universe kNN evals and recompute.")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rankers = Path(args.rankers_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build universe if needed
    universe = None
    if args.universe:
        from assayloop.tasks import load_screens
        from collections import Counter
        all_screens = load_screens(target_set="public")
        gene_freq = Counter()
        for s in all_screens:
            for g in set(s.genes):
                gene_freq[g] += 1
        universe = sorted(g for g in gene_freq if gene_freq[g] >= args.min_screen_freq)
        log.info("Universe: %d genes (freq >= %d)", len(universe), args.min_screen_freq)

    # cache for the (expensive-once) f2-universe kNN evals
    knn_cache_path = out_dir / "lopo_knn_universe.json"
    knn_cache = {}
    if args.universe and knn_cache_path.exists() and not args.recompute_knn:
        knn_cache = _load(knn_cache_path)

    rows: list[dict] = []
    histories: dict[str, list] = {}

    for fold in FOLDS:
        sup_dir = rankers / f"{args.prefix}-{fold}-sup"
        rl_dir = rankers / f"{args.prefix}-{fold}-rl"
        if not (sup_dir / "summary.json").exists():
            log.warning("Missing %s — skipping fold %s", sup_dir, fold)
            continue
        if not (rl_dir / "summary.json").exists():
            log.warning("Missing %s — skipping fold %s", rl_dir, fold)
            continue

        sup_summary = _load(sup_dir / "summary.json")
        rl_summary = _load(rl_dir / "summary.json")
        rl_hist = _load(rl_dir / "history.json")

        if args.universe and universe:
            # Re-evaluate on the fold's actual held-out test set (all screens of
            # the held-out phenotype from the full 1901-screen pool). The fold
            # model was trained on the OTHER phenotypes, so there is no leakage.
            eval_screens = _load_fold_test_screens(fold)
            if not eval_screens:
                log.warning("Fold %s: no test YAML found; skipping", fold)
                continue
            log.info("Fold %s: re-evaluating sup + rl on %d held-out test screens...",
                     fold, len(eval_screens))
            sup_nvr = _eval_checkpoint_universe(
                str(sup_dir), "model.pt", eval_screens, universe, device)
            rl_nvr = _eval_checkpoint_universe(
                str(rl_dir), "model.pt", eval_screens, universe, device)

            # Two kNN baselines on the SAME f2 universe + adjusted NVR:
            #   knn_gene   = nearest-gene in the fold's learned embeddings
            #   knn_screen = nearest-screen (the LaTeX-table baseline), retrieving
            #                only from the fold's training screens (fair LOPO)
            cached = knn_cache.get(fold, {})
            if cached.get("knn_gene_vals") is not None:
                kg_vals = cached["knn_gene_vals"]
            else:
                log.info("Fold %s: nearest-gene kNN over f2 universe...", fold)
                kg_vals = _eval_knn_gene_universe(str(sup_dir), eval_screens,
                                                  universe, device)
            if cached.get("knn_screen_vals") is not None:
                ks_vals = cached["knn_screen_vals"]
            else:
                log.info("Fold %s: nearest-screen kNN over f2 universe...", fold)
                ks_vals = _eval_knn_screen_universe(fold, eval_screens, universe)
            knn_cache[fold] = {"knn_gene_vals": kg_vals, "knn_screen_vals": ks_vals}
            knn_gene = float(np.mean(kg_vals)) if kg_vals else None
            knn_screen = float(np.mean(ks_vals)) if ks_vals else None

            rows.append({
                "fold": fold,
                "label": FOLD_LABELS.get(fold, fold),
                "n_test": len(eval_screens),
                "knn": knn_gene,
                "knn_gene": knn_gene,
                "knn_screen": knn_screen,
                "sup": sup_nvr,
                "rl_split_half": rl_nvr,
                "rl_best": rl_nvr,
                "rl_init": sup_nvr,
            })
        else:
            rows.append({
                "fold": fold,
                "label": FOLD_LABELS.get(fold, fold),
                "n_test": rl_summary.get("n_eval", 0),
                "knn": rl_summary.get("init_eval_knn"),
                "sup": sup_summary.get("best_val_n_hits_vs_random"),
                "rl_split_half": rl_summary.get("split_half_nvr"),
            "rl_best": rl_summary.get("best_eval_n_hits_vs_random"),
            "rl_init": rl_summary.get("init_eval_n_hits_vs_random"),
        })
        histories[fold] = rl_hist

    if not rows:
        log.error("No LOPO folds found under %s", rankers)
        return

    if args.universe:
        knn_cache_path.write_text(json.dumps(knn_cache, indent=2))
        log.info("Wrote %s", knn_cache_path)

    # whether we have the two-flavour f2-universe kNN bars
    two_knn = args.universe and all("knn_screen" in r for r in rows)

    # ---- CSV ----
    csv_path = out_dir / "lopo_summary.csv"
    fieldnames = ["fold", "label", "n_test", "knn", "sup", "rl_split_half",
                  "rl_best", "rl_lift_vs_sup", "rl_lift_vs_knn"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["rl_lift_vs_sup"] = (r["rl_split_half"] or 0) - (r["sup"] or 0)
            r2["rl_lift_vs_knn"] = (r["rl_split_half"] or 0) - (r["knn"] or 0)
            w.writerow(r2)
    log.info("Wrote %s", csv_path)

    # ---- printed table ----
    print(f"\n{'Fold':<14s} {'N':>5s} {'kNN':>7s} {'Sup':>7s} {'RL(sh)':>7s} "
          f"{'RL-Sup':>7s} {'RL-kNN':>7s}")
    print("-" * 62)
    for r in rows:
        rl_sh = r["rl_split_half"] or 0
        sup_v = r["sup"] or 0
        knn_v = r["knn"] or 0
        print(f"{r['label']:<14s} {r['n_test']:>5d} {knn_v:>7.3f} {sup_v:>7.3f} "
              f"{rl_sh:>7.3f} {rl_sh - sup_v:>+7.3f} {rl_sh - knn_v:>+7.3f}")
    # Mean row
    mean_knn = np.mean([r["knn"] for r in rows if r["knn"] is not None])
    mean_sup = np.mean([r["sup"] for r in rows if r["sup"] is not None])
    mean_rl = np.mean([r["rl_split_half"] for r in rows if r["rl_split_half"] is not None])
    print("-" * 62)
    print(f"{'Mean':<14s} {'':>5s} {mean_knn:>7.3f} {mean_sup:>7.3f} "
          f"{mean_rl:>7.3f} {mean_rl - mean_sup:>+7.3f} {mean_rl - mean_knn:>+7.3f}")

    # ---- Figure 1: grouped bar chart ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows] + ["Mean"]
    sup_vals = [r["sup"] or 0 for r in rows] + [mean_sup]
    rl_vals = [r["rl_split_half"] or 0 for r in rows] + [mean_rl]
    x = np.arange(len(labels))

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    if two_knn:
        # four bars: kNN (gene) | kNN (screen) | Supervised | RL
        mean_kg = np.mean([r["knn_gene"] for r in rows if r["knn_gene"] is not None])
        mean_ks = np.mean([r["knn_screen"] for r in rows if r["knn_screen"] is not None])
        kg_vals = [r["knn_gene"] or 0 for r in rows] + [mean_kg]
        ks_vals = [r["knn_screen"] or 0 for r in rows] + [mean_ks]
        width = 0.2
        series = [
            (kg_vals, "kNN (gene embedding)", "tab:gray", -1.5),
            (ks_vals, "kNN (nearest screen)", "tab:brown", -0.5),
            (sup_vals, "Supervised", "tab:blue", 0.5),
            (rl_vals, "RL", "tab:orange", 1.5),
        ]
    else:
        knn_vals = [r["knn"] or 0 for r in rows] + [mean_knn]
        width = 0.25
        series = [
            (knn_vals, "kNN", "tab:gray", -1.0),
            (sup_vals, "Supervised", "tab:blue", 0.0),
            (rl_vals, "RL", "tab:orange", 1.0),
        ]
    for vals, lab, color, off in series:
        bars = ax1.bar(x + off * width, vals, width, label=lab, color=color)
        ax1.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Enrichment Factor at 100×10 steps")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    # Separator before Mean
    ax1.axvline(x[-1] - 0.5, color="grey", ls=":", lw=0.8, alpha=0.6)
    fig1.tight_layout()
    fig1_path = out_dir / "lopo_bar_chart.png"
    fig1.savefig(fig1_path, dpi=150)
    fig1.savefig(out_dir / "lopo_bar_chart.pdf", bbox_inches="tight")
    log.info("Wrote %s(.png/.pdf)", fig1_path)

    # ---- Figure 2: RL training curves ----
    n_folds = len(histories)
    if n_folds > 0:
        fig2, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4), sharey=False)
        if n_folds == 1:
            axes = [axes]
        for ax, fold in zip(axes, FOLDS):
            if fold not in histories:
                continue
            hist = histories[fold]
            r = next(r for r in rows if r["fold"] == fold)

            eval_entries = [h for h in hist
                           if "eval" in h and isinstance(h["eval"], dict)
                           and "epoch" in h]
            epochs = [h["epoch"] for h in eval_entries]
            nvr_full = [h["eval"]["n_hits_vs_random"] for h in eval_entries]
            nvr_a = [h.get("eval_half_a_nvr") for h in eval_entries]
            nvr_b = [h.get("eval_half_b_nvr") for h in eval_entries]

            ax.plot(epochs, nvr_full, color="tab:orange", lw=1.5, label="eval (full)")
            if any(v is not None for v in nvr_a):
                ax.plot(epochs, nvr_a, color="tab:orange", lw=0.8, ls="--", alpha=0.5, label="half-A")
                ax.plot(epochs, nvr_b, color="tab:cyan", lw=0.8, ls="--", alpha=0.5, label="half-B")

            if r["rl_init"] is not None:
                ax.axhline(r["rl_init"], color="tab:blue", ls=":", lw=1, alpha=0.7, label="sup init")
            if r["knn"] is not None:
                ax.axhline(r["knn"], color="tab:gray", ls=":", lw=1, alpha=0.7, label="kNN")

            ax.set_xlabel("Epoch")
            ax.set_title(f"{FOLD_LABELS[fold]} (n={r['n_test']})", fontsize=10)
            ax.grid(True, alpha=0.2)
            if fold == FOLDS[0]:
                ax.set_ylabel("n_hits_vs_random")
                ax.legend(fontsize=7, loc="lower right")

        fig2.suptitle("LOPO RL training curves", fontsize=12)
        fig2.tight_layout()
        fig2_path = out_dir / "lopo_training_curves.png"
        fig2.savefig(fig2_path, dpi=150)
        log.info("Wrote %s", fig2_path)


if __name__ == "__main__":
    main()
