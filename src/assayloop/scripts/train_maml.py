"""MAML meta-training for the gene ranking baseline.

Each training screen is a "task". The outer loop learns an MLP that maps
screen descriptions to initial screen latents. The inner loop adapts the
latent via gradient descent on revealed (gene, hit) pairs. The outer loss
is evaluated on held-out genes using the adapted latent.

Usage::

    uv run python -m assayloop.scripts.train_maml \
        --run-name maml-bpmf-d10 --device cuda

    uv run python -m assayloop.scripts.train_maml \
        --run-name maml-bpmf-d10 --device cuda \
        --inner-steps 5 --inner-lr 0.01 --epochs 40
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from assayloop import config
from assayloop.amortized.data import GeneVocab, build_examples
from assayloop.amortized.text_embed import get_text_embedder, embed_screens
from assayloop.models.maml_ranker import ScreenMLP
from assayloop.tasks import load_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_maml")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"


def _load_bpmf_factors(pkl_path: str, vocab: GeneVocab, d_gene: int):
    """Load BPMF posterior-mean V and align to vocab."""
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    bpmf_genes = result.gene_names
    bpmf_V = result.V_samples.mean(axis=0)  # (n_genes, K)
    gene_to_row = {g: i for i, g in enumerate(bpmf_genes)}

    n = len(vocab)
    V = np.zeros((n, d_gene), dtype=np.float32)
    covered = 0
    for sym, vid in vocab.stoi.items():
        if vid == 0:
            continue
        row = gene_to_row.get(sym) or gene_to_row.get(sym.upper())
        if row is not None:
            V[vid] = bpmf_V[row, :d_gene].astype(np.float32)
            covered += 1
    log.info("BPMF factors: %d/%d vocab genes covered, K=%d->d_gene=%d",
             covered, n, bpmf_V.shape[1], d_gene)
    return V


def _eval_on_screens(screen_mlp, examples, V, bias, device,
                     inner_steps, inner_lr, context_frac=0.25):
    """Evaluate adapted NVR on a set of screens (no outer gradients)."""
    screen_mlp.eval()
    nvrs = []
    for ex in examples:
        n_genes = len(ex.gene_idx)
        n_ctx = max(1, int(n_genes * context_frac))
        perm = np.random.permutation(n_genes)
        ctx_idx = perm[:n_ctx]
        query_idx = perm[n_ctx:]
        if len(query_idx) == 0:
            continue

        desc_emb = torch.tensor(ex.desc_emb, dtype=torch.float32,
                               device=device).unsqueeze(0)
        with torch.no_grad():
            u0 = screen_mlp(desc_emb).squeeze(0)

        gene_ids = torch.tensor(ex.gene_idx, device=device)
        hits = torch.tensor(ex.hit, dtype=torch.float32, device=device)

        # Inner loop on context
        ctx_gene_ids = gene_ids[ctx_idx]
        ctx_hits = hits[ctx_idx]
        V_ctx = V[ctx_gene_ids]
        bias_ctx = bias[ctx_gene_ids]

        u = u0.clone().detach().requires_grad_(True)
        for _ in range(inner_steps):
            logits = V_ctx @ u + bias_ctx
            loss = F.binary_cross_entropy_with_logits(logits, ctx_hits)
            grad = torch.autograd.grad(loss, u)[0]
            u = u - inner_lr * grad

        # Score query genes
        query_gene_ids = gene_ids[query_idx]
        query_hits_q = hits[query_idx]
        V_q = V[query_gene_ids]
        bias_q = bias[query_gene_ids]
        with torch.no_grad():
            scores = V_q @ u.detach() + bias_q
            n_hits = query_hits_q.sum().item()
            if n_hits > 0:
                top_k = min(100, len(query_idx))
                _, topk_idx = scores.topk(top_k)
                hits_in_top = query_hits_q[topk_idx].sum().item()
                rand_exp = top_k * n_hits / len(query_idx)
                nvr = hits_in_top / rand_exp if rand_exp > 0 else 0
                nvrs.append(float(nvr))
    return float(np.mean(nvrs)) if nvrs else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--bpmf-pkl", default=None,
                    help="Path to BPMF result pkl. Auto-detected if not set.")
    ap.add_argument("--d-gene", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128, help="MLP hidden dim")
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--context-frac", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Screens per outer-loop update")
    ap.add_argument("--train-screen-set", default="public_train")
    ap.add_argument("--val-screen-set", default="public_validation")
    ap.add_argument("--text-backend", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wandb-project", default="assayloop-amortized-ranker")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--wandb-tags", default="maml")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load screens
    log.info("Loading screens...")
    train_screens = load_screens(target_set=args.train_screen_set)
    val_screens = load_screens(target_set=args.val_screen_set)

    # Build vocab from training screens
    all_genes = set()
    for s in train_screens + val_screens:
        all_genes.update(s.genes)
    vocab = GeneVocab(sorted(all_genes))
    log.info("Vocab: %d genes", len(vocab))

    # Get text embeddings
    embedder = get_text_embedder(args.text_backend)
    train_desc = embed_screens(train_screens, embedder)
    val_desc = embed_screens(val_screens, embedder)
    text_dim = next(iter(train_desc.values())).shape[0]

    # Build examples
    train_examples = build_examples(train_screens, train_desc, vocab)
    val_examples = build_examples(val_screens, val_desc, vocab)
    log.info("Train: %d screens, Val: %d screens", len(train_examples), len(val_examples))

    # Load BPMF gene factors
    if args.bpmf_pkl:
        bpmf_path = args.bpmf_pkl
    else:
        bpmf_dir = config.OUTPUT_PATH / "bpmf"
        matches = sorted(bpmf_dir.glob(
            f"bpmf_public_train_K{args.d_gene}_su1_sv1_*/bpmf_result.pkl"))
        if not matches:
            raise FileNotFoundError(
                f"No BPMF pkl found for K={args.d_gene} at {bpmf_dir}")
        bpmf_path = str(matches[-1])
    V_np = _load_bpmf_factors(bpmf_path, vocab, args.d_gene)
    assert V_np.shape[0] == len(vocab), \
        f"V shape {V_np.shape} doesn't match vocab size {len(vocab)}"
    V = torch.tensor(V_np, dtype=torch.float32, device=device)
    bias = torch.zeros(len(vocab), dtype=torch.float32, device=device)

    # Build model
    screen_mlp = ScreenMLP(text_dim, args.d_gene, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(screen_mlp.parameters(), lr=args.lr)

    # Output dir
    out_dir = RANKERS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    wb = None
    try:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            mode=args.wandb_mode,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            config=vars(args),
        )
    except Exception:
        pass

    # Training loop
    best_val = -float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        screen_mlp.train()
        rng = np.random.default_rng(args.seed + epoch)
        perm = rng.permutation(len(train_examples))
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_examples), args.batch_size):
            batch_idx = perm[batch_start:batch_start + args.batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for idx in batch_idx:
                ex = train_examples[idx]
                n_genes = len(ex.gene_idx)
                n_ctx = max(1, int(n_genes * args.context_frac))
                p = rng.permutation(n_genes)
                ctx_idx = p[:n_ctx]
                query_idx = p[n_ctx:]
                if len(query_idx) == 0:
                    continue

                desc_emb = torch.tensor(ex.desc_emb, dtype=torch.float32,
                                       device=device).unsqueeze(0)
                u0 = screen_mlp(desc_emb).squeeze(0)

                gene_ids = torch.tensor(ex.gene_idx, device=device)
                hits = torch.tensor(ex.hit, dtype=torch.float32, device=device)

                # Inner loop (FOMAML: detach after inner steps)
                ctx_gene_ids = gene_ids[ctx_idx]
                ctx_hits = hits[ctx_idx]
                V_ctx = V[ctx_gene_ids]
                bias_ctx = bias[ctx_gene_ids]

                u = u0
                for _ in range(args.inner_steps):
                    logits = V_ctx @ u + bias_ctx
                    inner_loss = F.binary_cross_entropy_with_logits(logits, ctx_hits)
                    grad_u = torch.autograd.grad(inner_loss, u, create_graph=False)[0]
                    u = u - args.inner_lr * grad_u.detach()

                # Outer loss on query genes (through u0 via the MLP)
                # For FOMAML: we detached the inner grads, so outer loss
                # backprops only through u0 -> screen_mlp
                # Re-derive u from u0 with one step for the outer gradient
                u_outer = u0
                logits_ctx = V_ctx @ u_outer + bias_ctx
                inner_loss_outer = F.binary_cross_entropy_with_logits(logits_ctx, ctx_hits)
                grad_outer = torch.autograd.grad(inner_loss_outer, u_outer,
                                                 create_graph=True)[0]
                u_adapted = u_outer - args.inner_lr * grad_outer

                query_gene_ids = gene_ids[query_idx]
                query_hits = hits[query_idx]
                V_q = V[query_gene_ids]
                bias_q = bias[query_gene_ids]
                logits_q = V_q @ u_adapted + bias_q
                outer_loss = F.binary_cross_entropy_with_logits(logits_q, query_hits)
                batch_loss = batch_loss + outer_loss

            if len(batch_idx) > 0:
                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(screen_mlp.parameters(), 1.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_nvr = _eval_on_screens(screen_mlp, val_examples, V, bias, device,
                                    args.inner_steps, args.inner_lr,
                                    args.context_frac)

        row = {"epoch": epoch, "train_loss": avg_loss, "val_nvr": val_nvr}
        history.append(row)

        if val_nvr > best_val:
            best_val = val_nvr
            best_state = copy.deepcopy(screen_mlp.state_dict())
            row["is_best"] = True

        log.info("[e%d] loss=%.4f val_nvr=%.3f%s",
                 epoch, avg_loss, val_nvr,
                 " *" if row.get("is_best") else "")

        if wb is not None:
            wb.log({"train/loss": avg_loss, "eval/val_nvr": val_nvr,
                    "epoch": epoch})

    # Save
    save_state = {
        "screen_mlp": best_state or screen_mlp.state_dict(),
        "V": V_np,
        "bias": np.zeros(len(vocab), dtype=np.float32),
    }
    torch.save(save_state, out_dir / "model.pt")
    torch.save({
        "screen_mlp": screen_mlp.state_dict(),
        "V": V_np,
        "bias": np.zeros(len(vocab), dtype=np.float32),
    }, out_dir / "model_last.pt")

    vocab.save(out_dir / "vocab.json")
    cfg = {
        "d_gene": args.d_gene,
        "text_dim": text_dim,
        "hidden": args.hidden,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "text_backend": args.text_backend,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    summary = {"best_val_nvr": best_val, "epochs": args.epochs}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if wb is not None:
        wb.finish()

    log.info("Saved to %s (best val NVR=%.3f)", out_dir, best_val)


if __name__ == "__main__":
    main()
