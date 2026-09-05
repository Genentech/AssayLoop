"""Training loop for the amortized gene ranker.

Trains :class:`~assayloop.amortized.model.RankerNet` on BioGRID public-train
screens to regress per-gene unmasked relevance scores given a randomly sampled
observed-gene context (round 0) and, optionally, on-policy contexts produced by
rolling the current model through the AL loop (DAgger rounds).

Tracks train MSE, validation MSE (cold-start), and the real AL objective
(`n_hits_vs_random`) on a validation screen sample via wandb. Saves a
checkpoint dir consumable by
:class:`~assayloop.models.amortized_ranker.AmortizedRankerModel`.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .. import config
from .data import (
    GeneVocab, RankerDataset, ScreenExample, TextCollator, build_examples, collate,
)
from .model import RankerConfig, RankerNet
from . import text_embed as te

log = logging.getLogger("assayloop.amortized.train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(device: str | None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - tgt) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def _masked_bce(pred_logits: torch.Tensor, tgt_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft-target BCE: match the target P(hit). ``pred_logits`` are the raw
    ranker scores (ranking by them == ranking by sigmoid)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logits, tgt_prob, reduction="none"
    ) * mask
    denom = mask.sum().clamp_min(1.0)
    return bce.sum() / denom


def _masked_probit_nll(score: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BPMF likelihood: ``P(hit) = Φ(û · V_g)``. Masked-mean negative log-
    likelihood of the binary hits under the probit link. Uses ``log_ndtr`` for a
    numerically stable ``log Φ`` (and ``log Φ(-x) = log(1 - Φ(x))``)."""
    logp1 = torch.special.log_ndtr(score)
    logp0 = torch.special.log_ndtr(-score)
    nll = -(y * logp1 + (1.0 - y) * logp0) * mask
    return nll.sum() / mask.sum().clamp_min(1.0)


def _ctx_shift_aux(
    s_cold: torch.Tensor,
    s_ctx: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Auxiliary loss that rewards the *context* prediction for ranking the hits.

    All tensors are ``(B, L)`` (per-screen rows, padded targets); ``mask`` is the
    valid-target mask, ``tgt`` the (binary/soft) hit labels, and ``s_cold`` /
    ``s_ctx`` the target logits from the cold-start (empty context) and the
    revealed-context encodings. Returns a scalar to be *added* to the loss (each
    branch is negated -- minimizing it maximizes the desired quantity).

    The ``directional``/``rank`` objectives are deliberately **scale-invariant**
    and reference **only ``s_ctx``**. An earlier ``cov(Δscore, label)`` form blew
    up: covariance is linear in the logit scale, so it was maximized by inflating
    ``‖û‖·‖V‖`` to infinity (logit std -> 1e3, cold-start collapsed), and the
    ``ctx-cold`` difference perversely rewarded making the *cold* start
    anti-aligned with the labels (spearman -> -1). Targeting the correlation /
    soft-AUC of ``s_ctx`` alone fixes both.

    Modes:
      * ``directional`` (default): point-biserial **correlation** of ``s_ctx``
        with the row-centered labels -- bounded in [-1, 1], scale-free.
      * ``rank``: soft-AUC -- mean ``sigmoid(s_ctx[hit] - s_ctx[non-hit])`` over
        hit/non-hit pairs (bounded in [0, 1]).
      * ``raw``: the literal mean|Δscore| / std(cold) metric (uses ``s_cold``,
        std detached). A direction-blind control that rewards magnitude shifts
        which need not reorder (the quiet-sweep-89 failure mode).
    """
    m = mask
    n = m.sum(1).clamp_min(1.0)
    if mode == "raw":
        delta = s_ctx - s_cold
        num = (delta.abs() * m).sum(1) / n
        cmean = ((s_cold * m).sum(1) / n).unsqueeze(1)
        var = (((s_cold - cmean) ** 2) * m).sum(1) / n
        std = var.clamp_min(1e-12).sqrt().detach()
        return -(num / (std + 1e-6)).mean()
    if mode == "rank":
        hit = (tgt > 0.5) & (m > 0)
        non = (tgt <= 0.5) & (m > 0)
        per = []
        for b in range(s_ctx.shape[0]):
            sh, sn = s_ctx[b][hit[b]], s_ctx[b][non[b]]
            if sh.numel() == 0 or sn.numel() == 0:
                continue
            per.append(torch.sigmoid(sh.unsqueeze(1) - sn.unsqueeze(0)).mean())
        if not per:
            return s_ctx.sum() * 0.0
        return -torch.stack(per).mean()
    # directional (default): point-biserial correlation corr(s_ctx, label)
    tmean = ((tgt * m).sum(1) / n).unsqueeze(1)
    tc = (tgt - tmean) * m
    smean = ((s_ctx * m).sum(1) / n).unsqueeze(1)
    sc = (s_ctx - smean) * m
    cov = (sc * tc * m).sum(1) / n
    ss = ((sc ** 2 * m).sum(1) / n).clamp_min(1e-12).sqrt()
    st = ((tc ** 2 * m).sum(1) / n).clamp_min(1e-12).sqrt()
    return -(cov / (ss * st + 1e-8)).mean()


def _gauss_nll_per_dim(x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Per-dimension Gaussian negative log-prior (drops the 2π const):
    ``log σ + mean(x²)/(2σ²)``. The ``log σ`` normaliser makes ``σ`` fittable
    (its MLE is ``σ² = mean(x²)``); with fixed σ it is a constant and only the
    quadratic shrinkage term matters."""
    inv_2var = (-2.0 * log_sigma).exp() * 0.5
    return log_sigma + x.pow(2).mean() * inv_2var


def _kl_normal_per_dim(mu: torch.Tensor, logvar: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Per-dimension ``KL(N(µ, σ_q²) ‖ N(0, σ_u²))``, averaged over batch & dims.
    Replaces the MAP û-prior in the variational objective; depends on the prior
    ``log σ_u`` so a fittable prior std is trained through it."""
    sigma2 = (2.0 * log_sigma).exp()
    kl = 0.5 * ((logvar.exp() + mu.pow(2)) / sigma2 - 1.0 - logvar + 2.0 * log_sigma)
    return kl.mean()


def _subsample(screens: list, n: int | None, seed: int) -> list:
    pool = sorted(screens, key=lambda s: s.dataset_name)
    if n is None or n <= 0 or n >= len(pool):
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, n)


@torch.no_grad()
def _val_mse(
    net: RankerNet,
    val_examples: list[ScreenExample],
    device: torch.device,
    tokenizer: Any = None,
    link: str = "identity",
) -> float:
    """Cold-start (empty context) error over all in-screen genes, averaged over
    screens (each screen weighted equally).

    ``link`` maps the raw ranker logits to the prediction space before squaring:
    ``"identity"`` (plain MSE, for the relevance objective), ``"sigmoid"`` (the
    BCE hit objective) or ``"probit"`` (the BPMF objective). For the two
    probabilistic links the result is a **Brier score** (MSE of the predicted
    P(hit) vs the binary target) -- bounded in [0, 1] and comparable across runs,
    unlike the raw-logit MSE which blows up as the logits grow confident."""
    if not val_examples:
        return float("nan")
    net.eval()
    per_screen = []
    empty_i = torch.zeros(1, 0, dtype=torch.int64, device=device)
    empty_b = torch.zeros(1, 0, dtype=torch.bool, device=device)
    for ex in val_examples:
        if net.is_text:
            from .text_embed import render_context_text

            desc_text = ex.desc_text if net.cfg.use_description else ""
            text = render_context_text(desc_text, [], [])
            enc = tokenizer(
                [text], padding=True, truncation=True,
                max_length=net.cfg.text_max_tokens, return_tensors="pt",
            )
            repr_ = net.encode_text(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            )
        else:
            desc = torch.from_numpy(ex.desc_emb).float().unsqueeze(0).to(device)
            repr_ = net.encode(desc, empty_i, empty_i, empty_b)
        ids = torch.from_numpy(ex.gene_idx).long().unsqueeze(0).to(device)
        pred = net.score_ids(repr_, ids)[0]
        if link == "sigmoid":
            pred = torch.sigmoid(pred)
        elif link == "probit":
            pred = torch.special.ndtr(pred)
        tgt = torch.from_numpy(ex.score).float().to(device)
        per_screen.append(float(((pred - tgt) ** 2).mean().item()))
    return float(np.mean(per_screen))


def _build_inference_model(
    net: RankerNet,
    vocab: GeneVocab,
    desc_by_name: dict,
    device: torch.device,
    *,
    tokenizer: Any = None,
    desc_text_by_name: dict | None = None,
):
    from ..models.amortized_ranker import AmortizedRankerModel

    return AmortizedRankerModel(
        net=net, vocab=vocab, desc_emb_by_name=desc_by_name, device=str(device),
        tokenizer=tokenizer, desc_text_by_name=desc_text_by_name,
    )


def _al_eval(
    net: RankerNet,
    vocab: GeneVocab,
    screens: list,
    desc_by_name: dict,
    *,
    screen_set: str,
    batch_size: int,
    n_steps: int,
    device: torch.device,
    tokenizer: Any = None,
    desc_text_by_name: dict | None = None,
) -> dict[str, float]:
    """Run the real AL loop (greedy on the model's scores) over ``screens`` and
    return mean ``n_hits_vs_random`` / ``hits_auc``. Restores train mode after."""
    from ..experiment.runner import RunConfig, run_one_screen

    if not screens:
        return {}
    inf = _build_inference_model(
        net, vocab, desc_by_name, device,
        tokenizer=tokenizer, desc_text_by_name=desc_text_by_name,
    )
    cfg = RunConfig(
        screen_set=screen_set, model="amortized_ranker", acq="greedy",
        batch_size=batch_size, n_steps=n_steps, persist=False,
        metrics=["hits_auc"], parallel=1, max_shortfall_frac=1.0,
    )
    n_vs, aucs = [], []
    for s in screens:
        try:
            res = run_one_screen(s, cfg, model_obj=inf, verbose=False)
            fm = res.final_metrics or {}
            v = fm.get("n_hits_vs_random")
            a = fm.get("hits_auc")
            if isinstance(v, (int, float)):
                n_vs.append(float(v))
            if isinstance(a, (int, float)):
                aucs.append(float(a))
        except Exception as e:  # noqa: BLE001
            log.warning("AL eval failed on %s: %s", s.dataset_name, e)
    net.train()
    out = {}
    if n_vs:
        out["val_n_hits_vs_random"] = float(np.mean(n_vs))
    if aucs:
        out["val_hits_auc"] = float(np.mean(aucs))
    out["val_al_screens"] = float(len(n_vs))
    return out


# ---------------------------------------------------------------------------
# DAgger: on-policy context collection
# ---------------------------------------------------------------------------


class _OnPolicyDataset(Dataset):
    """Fixed (context, targets) samples harvested from model AL rollouts."""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        it = self.items[i]
        return {
            "desc_emb": torch.from_numpy(it["desc_emb"]),
            "ctx_idx": torch.from_numpy(it["ctx_idx"]),
            "ctx_hit": torch.from_numpy(it["ctx_hit"]),
            "tgt_idx": torch.from_numpy(it["tgt_idx"]),
            "tgt_score": torch.from_numpy(it["tgt_score"]),
            "desc_text": it.get("desc_text", ""),
            "ctx_sym": list(it.get("ctx_sym", [])),
            "ctx_hit_list": [int(x) for x in it["ctx_hit"]],
        }


def _collect_onpolicy(
    net: RankerNet,
    vocab: GeneVocab,
    screens: list,
    examples_by_name: dict[str, ScreenExample],
    desc_by_name: dict,
    *,
    batch_size: int,
    n_steps: int,
    max_targets: int | None,
    device: torch.device,
    seed: int,
    leave_out: bool = True,
    tokenizer: Any = None,
    desc_text_by_name: dict | None = None,
    target_mode: str = "relevance",
) -> list[dict[str, Any]]:
    """Roll the current model through the AL loop on ``screens`` and turn the
    observed prefixes (after each step) into fixed training samples."""
    from ..experiment.runner import RunConfig, run_one_screen

    inf = _build_inference_model(
        net, vocab, desc_by_name, device,
        tokenizer=tokenizer, desc_text_by_name=desc_text_by_name,
    )
    cfg = RunConfig(
        screen_set="train", model="amortized_ranker", acq="greedy",
        batch_size=batch_size, n_steps=n_steps, persist=False,
        metrics=["hits_auc"], parallel=1, max_shortfall_frac=1.0,
    )
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for s in screens:
        ex = examples_by_name.get(s.dataset_name)
        if ex is None:
            continue
        sym_to_pos = {g: i for i, g in enumerate(s.genes[: len(ex.gene_idx)])}
        try:
            res = run_one_screen(s, cfg, model_obj=inf, verbose=False)
        except Exception as e:  # noqa: BLE001
            log.warning("DAgger rollout failed on %s: %s", s.dataset_name, e)
            continue
        # Accumulate observed positions across steps; emit a sample per step.
        ctx_positions: list[int] = []
        for step in res.history:
            for o in step.new_observations or []:
                p = sym_to_pos.get(str(o.candidate))
                if p is not None:
                    ctx_positions.append(p)
            if not ctx_positions:
                continue
            cpos = np.array(ctx_positions, dtype=np.int64)
            # Targets: held-out genes under leave-out (exclude observed cpos),
            # else all in-screen genes; capped, keeping hits.
            g = len(ex.gene_idx)
            if leave_out:
                ctx_set = set(int(p) for p in cpos)
                cand = np.array([p for p in range(g) if p not in ctx_set], dtype=np.int64)
                if cand.size == 0:
                    cand = np.arange(g)
            else:
                cand = np.arange(g)
            if max_targets is None or cand.size <= max_targets:
                tpos = cand
            else:
                is_hit = ex.hit[cand] == 1
                hit_pos = cand[is_hit]
                non_hit = cand[~is_hit]
                budget = max(0, max_targets - len(hit_pos))
                samp = np.array(rng.sample(list(non_hit), min(budget, len(non_hit))),
                                dtype=np.int64) if budget > 0 else np.empty(0, np.int64)
                tpos = np.concatenate([hit_pos, samp])
            tgt_ids = ex.gene_idx[tpos]
            if target_mode == "hits":
                tgt_score = ex.hit[tpos].astype(np.float32)
            else:
                tgt_score = ex.score[tpos]
            items.append({
                "desc_emb": ex.desc_emb,
                "ctx_idx": ex.gene_idx[cpos],
                "ctx_hit": ex.hit[cpos],
                "tgt_idx": tgt_ids,
                "tgt_score": tgt_score,
                "desc_text": ex.desc_text,
                "ctx_sym": [ex.genes[p] for p in cpos],
            })
    net.train()
    return items


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(
    *,
    out_dir: str | Path | None = None,
    train_size: int | None = None,
    train_screen_set: str = "train",
    val_screen_set: str = "paper_validation",
    val_size: int | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    d_model: int = 256,
    d_gene: int = 256,
    d_hit: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    encoder_type: str = "transformer",
    bert_model: str = "answerdotai/ModernBERT-base",
    bert_lr: float = 2e-5,
    freeze_bert: bool = False,
    bert_pool: str = "cls",
    text_max_tokens: int = 2048,
    use_description: bool = True,
    max_context: int = 1024,
    max_targets: int | None = 4096,
    hit_context_frac: float = 0.5,
    leave_out: bool = True,
    init_gene_factors: str | None = None,
    freeze_gene_factors: bool = True,
    warm_start_glm_train: str | None = None,
    warm_start_prob: float = 0.0,
    warm_start_n: str | int = "random",
    objective: str = "relevance",
    bpmf_sigma_u: float = 1.0,
    bpmf_sigma_v: float = 1.0,
    bpmf_fit_sigma: bool = False,
    bpmf_variational: bool = False,
    disable_gene_bias: bool = False,
    gene_norm_reg: float = 0.0,
    gene_norm_target: float = 0.0,
    ctx_shift_coef: float = 0.0,
    ctx_shift_mode: str = "directional",
    cold_anchor_coef: float = 0.0,
    em_period: int = 0,
    num_workers: int = 0,
    al_batch_size: int = 100,
    al_n_steps: int = 10,
    dagger_rounds: int = 1,
    dagger_screens: int = 64,
    val_al_screens: int = 20,
    eval_every: int = 1,
    text_backend: str = "auto",
    text_model: str | None = None,
    # None => wandb uses the account default (or $WANDB_ENTITY).
    wandb_entity: str | None = None,
    wandb_project: str = "assayloop-amortized-ranker",
    wandb_mode: str = "online",
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    reuse_wandb: bool = False,
    run_name: str | None = None,
    seed: int = 0,
    device: str | None = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the amortized ranker; save artifacts; return a summary dict."""
    from ..tasks import load_screens

    if encoder_type not in ("transformer", "modernbert_embed", "modernbert_text"):
        raise ValueError(f"Unknown encoder_type {encoder_type!r}.")
    if objective not in ("relevance", "hits", "bpmf"):
        raise ValueError(f"Unknown objective {objective!r}; pick relevance|hits|bpmf.")
    is_bpmf = objective == "bpmf"
    if ctx_shift_mode not in ("directional", "rank", "raw"):
        raise ValueError(
            f"Unknown ctx_shift_mode {ctx_shift_mode!r}; pick directional|rank|raw."
        )
    is_text = encoder_type == "modernbert_text"
    is_bert = encoder_type in ("modernbert_embed", "modernbert_text")
    if (ctx_shift_coef > 0.0 or cold_anchor_coef > 0.0) and is_text:
        raise ValueError(
            "--ctx-shift-coef / --cold-anchor-coef need an explicit empty-context "
            "encode, which the modernbert_text encoder can't form from a batch "
            "(context is baked into the rendered text). Use a "
            "transformer/modernbert_embed encoder."
        )

    _seed_all(seed)
    dev = _pick_device(device)
    if is_bert and dev.type != "cuda" and not freeze_bert:
        log.warning(
            "Encoder %s with a full ModernBERT fine-tune on %s will be very slow; "
            "use --device cuda on a GPU host or --freeze-bert for a CPU smoke test.",
            encoder_type, dev.type,
        )
    run_name = run_name or f"ranker-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(out_dir) if out_dir else (config.OUTPUT_PATH / "rankers" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading screens (%s / %s / public)...", train_screen_set, val_screen_set)
    _t = time.time()
    train_screens = _subsample(load_screens(target_set=train_screen_set), train_size, seed)
    log.info("  loaded %s: %d screens (%.1fs)", train_screen_set, len(train_screens), time.time() - _t)
    _t = time.time()
    val_screens = _subsample(load_screens(target_set=val_screen_set), val_size, seed)
    log.info("  loaded %s: %d screens (%.1fs)", val_screen_set, len(val_screens), time.time() - _t)
    _t = time.time()
    test_screens = load_screens(target_set="paper_test")  # genes only, for vocab
    log.info("  loaded paper_test (vocab only): %d screens (%.1fs)", len(test_screens), time.time() - _t)

    # Vocabulary over the union so eval genes get (cold) slots.
    vocab = GeneVocab.build([train_screens, val_screens, test_screens])
    cov_val = vocab.coverage(val_screens)
    cov_test = vocab.coverage(test_screens)
    log.info("Vocab size %d | val coverage %.1f%% | test coverage %.1f%%",
             len(vocab), 100 * cov_val["frac"], 100 * cov_test["frac"])

    # Text tokenizer (text encoder) and/or description embeddings (others).
    # Description embeddings are only needed for the embedding-token encoders
    # *when the description is used*; the text encoder reads text directly, and
    # the no-description ablation drops the embedding entirely.
    tokenizer = None
    if is_text:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        log.info("Text encoder %s; tokenizer %s; no Azure embeddings needed.",
                 encoder_type, bert_model)

    need_desc_emb = (not is_text) and use_description
    if need_desc_emb:
        embedder = te.get_text_embedder(
            text_backend, **({"model": text_model} if text_model else {})
        )
        log.info("Text embedder: %s (dim=%d); embedding descriptions "
                 "(cached on disk)...", embedder.name, embedder.dim)
        text_dim = embedder.dim
        _t = time.time()
        train_desc = te.embed_screens(train_screens, embedder)
        val_desc = te.embed_screens(val_screens, embedder)
        log.info("  description embeddings ready (%.1fs); building examples...",
                 time.time() - _t)
        _t = time.time()
        train_examples = build_examples(train_screens, train_desc, vocab)
        val_examples = build_examples(val_screens, val_desc, vocab)
        log.info("  built %d train / %d val examples (%.1fs)",
                 len(train_examples), len(val_examples), time.time() - _t)
    else:
        embedder = None
        text_dim = 1
        train_desc, val_desc = {}, {}
        _t = time.time()
        train_examples = build_examples(train_screens, {}, vocab, require_desc_emb=False)
        val_examples = build_examples(val_screens, {}, vocab, require_desc_emb=False)
        log.info("Built %d train / %d val examples (%.1fs)",
                 len(train_examples), len(val_examples), time.time() - _t)
        if not is_text:
            log.info("use_description=False: skipping description embeddings; "
                     "the encoder uses a constant query token.")

    examples_by_name = {ex.name: ex for ex in train_examples}
    if not train_examples:
        raise RuntimeError("No training examples built (check embeddings/screens).")
    desc_text_by_name = {ex.name: ex.desc_text for ex in (*train_examples, *val_examples)}

    # Stage 1 of training: gene token initialisation (paper §4.1.1). The source
    # is the ablation axis of §6.7.1 -- bpmf / svd / mf / presage / k562 / random.
    marginal = None              # lazily computed; shared by the factor sources
    factors_init = None          # (V_vocab, bias_vocab) or None
    factors_freeze = False
    init_spec = init_gene_factors
    if init_spec:
        src, _, arg = init_spec.partition(":")
        src = src.strip().lower()
        log.info("Building gene factors from --init-gene-factors %s ...", init_spec)
        _t_fac = time.time()
        if src == "bpmf":
            from .gene_factors import BPMFFactors, marginal_hit_freq

            if not arg:
                raise ValueError("--init-gene-factors bpmf:<path> needs a pkl path.")
            import glob as _glob
            matches = sorted(_glob.glob(arg))
            if matches:
                arg = matches[0]
                log.info("Resolved bpmf pkl glob to: %s", arg)
            elif not Path(arg).exists():
                raise FileNotFoundError(
                    f"--init-gene-factors bpmf: no file matching {arg!r}")
            if marginal is None:
                marginal = marginal_hit_freq(train_examples, vocab)
            fac_src = BPMFFactors(arg, vocab, marginal=marginal)
            factors_init = fac_src.vocab_factors()  # (V_vocab, bias_vocab)
            d_gene = fac_src.K  # bilinear head lives in BPMF's K-dim
            log.info("init-gene-factors bpmf: %s gene factors to posterior-mean V "
                     "(d_gene=K=%d) (%.1fs).",
                     "frozen" if freeze_gene_factors else "trainable",
                     fac_src.K, time.time() - _t_fac)
        elif src == "presage":
            from ..data.gene_embeddings.presage import genept_vocab_factors
            from .gene_factors import marginal_hit_freq

            source = arg.strip() or "genept"
            if marginal is None:
                marginal = marginal_hit_freq(train_examples, vocab)
            factors_init = genept_vocab_factors(
                vocab, d_gene, marginal, source=source,
            )
            log.info("init-gene-factors presage:%s: %s PCA(d_gene=%d) gene factors "
                     "(%.1fs).", source,
                     "frozen" if freeze_gene_factors else "trainable", d_gene,
                     time.time() - _t_fac)
        elif src == "mf":
            # Pickle-free BPMF surrogate: masked ALS matrix *completion* of the
            # observed gene x screen hit matrix (proper missing-data handling,
            # unlike svd's zero-fill). arg = ridge reg (default 0.1).
            from .gene_factors import marginal_hit_freq, mf_vocab_factors

            mf_parts = arg.split(":") if arg else []
            reg = float(mf_parts[0]) if mf_parts and mf_parts[0] else 0.1
            normalize = not (len(mf_parts) > 1 and mf_parts[1] in ("raw", "nonorm"))
            if marginal is None:
                marginal = marginal_hit_freq(train_examples, vocab)
            factors_init = mf_vocab_factors(
                train_examples, vocab, d_gene, marginal, reg=reg, seed=seed,
                normalize=normalize,
            )
            log.info("init-gene-factors mf: %s ALS-completion gene factors "
                     "(d_gene=%d, reg=%.3g, %s) over %d screens (%.1fs).",
                     "frozen" if freeze_gene_factors else "trainable", d_gene,
                     reg, "shell" if normalize else "raw-norm",
                     len(train_examples), time.time() - _t_fac)
        elif src == "svd":
            # Pickle-free BPMF-like geometry: truncated SVD of the marginal-
            # centered gene x screen hit matrix -> heterogeneous-norm (high-
            # contrast) gene factors + logit(marginal) bias floor.
            from .gene_factors import marginal_hit_freq, svd_vocab_factors

            svd_parts = arg.split(":") if arg else []
            tgt_norm = float(svd_parts[0]) if svd_parts and svd_parts[0] else 1.0
            normalize = not (len(svd_parts) > 1 and svd_parts[1] in ("raw", "nonorm"))
            if marginal is None:
                marginal = marginal_hit_freq(train_examples, vocab)
            factors_init = svd_vocab_factors(
                train_examples, vocab, d_gene, marginal,
                seed=seed, target_norm=tgt_norm, normalize=normalize,
            )
            log.info("init-gene-factors svd: %s truncated-SVD gene factors "
                     "(d_gene=%d, norm=%.2g, %s) over %d screens (%.1fs).",
                     "frozen" if freeze_gene_factors else "trainable", d_gene,
                     tgt_norm, "shell" if normalize else "raw-norm",
                     len(train_examples), time.time() - _t_fac)
        elif src == "random":
            # Control geometry: Gaussian factors (no learned structure). Used
            # both frozen (does *any* fixed low-rank bilinear bottleneck help?)
            # and trainable (a matched-scale baseline for the learned sources).
            std = float(arg) if arg else 1.0
            rng = np.random.default_rng(seed)
            V_vocab = rng.normal(0.0, std, size=(len(vocab), d_gene)).astype(np.float32)
            V_vocab[0] = 0.0  # keep padding/<unk> id at 0
            bias_vocab = np.zeros(len(vocab), dtype=np.float32)
            factors_init = (V_vocab, bias_vocab)
            log.info("init-gene-factors random: %s N(0,%.3g) gene factors "
                     "(d_gene=%d, seed=%d) (%.1fs).",
                     "frozen" if freeze_gene_factors else "trainable", std, d_gene,
                     seed, time.time() - _t_fac)
        elif src == "k562":
            from ..data.gene_embeddings.presage import k562_perturbseq_vocab_factors
            from .gene_factors import marginal_hit_freq
            if marginal is None:
                marginal = marginal_hit_freq(train_examples, vocab)
            dataset = arg.strip() or "replogle_k562_gw"
            factors_init = k562_perturbseq_vocab_factors(
                vocab, d_gene, marginal, dataset=dataset,
            )
            log.info("init-gene-factors k562:%s: %s PCA(d_gene=%d) gene factors "
                     "(%.1fs).", dataset,
                     "frozen" if freeze_gene_factors else "trainable", d_gene,
                     time.time() - _t_fac)
        else:
            raise ValueError(
                f"Unknown --init-gene-factors source {src!r}; use bpmf:<pkl>, "
                "presage:<source>, mf[:<reg>], svd, or random[:<std>]."
            )
        factors_freeze = bool(freeze_gene_factors)

    arch = RankerConfig(
        vocab_size=len(vocab), text_dim=text_dim, d_model=d_model, d_gene=d_gene,
        d_hit=d_hit, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout,
        encoder_type=encoder_type, bert_model=bert_model, freeze_bert=freeze_bert,
        bert_pool=bert_pool, text_max_tokens=text_max_tokens,
        use_description=use_description, freeze_factors=factors_freeze,
        disable_gene_bias=bool(disable_gene_bias),
        variational=bool(is_bpmf and bpmf_variational),
    )
    net = RankerNet(arch).to(dev)
    if factors_init is not None:
        V_vocab, bias_vocab = factors_init
        net.set_gene_factors(
            torch.from_numpy(V_vocab).to(dev),
            torch.from_numpy(bias_vocab).to(dev),
            freeze=factors_freeze,
        )

    # BPMF objective: the Gaussian prior on the gene factors V is applied
    # explicitly in the loss, so keep ``gene_emb`` out of AdamW weight decay to
    # avoid double-shrinking it. The prior std(s) σ_u, σ_v are either fixed
    # constants or fitted parameters (their MLE balances shrinkage vs the log-σ
    # normaliser). σ are stored as unconstrained log-σ for stable optimisation.
    log_sigma_u = log_sigma_v = None
    sigma_groups: list[dict[str, Any]] = []
    if is_bpmf:
        _lsu = torch.tensor(math.log(max(float(bpmf_sigma_u), 1e-4)), device=dev)
        _lsv = torch.tensor(math.log(max(float(bpmf_sigma_v), 1e-4)), device=dev)
        if bpmf_fit_sigma:
            log_sigma_u = torch.nn.Parameter(_lsu)
            log_sigma_v = torch.nn.Parameter(_lsv)
            sigma_groups = [{"params": [log_sigma_u, log_sigma_v], "lr": lr, "weight_decay": 0.0}]
        else:
            log_sigma_u, log_sigma_v = _lsu, _lsv

    def _is_gene_emb(name: str) -> bool:
        return name == "gene_emb.weight"

    # Keep gene_emb out of AdamW weight decay when the BPMF objective applies its
    # own Gaussian prior, OR when V is a (trainable) externally-initialised
    # geometry we don't want decayed back toward zero.
    gene_emb_no_wd = is_bpmf or (factors_init is not None and not factors_freeze)

    # Discriminative LRs: pretrained backbone slow, new heads fast.
    if is_bert and not freeze_bert:
        bb = [p for n, p in net.named_parameters() if n.startswith("bert.") and p.requires_grad]
        head = [p for n, p in net.named_parameters()
                if not n.startswith("bert.") and p.requires_grad
                and not (gene_emb_no_wd and _is_gene_emb(n))]
        groups = [{"params": bb, "lr": bert_lr}, {"params": head, "lr": lr}]
        if gene_emb_no_wd:
            ge = [p for n, p in net.named_parameters() if _is_gene_emb(n) and p.requires_grad]
            if ge:
                groups.append({"params": ge, "lr": lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups + sigma_groups, weight_decay=weight_decay)
        log.info("AdamW param groups: backbone lr=%g (%d tensors), head lr=%g (%d tensors)",
                 bert_lr, len(bb), lr, len(head))
    elif gene_emb_no_wd:
        ge = [p for n, p in net.named_parameters() if _is_gene_emb(n) and p.requires_grad]
        other = [p for n, p in net.named_parameters()
                 if not _is_gene_emb(n) and p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": other, "lr": lr, "weight_decay": weight_decay},
             {"params": ge, "lr": lr, "weight_decay": 0.0}] + sigma_groups,
            lr=lr,
        )
    else:
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # EM / coordinate-ascent alternation: split the trainable params into the
    # gene side (V, bias) and the transformer side (encoder/head). Both must be
    # in the optimizer at construction (they are, since they're trainable here);
    # each epoch we flip requires_grad so AdamW only steps one block at a time
    # (amortized ALS: solve the inference net given V, then refine V given it).
    em_gene_params: list[torch.nn.Parameter] = []
    em_other_params: list[torch.nn.Parameter] = []
    if em_period > 0:
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            (em_gene_params if (_is_gene_emb(n) or n == "gene_bias")
             else em_other_params).append(p)
        if not em_gene_params:
            raise ValueError(
                "--em-period requires trainable gene factors; pass "
                "--no-freeze-gene-factors (and an --init-gene-factors / from-scratch "
                "table), otherwise there is no V block to alternate on.")
        log.info("EM alternation every %d epoch(s): %d gene tensors <-> %d "
                 "transformer tensors (epoch 0 = transformer phase).",
                 em_period, len(em_gene_params), len(em_other_params))

    # wandb (optional; creds via ~/.netrc).
    wb = None
    try:
        import wandb  # noqa: PLC0415

        _wb_cfg = {
                "arch": arch.to_dict(), "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "weight_decay": weight_decay, "dagger_rounds": dagger_rounds,
                "n_train": len(train_examples), "n_val": len(val_examples),
                "vocab_size": len(vocab),
                "text_embedder": embedder.name if embedder else f"text:{bert_model}",
                "encoder_type": encoder_type, "bert_model": bert_model,
                "bert_lr": bert_lr, "freeze_bert": freeze_bert,
                "val_coverage": cov_val["frac"], "test_coverage": cov_test["frac"],
                "max_context": max_context, "max_targets": max_targets,
                "hit_context_frac": hit_context_frac, "leave_out": leave_out,
                "use_description": use_description,
                "init_gene_factors": init_gene_factors,
                "freeze_gene_factors": freeze_gene_factors,
                "objective": objective,
                "bpmf_sigma_u": bpmf_sigma_u,
                "bpmf_sigma_v": bpmf_sigma_v,
                "bpmf_fit_sigma": bpmf_fit_sigma,
                "bpmf_variational": bpmf_variational,
                "disable_gene_bias": disable_gene_bias,
                "em_period": em_period,
                "seed": seed,
                "train_screen_set": train_screen_set,
                "val_screen_set": val_screen_set,
        }
        if reuse_wandb and wandb.run is not None:
            # A sweep agent (or caller) already owns the run; attach to it.
            wb = wandb.run
            wb.config.update(_wb_cfg, allow_val_change=True)
        else:
            wb = wandb.init(
                entity=wandb_entity, project=wandb_project, name=run_name, mode=wandb_mode,
                group=wandb_group, tags=wandb_tags or None, config=_wb_cfg,
            )
    except Exception as e:  # noqa: BLE001
        log.warning("wandb unavailable (%s); training without it.", e)

    warm_start = None
    if warm_start_glm_train and warm_start_prob > 0.0:
        from .warmstart import WarmStart
        warm_start = WarmStart.from_train_jsonl(warm_start_glm_train)
        log.info("GLM warm-start contexts enabled (prob=%.2f, n=%s) from %s.",
                 warm_start_prob, warm_start_n, warm_start_glm_train)

    # The BPMF objective trains on binary hit labels (probit likelihood), so it
    # uses the same per-screen hit targets as the self-supervised ``hits`` mode.
    ds_target_mode = "hits" if is_bpmf else objective
    base_ds = RankerDataset(
        train_examples, max_context=max_context, max_targets=max_targets,
        hit_context_frac=hit_context_frac, leave_out=leave_out, seed=seed,
        target_mode=ds_target_mode,
        warm_start=warm_start, warm_start_prob=warm_start_prob,
        warm_start_n=warm_start_n,
    )
    # Soft-target BCE for the self-supervised hit objective (binary hit
    # targets); MSE only for static relevance.
    use_bce = objective == "hits"
    # Objective-aware metric labels (the optimized train loss is BCE for hits,
    # the probit NLL + priors for bpmf, and MSE only for relevance; the
    # val error is a Brier score under the matching link for the probabilistic
    # objectives -- see _val_mse).
    train_metric_name = "loss" if is_bpmf else ("bce" if use_bce else "mse")
    val_metric_name = "brier" if (is_bpmf or use_bce) else "mse"
    val_link = "probit" if is_bpmf else ("sigmoid" if use_bce else "identity")
    # Cold-start anchor target: the marginal hit rate (best context-free guess).
    # Supervising the empty-context prediction toward it pins the cold-start to a
    # sane prior, making the cold->ctx gap (context reordering) well-posed for the
    # ctx-shift aux instead of being gamed by degrading the cold-start.
    marg_t = None
    if cold_anchor_coef > 0.0:
        from .gene_factors import marginal_hit_freq

        if marginal is None:
            marginal = marginal_hit_freq(train_examples, vocab)
        marg_t = torch.tensor(
            np.clip(np.asarray(marginal, dtype=np.float32), 1e-6, 1.0 - 1e-6),
            device=dev,
        )
    collate_fn = (
        TextCollator(tokenizer, text_max_tokens, use_description=use_description)
        if is_text else collate
    )

    best_val = -float("inf")
    best_state = copy.deepcopy(net.state_dict())
    history: list[dict[str, Any]] = []
    global_step = 0

    val_al_set = _subsample(val_screens, val_al_screens, seed)

    for rnd in range(dagger_rounds + 1):
        if rnd == 0:
            ds: Dataset = base_ds
        else:
            log.info("[DAgger round %d] collecting on-policy contexts...", rnd)
            roll_screens = _subsample(train_screens, dagger_screens, seed + rnd)
            items = _collect_onpolicy(
                net, vocab, roll_screens, examples_by_name, train_desc,
                batch_size=al_batch_size, n_steps=al_n_steps,
                max_targets=max_targets, device=dev, seed=seed + rnd,
                leave_out=leave_out,
                tokenizer=tokenizer, desc_text_by_name=desc_text_by_name,
                target_mode=ds_target_mode,
            )
            log.info("[DAgger round %d] %d on-policy samples", rnd, len(items))
            ds = ConcatDataset([base_ds, _OnPolicyDataset(items)]) if items else base_ds

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=num_workers,
        )

        for epoch in range(epochs):
            net.train()
            em_emb_phase = False
            if em_period > 0:
                # epoch 0 -> transformer phase (warm the inference net on the
                # current V), then alternate every em_period epochs.
                em_emb_phase = ((epoch // em_period) % 2 == 1)
                for p in em_gene_params:
                    p.requires_grad_(em_emb_phase)
                for p in em_other_params:
                    p.requires_grad_(not em_emb_phase)
            ep_loss, ep_n, ep_reg, ep_aux, ep_canch = 0.0, 0, 0.0, 0.0, 0.0
            for batch in loader:
                tgt_idx = batch["tgt_idx"].to(dev)
                tgt_score = batch["tgt_score"].to(dev)
                tgt_mask = batch["tgt_mask"].to(dev)

                if is_text:
                    text_args = (batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
                else:
                    desc = batch["desc_emb"].to(dev)
                    ctx_idx = batch["ctx_idx"].to(dev)
                    ctx_hit = batch["ctx_hit"].to(dev)
                    ctx_pad = batch["ctx_pad"].to(dev)

                if is_bpmf:
                    # Amortized BPMF: û = encoder(context); P(hit) = Φ(û · V_g).
                    if arch.variational:
                        mu, logvar = (
                            net.encode_text_var(*text_args) if is_text
                            else net.encode_var(desc, ctx_idx, ctx_hit, ctx_pad)
                        )
                        u = mu + (0.5 * logvar).exp() * torch.randn_like(logvar)
                    else:
                        u = (
                            net.encode_text(*text_args) if is_text
                            else net.encode(desc, ctx_idx, ctx_hit, ctx_pad)
                        )
                    pred = net.score_ids(u, tgt_idx)
                    nll = _masked_probit_nll(pred, tgt_score, tgt_mask.float())
                    u_prior = (
                        _kl_normal_per_dim(mu, logvar, log_sigma_u) if arch.variational
                        else _gauss_nll_per_dim(u, log_sigma_u)
                    )
                    v_prior = _gauss_nll_per_dim(net.gene_emb.weight, log_sigma_v)
                    loss = nll + u_prior + v_prior
                else:
                    if is_text:
                        pred = net.score_ids(net.encode_text(*text_args), tgt_idx)
                    else:
                        pred = net(desc, ctx_idx, ctx_hit, ctx_pad, tgt_idx)
                    if use_bce:
                        loss = _masked_bce(pred, tgt_score, tgt_mask.float())
                    else:
                        loss = _masked_mse(pred, tgt_score, tgt_mask.float())

                # Context-shift auxiliary: encode the same screens with an *empty*
                # context and reward the revealed-context encoding for reordering
                # the targets toward the hits (see _ctx_shift_aux). Manufactures
                # the cold->ctx sharpening that the bpmf probit objective lacks.
                need_ctx = ctx_shift_coef > 0.0 and not is_text
                need_cold = cold_anchor_coef > 0.0 and not is_text
                if need_ctx or need_cold:
                    s_ctx = net.score_ids(
                        net.encode(desc, ctx_idx, ctx_hit, ctx_pad), tgt_idx
                    )
                    s_cold = s_ctx  # default; replaced below when an empty encode is needed
                    if need_cold or (need_ctx and ctx_shift_mode == "raw"):
                        B = tgt_idx.shape[0]
                        e_idx = ctx_idx.new_zeros((B, 0))
                        e_hit = ctx_hit.new_zeros((B, 0))
                        e_pad = ctx_pad.new_zeros((B, 0))
                        s_cold = net.score_ids(
                            net.encode(desc, e_idx, e_hit, e_pad), tgt_idx
                        )
                    if need_ctx:
                        aux = _ctx_shift_aux(
                            s_cold, s_ctx, tgt_score, tgt_mask.float(), ctx_shift_mode
                        )
                        loss = loss + ctx_shift_coef * aux
                        ep_aux += float(aux.item())
                    if need_cold:
                        canch = _masked_bce(
                            s_cold, marg_t[tgt_idx], tgt_mask.float()
                        )
                        loss = loss + cold_anchor_coef * canch
                        ep_canch += float(canch.item())

                # Sphere (shell) regularizer: pull each trained gene factor's
                # row-norm toward a common radius so ranking is decided by
                # *direction* (context-controllable) rather than by a few high-
                # norm genes. Mirrors the mf/svd shell init for the from-scratch
                # bpmf objective (no-op when gene_emb is frozen).
                if gene_norm_reg > 0.0 and net.gene_emb.weight.requires_grad:
                    bids = tgt_idx[tgt_mask.bool()]
                    bids = bids[bids > 0]
                    if bids.numel() > 0:
                        norms = net.gene_emb.weight.index_select(0, bids).norm(dim=1)
                        r = (
                            norms.new_full((), float(gene_norm_target))
                            if gene_norm_target > 0.0 else norms.mean().detach()
                        )
                        shell_reg = gene_norm_reg * ((norms - r) ** 2).mean()
                        loss = loss + shell_reg
                        ep_reg += float(shell_reg.item())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()

                ep_loss += float(loss.item())
                ep_n += 1
                global_step += 1
                if wb is not None:
                    wb.log({"train/loss_step": float(loss.item()), "step": global_step})

            train_loss = ep_loss / max(ep_n, 1)
            row: dict[str, Any] = {"round": rnd, "epoch": epoch, "train_loss": train_loss}
            if em_period > 0:
                row["em_phase"] = "embedding" if em_emb_phase else "transformer"
            if gene_norm_reg > 0.0:
                row["train_shell_reg"] = ep_reg / max(ep_n, 1)
            if ctx_shift_coef > 0.0:
                row["train_ctx_shift"] = ep_aux / max(ep_n, 1)
            if cold_anchor_coef > 0.0:
                row["train_cold_anchor"] = ep_canch / max(ep_n, 1)

            if (epoch % eval_every) == 0 or epoch == epochs - 1:
                vmse = _val_mse(net, val_examples, dev, tokenizer=tokenizer, link=val_link)
                al = _al_eval(
                    net, vocab, val_al_set, val_desc,
                    screen_set=val_screen_set,
                    batch_size=al_batch_size, n_steps=al_n_steps, device=dev,
                    tokenizer=tokenizer, desc_text_by_name=desc_text_by_name,
                )
                row["val_err"] = vmse
                row.update(al)
                val_metric = al.get("val_n_hits_vs_random", -float("inf"))
                if val_metric > best_val:
                    best_val = val_metric
                    best_state = copy.deepcopy(net.state_dict())
                    row["is_best"] = True
                if wb is not None:
                    wb_log = {
                        "train/loss_epoch": train_loss,
                        f"val/{val_metric_name}": vmse,
                        **{f"val/{k}": v for k, v in al.items()},
                        "epoch": epoch, "round": rnd, "step": global_step,
                    }
                    if gene_norm_reg > 0.0:
                        wb_log["train/shell_reg_epoch"] = row["train_shell_reg"]
                    if ctx_shift_coef > 0.0:
                        wb_log["train/ctx_shift_epoch"] = row["train_ctx_shift"]
                    if cold_anchor_coef > 0.0:
                        wb_log["train/cold_anchor_epoch"] = row["train_cold_anchor"]
                    if best_val != -float("inf"):
                        wb_log["val/best_n_hits_vs_random"] = best_val
                    if is_bpmf:
                        wb_log["bpmf/sigma_u"] = float(log_sigma_u.exp().item())
                        wb_log["bpmf/sigma_v"] = float(log_sigma_v.exp().item())
                    wb.log(wb_log)

            history.append(row)
            if verbose:
                reg_str = (
                    f" shell_reg={row['train_shell_reg']:.5f}"
                    if "train_shell_reg" in row else ""
                )
                if "train_ctx_shift" in row:
                    reg_str += f" ctx_shift={row['train_ctx_shift']:.5f}"
                if "train_cold_anchor" in row:
                    reg_str += f" cold_anchor={row['train_cold_anchor']:.5f}"
                log.info(
                    "[r%d e%d] train_%s=%.5f%s%s",
                    rnd, epoch, train_metric_name, train_loss, reg_str,
                    (f" val_{val_metric_name}={row['val_err']:.5f} "
                     f"val_n_vs_rand={row.get('val_n_hits_vs_random', float('nan')):.3f}"
                     if "val_err" in row else ""),
                )

    # Persist checkpoints + artifacts.
    torch.save(net.state_dict(), out_dir / "model_last.pt")
    net.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    vocab.save(out_dir / "vocab.json")
    np.save(out_dir / "gene_embeddings.npy", net.gene_emb.weight.detach().cpu().numpy())
    if embedder is None:
        text_backend_resolved = "none"
    else:
        # ``name`` is "<backend>:<model>". Record the backend that produced the
        # vectors so a reload knows which endpoint would pay for a cache miss.
        # (openai and azure are the same model and share cache entries, so this
        # is informational; only "local" implies a different vector space.)
        text_backend_resolved = embedder.name.split(":", 1)[0]
    (out_dir / "config.json").write_text(json.dumps({
        "arch": arch.to_dict(),
        "text_backend": text_backend_resolved,
        "text_model": getattr(embedder, "model", None) if embedder else None,
        "encoder_type": encoder_type,
        "bert_model": bert_model,
        "bert_pool": bert_pool,
        "text_max_tokens": text_max_tokens,
        "train_args": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "bert_lr": bert_lr, "freeze_bert": freeze_bert,
            "dagger_rounds": dagger_rounds, "leave_out": leave_out, "seed": seed,
            "use_description": use_description,
            "init_gene_factors": init_gene_factors,
            "freeze_gene_factors": freeze_gene_factors,
            "objective": objective,
            "bpmf": {
                "sigma_u_init": bpmf_sigma_u,
                "sigma_v_init": bpmf_sigma_v,
                "fit_sigma": bpmf_fit_sigma,
                "variational": bpmf_variational,
                "sigma_u": float(log_sigma_u.exp().item()) if is_bpmf else None,
                "sigma_v": float(log_sigma_v.exp().item()) if is_bpmf else None,
            } if is_bpmf else None,
            "disable_gene_bias": disable_gene_bias,
            "gene_norm_reg": gene_norm_reg,
            "gene_norm_target": gene_norm_target,
            "ctx_shift_coef": ctx_shift_coef,
            "ctx_shift_mode": ctx_shift_mode if ctx_shift_coef > 0.0 else None,
            "cold_anchor_coef": cold_anchor_coef,
            "train_screen_set": train_screen_set,
            "val_screen_set": val_screen_set,
            "wandb_group": wandb_group,
            "wandb_tags": wandb_tags or [],
            "n_train": len(train_examples), "n_val": len(val_examples),
            "warm_start": {
                "glm_train": warm_start_glm_train,
                "prob": warm_start_prob,
                "n": warm_start_n,
            } if warm_start is not None else None,
        },
    }, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "run_name": run_name,
        "best_val_n_hits_vs_random": best_val if best_val != -float("inf") else None,
        "vocab_size": len(vocab),
        "val_coverage": cov_val["frac"],
        "test_coverage": cov_test["frac"],
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "train_screen_set": train_screen_set,
        "val_screen_set": val_screen_set,
        "text_embedder": embedder.name if embedder else f"text:{bert_model}",
        "encoder_type": encoder_type,
        "use_description": use_description,
        "init_gene_factors": init_gene_factors,
        "freeze_gene_factors": freeze_gene_factors,
        "disable_gene_bias": disable_gene_bias,
        "objective": (
            "amortized_bpmf_probit" + ("_variational" if bpmf_variational else "_map")
            if is_bpmf
            else "self_supervised_hit_bce" if objective == "hits"
            else "relevance_mse"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wb is not None:
        try:
            wb_summary = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            if summary.get("best_val_n_hits_vs_random") is not None:
                wb_summary["val/best_n_hits_vs_random"] = summary["best_val_n_hits_vs_random"]
            wb.summary.update(wb_summary)
            wb.finish()
        except Exception:  # noqa: BLE001
            pass
    log.info("Saved ranker to %s (best val_n_hits_vs_random=%s)", out_dir, summary["best_val_n_hits_vs_random"])
    return summary


__all__ = ["run_training"]
