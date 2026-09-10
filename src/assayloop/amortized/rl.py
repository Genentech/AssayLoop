"""GRPO-style RL fine-tuning for the amortized gene ranker.

Trains a warm-started :class:`~assayloop.amortized.model.RankerNet` to directly
optimize the deployed active-learning metrics instead of a pointwise MSE
surrogate. Each training step rolls the *policy* through the AL loop on a screen
``G`` times (a GRPO group), scoring per-step batches with
``batch_random_adjusted_hit_rate`` and a terminal ``n_hits_vs_random`` bonus,
then updates with group-normalized advantages plus a KL anchor to the frozen
warm-start checkpoint and an entropy bonus.

Design notes
------------
- **Differentiable rollout.** Unlike the inference stack
  (``AmortizedRankerModel.predict`` is ``@torch.no_grad``), this module keeps the
  policy graph live across the trajectory so policy-gradient flows through every
  per-step scoring distribution.
- **Action = a batch of ``k`` genes sampled without replacement** via
  Gumbel-top-k (equivalent to Plackett-Luce sampling). The per-step log-prob is
  the exact ordered-PL log-prob along the realized order.
- **Warm start + KL anchor** are essential: REINFORCE from scratch over a
  ~24k-action space with sparse reward will not learn, and the supervised init
  already provides a decent static ranking we don't want to destroy.
- Only the embedding-token encoders (``transformer`` / ``modernbert_embed``) are
  supported here; ``modernbert_text`` would need per-step re-tokenization and is
  out of scope for this prototype.

The saved checkpoint dir is byte-compatible with
:class:`~assayloop.models.amortized_ranker.AmortizedRankerModel`, so
``eval-ranker`` / the dashboard consume it unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .. import config
from ..metrics.hits_auc import adjusted_ef_value, n_hits_vs_random_value
from .data import GeneVocab, ScreenExample, build_examples
from .model import RankerConfig, RankerNet
from .warmstart import WarmStart, load_run_traces

# Optional second location for runs (LLM handoff traces live here or under
# output/runs). Set ASSAYLOOP_SHARED_PATH to share runs across a team.
_SHARED_RUNS_DIR = config.SHARED_PATH / "runs"
from . import text_embed as te

log = logging.getLogger("assayloop.amortized.rl")

_NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_advantage(
    ret: torch.Tensor, std_floor: float, clip: float
) -> torch.Tensor:
    """Group-normalized advantage with a configurable std floor and optional
    magnitude clip.

    The std floor matters: when a group's returns are near-identical (e.g. a
    near-deterministic / low-entropy policy on one screen) ``ret.std`` collapses
    and a tiny default floor (1e-6) makes the normalized advantage explode,
    which — with no KL trust region — can drive a single destructive update.
    A larger floor caps that blow-up; ``clip`` additionally bounds |advantage|.
    """
    adv = (ret - ret.mean(0, keepdim=True)) / ret.std(0, keepdim=True).clamp_min(std_floor)
    if clip and clip > 0:
        adv = adv.clamp(-clip, clip)
    return adv


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


def _dist_info() -> tuple[bool, int, int, int]:
    """Read the torchrun environment -> (is_distributed, rank, world_size,
    local_rank). Falls back to single-process when not launched under torchrun."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1 and "RANK" in os.environ:
        return True, int(os.environ["RANK"]), ws, int(os.environ.get("LOCAL_RANK", "0"))
    return False, 0, 1, 0


def _all_reduce_grads(params: list[torch.nn.Parameter], world_size: int) -> None:
    """Average gradients across ranks in one coalesced collective. Missing grads
    (a rank whose screen window was entirely empty) are zero-filled so the
    collective stays symmetric across ranks. Dividing by ``world_size`` means an
    empty rank slightly dilutes the step, which is rare and harmless."""
    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def _subsample(screens: list, n: int | None, seed: int) -> list:
    pool = sorted(screens, key=lambda s: s.dataset_name)
    if n is None or n <= 0 or n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


class _TensorExample:
    """Per-screen tensors cached on-device for fast repeated rollouts.

    ``genes`` / ``desc_text`` are only needed by the ``modernbert_text`` encoder,
    which re-renders the observed context to text each step.
    """

    __slots__ = ("name", "desc", "gene_ids", "hits", "n", "n_library", "total_hits",
                 "genes", "desc_text")

    def __init__(self, ex: ScreenExample, device: torch.device):
        self.name = ex.name
        self.desc = torch.from_numpy(ex.desc_emb).float().to(device)
        self.gene_ids = torch.from_numpy(ex.gene_idx).long().to(device)
        self.hits = torch.from_numpy(ex.hit).float().to(device)
        self.n = int(self.gene_ids.shape[0])
        # # in-library genes (positions [0, n_library) are in-screen; the rest are
        # appended universe-extras). Falls back to n for non-full-genome examples.
        self.n_library = int(getattr(ex, "n_library", self.n) or self.n)
        self.total_hits = float(self.hits.sum().item())
        self.genes = list(ex.genes)
        self.desc_text = ex.desc_text


def _score_text(
    net: RankerNet,
    ex: "_TensorExample",
    obs_pos_per_rollout: list[list[int]],
    tokenizer: Any,
    max_tokens: int,
) -> torch.Tensor:
    """Render+tokenize each rollout's observed context and run the text encoder
    -> scores (G, n). The observed genes become text, so the context is fed via
    re-tokenization each step (no growing id tensor)."""
    from .text_embed import render_context_text

    device = ex.gene_ids.device
    G = len(obs_pos_per_rollout)
    desc_text = ex.desc_text if net.cfg.use_description else ""
    texts = [
        render_context_text(
            desc_text,
            [ex.genes[p] for p in op],
            [int(ex.hits[p].item()) for p in op],
        )
        for op in obs_pos_per_rollout
    ]
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt",
    )
    repr_ = net.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    gene_ids_b = ex.gene_ids.unsqueeze(0).expand(G, ex.n)
    return net.score_ids(repr_, gene_ids_b)


def _encode_group(
    net: RankerNet,
    desc_b: torch.Tensor,        # (B, text_dim)
    gene_ids_b: torch.Tensor,    # (B, n)
    hits_b: torch.Tensor,        # (B, n) float
    ctx_pos: torch.Tensor,       # (B, Lc) long observed positions (may be 0-wide)
    rl_max_context: int,
) -> torch.Tensor:
    """Run ``net.encode`` for a group given observed positions -> scores (B, n)."""
    B = desc_b.shape[0]
    device = desc_b.device
    if ctx_pos.shape[1] > 0:
        cp = ctx_pos
        if cp.shape[1] > rl_max_context:
            keep = torch.randperm(cp.shape[1], device=device)[:rl_max_context]
            cp = cp[:, keep]
        ctx_idx = torch.gather(gene_ids_b, 1, cp)
        ctx_hit = torch.gather(hits_b, 1, cp).long()
        ctx_pad = torch.zeros(B, cp.shape[1], dtype=torch.bool, device=device)
    else:
        ctx_idx = torch.zeros(B, 0, dtype=torch.int64, device=device)
        ctx_hit = torch.zeros(B, 0, dtype=torch.int64, device=device)
        ctx_pad = torch.zeros(B, 0, dtype=torch.bool, device=device)
    desc_repr = net.encode(desc_b, ctx_idx, ctx_hit, ctx_pad)   # (B, d_gene)
    return net.score_ids(desc_repr, gene_ids_b)                 # (B, n)


@torch.no_grad()
def _log_sample_io(
    net: RankerNet, ex: _TensorExample, *, n_steps: int, batch_size: int,
    rl_max_context: int, tokenizer: Any, max_tokens: int, is_text: bool, log: logging.Logger,
) -> None:
    """One-time sanity dump of exactly what the encoder is fed and predicts.

    Runs a greedy rollout on one screen with the *same* context handling as
    training so confounds are visible at a glance — e.g. the ``rl_max_context``
    cap silently subsampling observations, a zeroed description embedding, or
    text truncation at ``max_tokens``.
    """
    from .text_embed import render_context_text  # noqa: PLC0415

    device = ex.gene_ids.device
    was_training = net.training
    net.eval()
    use_desc = net.cfg.use_description
    log.info("=" * 78)
    log.info("SAMPLE INPUT/OUTPUT (one-time sanity check) | screen=%r", ex.name)
    log.info("  n_genes=%d  total_hits=%d  encoder=%s  use_description=%s  "
             "rl_max_context=%s  max_tokens=%s",
             ex.n, int(ex.total_hits), "modernbert_text" if is_text else "embedding",
             use_desc, "n/a" if is_text else rl_max_context, max_tokens if is_text else "n/a")
    if not is_text:
        dnorm = float(ex.desc.float().norm().item())
        log.info("  desc_emb: dim=%d  L2norm=%.4f%s", ex.desc.numel(), dnorm,
                 "  <-- ZERO (description not contributing)" if dnorm == 0.0 else "")

    obs: list[int] = []
    obs_mask = torch.zeros(ex.n, dtype=torch.bool, device=device)
    for step in range(n_steps):
        n_obs = len(obs)
        remaining = ex.n - n_obs
        if remaining <= 0:
            break
        k_eff = min(batch_size, remaining)
        if is_text:
            scores = _score_text(net, ex, [obs], tokenizer, max_tokens)[0]
            txt = render_context_text(
                ex.desc_text if use_desc else "",
                [ex.genes[p] for p in obs], [int(ex.hits[p].item()) for p in obs])
            ntok = len(tokenizer(txt, truncation=True, max_length=max_tokens)["input_ids"])
            fed = (f"ctx_tokens={ntok}/{max_tokens}"
                   f"{' (TRUNCATED)' if ntok >= max_tokens else ''}")
            if step in (0, n_steps // 2):
                log.info("    [step %d rendered context preview] %s", step,
                         (txt[:300] + " ...") if len(txt) > 300 else txt)
        else:
            cp = torch.tensor(obs, device=device, dtype=torch.long).unsqueeze(0)
            if n_obs > rl_max_context:
                keep = torch.randperm(n_obs, device=device)[:rl_max_context]
                cp = cp[:, keep]
            ctx_idx = ex.gene_ids.unsqueeze(0).gather(1, cp) if n_obs else \
                torch.zeros(1, 0, dtype=torch.long, device=device)
            ctx_hit = ex.hits.unsqueeze(0).gather(1, cp).long() if n_obs else \
                torch.zeros(1, 0, dtype=torch.long, device=device)
            pad = torch.zeros(1, cp.shape[1] if n_obs else 0, dtype=torch.bool, device=device)
            rep = net.encode(ex.desc.unsqueeze(0), ctx_idx, ctx_hit, pad)
            scores = net.score_ids(rep, ex.gene_ids.unsqueeze(0))[0]
            ctx_hits_fed = int(ctx_hit.sum().item())
            fed = (f"ctx_fed={ctx_idx.shape[1]}/{n_obs}"
                   f"{' (TRUNCATED random subset)' if n_obs > rl_max_context else ''}"
                   f"  ctx_hits_fed={ctx_hits_fed}")
        top = torch.topk(scores.masked_fill(obs_mask, _NEG_INF), k_eff).indices
        top5 = top[:5].tolist()
        preview = ", ".join(f"{ex.genes[p]}{'*' if ex.hits[p].item() else ''}" for p in top5)
        n_hits_batch = int(ex.hits[top].sum().item())
        log.info("  step %2d | observed=%4d | %s | pick %d -> hits=%d | top5: %s",
                 step, n_obs, fed, k_eff, n_hits_batch, preview)
        for p in top.tolist():
            obs.append(p)
            obs_mask[p] = True
    log.info("(* = true hit) %s", "=" * 64)
    if was_training:
        net.train()


# ---------------------------------------------------------------------------
# Differentiable rollout + GRPO loss (one screen, G rollouts)
# ---------------------------------------------------------------------------


def rollout_group(
    net: RankerNet,
    ref: RankerNet,
    ex: _TensorExample,
    *,
    group_size: int,
    n_steps: int,
    batch_size: int,
    gamma: float,
    temperature: float,
    rl_max_context: int,
    kl_coef: float,
    ent_coef: float,
    terminal_coef: float,
    reward_mode: str = "telescope",
    aux_hit_coef: float = 0.0,
    baseline_net: "RankerNet | None" = None,
    adv_std_floor: float = 1e-6,
    adv_clip: float = 0.0,
    warm_pos: torch.Tensor | None = None,
    n_warm_rounds: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll the policy ``group_size`` times on one screen and return
    ``(loss, stats)`` for a GRPO-style group-normalized update.

    Warm start: when ``warm_pos`` (gene positions GLM acquired in its first
    ``n_warm_rounds`` rounds) is given, those genes are pre-observed (with their
    true labels) for every rollout and the policy only acts for the remaining
    ``n_steps - n_warm_rounds`` rounds. The terminal ``n_hits_vs_random`` is still
    computed over the full budget, so reward credits the hybrid GLM->transformer
    handoff while gradients flow only through the transformer's rounds.

    ``reward_mode``:

    - ``telescope`` (default): per-step reward = the *increment* of cumulative
      ``n_hits_vs_random`` (``nvr_t - nvr_{t-1}``). The undiscounted return then
      telescopes to exactly the deployed final ``n_hits_vs_random``, so
      "reward up" means "metric up". No separate terminal bonus.
    - ``context_delta``: per-step reward = (hits in the policy's context-informed
      batch) - (hits the **frozen reference's no-context/static top-k** would pick
      from the same remaining pool). This *forces context use*: a policy that
      ignores observations earns ~0, and the static baseline is the frozen
      reference (not the live policy), so reward can't be inflated by degrading
      the policy's own static ranking. Targets exactly the "does context help"
      signal the offline context-value diagnostic measured.
    - ``adjusted``: per-step ``batch_random_adjusted_hit_rate`` (dense) plus a
      ``terminal_coef``-weighted terminal ``n_hits_vs_random`` bonus. Kept for
      comparison; note the summed per-step ratios can dominate and decouple from
      the cumulative metric.

    Advantages are group-normalized across the group at each step; the loss adds
    a KL anchor to ``ref`` and an entropy bonus.
    """
    device = ex.desc.device
    G, n = group_size, ex.n
    desc_b = ex.desc.unsqueeze(0).expand(G, -1)
    gene_ids_b = ex.gene_ids.unsqueeze(0).expand(G, n)
    hits_b = ex.hits.unsqueeze(0).expand(G, n)
    total_hits = torch.full((G,), ex.total_hits, device=device)

    obs_mask = torch.zeros(G, n, dtype=torch.bool, device=device)
    ctx_pos = torch.zeros(G, 0, dtype=torch.int64, device=device)
    hits_observed = torch.zeros(G, device=device)
    prev_nvr = torch.zeros(G, device=device)

    # Warm start: pre-observe GLM's acquired genes (shared across the group).
    # Budget accounting: GLM was *budgeted* batch_size genes per round but may
    # acquire fewer (shortfall). Like the canonical HitsAUC metric (and GLM's own
    # n_hits_vs_random), the NVR denominator charges the REQUESTED budget
    # (n_warm_rounds * batch_size) for the GLM rounds, not the actual gene count -
    # so under-supply is penalized, not rewarded with a smaller random baseline.
    n_warm = 0
    warm_budget = 0
    if warm_pos is not None and warm_pos.numel() > 0 and n_warm_rounds > 0:
        wp = warm_pos.to(device=device, dtype=torch.long)
        obs_mask[:, wp] = True
        ctx_pos = wp.unsqueeze(0).expand(G, wp.numel()).contiguous()
        hits_observed = hits_observed + ex.hits[wp].sum()
        n_warm = int(wp.numel())
        warm_budget = n_warm_rounds * batch_size
        if reward_mode == "telescope":
            denom0 = (min(warm_budget, n) / n) * ex.total_hits
            if denom0 > 0:
                prev_nvr = prev_nvr + float(hits_observed[0].item()) / denom0
    steps_remaining = max(0, n_steps - (n_warm_rounds if n_warm else 0))

    def _frac_budget(n_obs_now: int) -> float:
        b = (n_obs_now - n_warm) + warm_budget
        return min(max(b, n_obs_now), n) / n if n else 0.0

    # Frozen baseline's static (no-context) scores: the ungameable baseline for
    # the context_delta reward. Fixed across steps (empty context), detached.
    # ``context_delta_reset`` periodically refreshes ``baseline_net`` to the
    # current policy so the baseline tracks the policy (keeps rewards small and
    # within-group variance healthy instead of saturating vs a stale reference).
    ctx_delta = reward_mode in ("context_delta", "context_delta_reset")
    bn = baseline_net if baseline_net is not None else ref
    ref_nc_scores = None
    if ctx_delta:
        with torch.no_grad():
            e_i = torch.zeros(G, 0, dtype=torch.int64, device=device)
            e_p = torch.zeros(G, 0, dtype=torch.bool, device=device)
            ref_nc_scores = bn.score_ids(bn.encode(desc_b, e_i, e_i, e_p), gene_ids_b)

    step_logp: list[torch.Tensor] = []
    step_kl: list[torch.Tensor] = []
    step_ent: list[torch.Tensor] = []
    step_reward: list[torch.Tensor] = []
    step_aux: list[torch.Tensor] = []

    n_obs = n_warm
    for _ in range(steps_remaining):
        remaining = n - n_obs
        if remaining <= 0:
            break
        k_eff = min(batch_size, remaining)

        raw_scores = _encode_group(net, desc_b, gene_ids_b, hits_b, ctx_pos, rl_max_context)
        # Auxiliary self-supervised signal: predict the held-out genes' own hit
        # labels from the observed context (reuses the policy's forward; gives V
        # + encoder a dense gradient every rollout, no teacher needed).
        if aux_hit_coef > 0:
            held = (~obs_mask).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores, hits_b.float(), reduction="none") * held
            step_aux.append(bce.sum(1) / held.sum(1).clamp_min(1.0))
        scores = (raw_scores / temperature).masked_fill(obs_mask, _NEG_INF)
        logp_all = torch.log_softmax(scores, dim=1)
        p_all = logp_all.exp()
        ent = -(p_all * logp_all).sum(1)

        with torch.no_grad():
            ref_scores = _encode_group(ref, desc_b, gene_ids_b, hits_b, ctx_pos, rl_max_context)
            ref_scores = (ref_scores / temperature).masked_fill(obs_mask, _NEG_INF)
            ref_logp = torch.log_softmax(ref_scores, dim=1)
        kl = (p_all * (logp_all - ref_logp)).sum(1)

        # Sample k_eff without replacement (Gumbel-top-k == Plackett-Luce).
        u = torch.rand_like(scores).clamp_min(1e-12)
        gumbel = -torch.log(-torch.log(u))
        perturbed = (scores + gumbel).masked_fill(obs_mask, -float("inf"))
        _, chosen = torch.topk(perturbed, k_eff, dim=1)        # (G, k_eff), PL order

        chosen_logp = torch.gather(logp_all, 1, chosen)        # (G, k_eff)
        chosen_p = chosen_logp.exp()
        cum_before = torch.cumsum(chosen_p, dim=1) - chosen_p  # exclusive prefix mass
        denom = (1.0 - cum_before).clamp_min(1e-8)
        step_lp = (chosen_logp - torch.log(denom)).sum(1)      # ordered-PL log-prob

        chosen_hits = torch.gather(hits_b, 1, chosen)
        n_hits_batch = chosen_hits.sum(1)

        if reward_mode == "adjusted":
            remaining_hits = (total_hits - hits_observed).clamp_min(0.0)
            exp_rand = remaining_hits * k_eff / remaining
            reward = torch.where(exp_rand > 0, n_hits_batch / exp_rand,
                                 torch.zeros_like(exp_rand))
        elif ctx_delta:
            base = ref_nc_scores.masked_fill(obs_mask, _NEG_INF)   # same remaining pool
            base_top = torch.topk(base, k_eff, dim=1).indices
            h_noctx = torch.gather(hits_b, 1, base_top).sum(1)
            reward = n_hits_batch - h_noctx
        else:
            # nvr_terminal: no per-step shaping; the only reward is the final
            # cumulative n_hits_vs_random injected after the loop.
            reward = torch.zeros_like(n_hits_batch)

        hits_observed = hits_observed + n_hits_batch
        obs_mask = obs_mask.scatter(1, chosen, True)
        ctx_pos = torch.cat([ctx_pos, chosen], dim=1)
        n_obs += k_eff

        if reward_mode == "telescope":
            # Increment of cumulative n_hits_vs_random; sum over steps == final.
            denom_now = _frac_budget(n_obs) * total_hits
            nvr_now = torch.where(denom_now > 0, hits_observed / denom_now,
                                  torch.zeros_like(denom_now))
            reward = nvr_now - prev_nvr
            prev_nvr = nvr_now

        step_logp.append(step_lp)
        step_kl.append(kl)
        step_ent.append(ent)
        step_reward.append(reward.detach())

    if not step_logp:
        return torch.zeros((), device=device), {}

    R = torch.stack(step_reward, dim=1)                        # (G, T) detached
    frac_budget = _frac_budget(n_obs)
    denom_term = total_hits * frac_budget
    nvr = torch.where(denom_term > 0, hits_observed / denom_term, torch.zeros_like(denom_term))
    if reward_mode == "adjusted":
        R[:, -1] = R[:, -1] + terminal_coef * nvr.detach()
    elif reward_mode == "nvr_terminal":
        R[:, -1] = R[:, -1] + nvr.detach()

    # Discounted returns-to-go, then per-step group-normalized advantages.
    T = R.shape[1]
    ret = torch.zeros_like(R)
    running = torch.zeros(G, device=device)
    for t in reversed(range(T)):
        running = R[:, t] + gamma * running
        ret[:, t] = running
    adv = _group_advantage(ret, adv_std_floor, adv_clip)

    logp = torch.stack(step_logp, dim=1)
    kl_t = torch.stack(step_kl, dim=1)
    ent_t = torch.stack(step_ent, dim=1)

    pg_loss = -(adv * logp).mean()
    kl_loss = kl_t.mean()
    ent_loss = -ent_t.mean()
    loss = pg_loss + kl_coef * kl_loss + ent_coef * ent_loss

    aux_val = 0.0
    if aux_hit_coef > 0 and step_aux:
        aux_loss = torch.stack(step_aux, dim=1).mean()
        loss = loss + aux_hit_coef * aux_loss
        aux_val = float(aux_loss.item())

    stats = {
        "loss": float(loss.item()),
        "pg_loss": float(pg_loss.item()),
        "kl": float(kl_loss.item()),
        "entropy": float(ent_t.mean().item()),
        "reward_mean": float(R.mean().item()),
        "nvr_mean": float(nvr.mean().item()),
        "aux_hit_bce": aux_val,
        "adv_absmax": float(adv.abs().max().item()),
        "ret_std_min": float(ret.std(0).clamp_min(0).min().item()),
    }
    return loss, stats


def rollout_group_text(
    net: RankerNet,
    ref: RankerNet,
    ex: _TensorExample,
    *,
    tokenizer: Any,
    max_tokens: int,
    group_size: int,
    n_steps: int,
    batch_size: int,
    gamma: float,
    temperature: float,
    kl_coef: float,
    ent_coef: float,
    terminal_coef: float,
    reward_mode: str,
    aux_hit_coef: float = 0.0,
    baseline_net: "RankerNet | None" = None,
    adv_std_floor: float = 1e-6,
    adv_clip: float = 0.0,
    loss_scale: float = 1.0,
    warm_pos: torch.Tensor | None = None,
    n_warm_rounds: int = 0,
) -> dict[str, float]:
    """Memory-safe GRPO rollout for the ``modernbert_text`` encoder.

    Warm start (see :func:`rollout_group`): pre-observe GLM's first
    ``n_warm_rounds`` acquired genes (``warm_pos``) for every rollout and act for
    the remaining rounds; terminal NVR is over the full budget.

    The text encoder must re-tokenize the rendered context every step, so a
    single-pass graph over ``n_steps`` x ``group_size`` ModernBERT forwards would
    OOM. Instead this runs **two passes**: pass 1 (no grad) samples the
    trajectory and computes group-normalized advantages; pass 2 re-encodes each
    step **with grad** and calls ``backward()`` per step, so only one step's
    activations are held at a time. Gradients accumulate into ``net`` (scaled by
    ``loss_scale``); the caller does ``opt.step()``. Returns stats only.
    """
    device = ex.gene_ids.device
    G, n = group_size, ex.n
    obs_mask = torch.zeros(G, n, dtype=torch.bool, device=device)
    obs_pos: list[list[int]] = [[] for _ in range(G)]
    hits_b = ex.hits.unsqueeze(0).expand(G, n)
    total_hits = torch.full((G,), ex.total_hits, device=device)
    hits_observed = torch.zeros(G, device=device)
    prev_nvr = torch.zeros(G, device=device)

    # Warm start: pre-observe GLM's acquired genes (shared across the group).
    # NVR denominator charges GLM's REQUESTED budget (n_warm_rounds * batch_size),
    # not the actual gene count, matching the canonical HitsAUC metric (see
    # rollout_group).
    n_warm = 0
    warm_budget = 0
    if warm_pos is not None and warm_pos.numel() > 0 and n_warm_rounds > 0:
        wp = warm_pos.to(device=device, dtype=torch.long)
        warm_list = wp.tolist()
        obs_mask[:, wp] = True
        for g in range(G):
            obs_pos[g].extend(warm_list)
        hits_observed = hits_observed + ex.hits[wp].sum()
        n_warm = int(wp.numel())
        warm_budget = n_warm_rounds * batch_size
        if reward_mode == "telescope":
            denom0 = (min(warm_budget, n) / n) * ex.total_hits
            if denom0 > 0:
                prev_nvr = prev_nvr + float(hits_observed[0].item()) / denom0
    steps_remaining = max(0, n_steps - (n_warm_rounds if n_warm else 0))

    def _frac_budget(n_obs_now: int) -> float:
        b = (n_obs_now - n_warm) + warm_budget
        return min(max(b, n_obs_now), n) / n if n else 0.0

    ctx_delta = reward_mode in ("context_delta", "context_delta_reset")
    bn = baseline_net if baseline_net is not None else ref
    ref_nc_scores = None
    if ctx_delta:
        with torch.no_grad():
            ref_nc_scores = _score_text(bn, ex, [[] for _ in range(G)], tokenizer, max_tokens)

    # Pass 1: sample the trajectory (no grad). Record per-step context snapshots,
    # chosen actions, masks, and rewards.
    snap_obs: list[list[list[int]]] = []
    snap_mask: list[torch.Tensor] = []
    chosen_steps: list[torch.Tensor] = []
    step_reward: list[torch.Tensor] = []
    n_obs = n_warm
    with torch.no_grad():
        for _ in range(steps_remaining):
            remaining = n - n_obs
            if remaining <= 0:
                break
            k_eff = min(batch_size, remaining)
            snap_obs.append([list(op) for op in obs_pos])
            snap_mask.append(obs_mask.clone())

            scores = _score_text(net, ex, obs_pos, tokenizer, max_tokens)
            scores = (scores / temperature).masked_fill(obs_mask, _NEG_INF)
            u = torch.rand_like(scores).clamp_min(1e-12)
            perturbed = (scores + (-torch.log(-torch.log(u)))).masked_fill(obs_mask, -float("inf"))
            _, chosen = torch.topk(perturbed, k_eff, dim=1)
            chosen_steps.append(chosen)

            n_hits_batch = torch.gather(hits_b, 1, chosen).sum(1)
            if reward_mode == "adjusted":
                remaining_hits = (total_hits - hits_observed).clamp_min(0.0)
                exp_rand = remaining_hits * k_eff / remaining
                reward = torch.where(exp_rand > 0, n_hits_batch / exp_rand,
                                     torch.zeros_like(exp_rand))
            elif ctx_delta:
                base = ref_nc_scores.masked_fill(obs_mask, _NEG_INF)
                base_top = torch.topk(base, k_eff, dim=1).indices
                reward = n_hits_batch - torch.gather(hits_b, 1, base_top).sum(1)
            else:
                # nvr_terminal: only the final cumulative NVR (injected below).
                reward = torch.zeros_like(n_hits_batch)

            hits_observed = hits_observed + n_hits_batch
            for g in range(G):
                cl = chosen[g].tolist()
                obs_pos[g].extend(cl)
                obs_mask[g, cl] = True
            n_obs += k_eff

            if reward_mode == "telescope":
                denom_now = _frac_budget(n_obs) * total_hits
                nvr_now = torch.where(denom_now > 0, hits_observed / denom_now,
                                      torch.zeros_like(denom_now))
                reward = nvr_now - prev_nvr
                prev_nvr = nvr_now
            step_reward.append(reward)

    if not chosen_steps:
        return {}

    R = torch.stack(step_reward, dim=1)
    frac_budget = _frac_budget(n_obs)
    denom_term = total_hits * frac_budget
    nvr = torch.where(denom_term > 0, hits_observed / denom_term, torch.zeros_like(denom_term))
    if reward_mode == "adjusted":
        R[:, -1] = R[:, -1] + terminal_coef * nvr
    elif reward_mode == "nvr_terminal":
        R[:, -1] = R[:, -1] + nvr
    T = R.shape[1]
    ret = torch.zeros_like(R)
    running = torch.zeros(G, device=device)
    for t in reversed(range(T)):
        running = R[:, t] + gamma * running
        ret[:, t] = running
    adv = _group_advantage(ret, adv_std_floor, adv_clip)

    # Pass 2: re-encode each step WITH grad and backward per step (bounded memory).
    tot_pg = tot_kl = tot_ent = tot_aux = 0.0
    for t in range(T):
        raw_scores = _score_text(net, ex, snap_obs[t], tokenizer, max_tokens)
        scores = (raw_scores / temperature).masked_fill(snap_mask[t], _NEG_INF)
        logp_all = torch.log_softmax(scores, dim=1)
        p_all = logp_all.exp()
        ent = -(p_all * logp_all).sum(1)
        with torch.no_grad():
            ref_scores = _score_text(ref, ex, snap_obs[t], tokenizer, max_tokens)
            ref_scores = (ref_scores / temperature).masked_fill(snap_mask[t], _NEG_INF)
            ref_logp = torch.log_softmax(ref_scores, dim=1)
        kl = (p_all * (logp_all - ref_logp)).sum(1)

        chosen = chosen_steps[t]
        chosen_logp = torch.gather(logp_all, 1, chosen)
        chosen_p = chosen_logp.exp()
        denom = (1.0 - (torch.cumsum(chosen_p, dim=1) - chosen_p)).clamp_min(1e-8)
        step_lp = (chosen_logp - torch.log(denom)).sum(1)

        pg = -(adv[:, t] * step_lp).mean()
        loss_t = pg + kl_coef * kl.mean() + ent_coef * (-ent.mean())
        if aux_hit_coef > 0:
            held = (~snap_mask[t]).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores, hits_b.float(), reduction="none") * held
            aux = (bce.sum(1) / held.sum(1).clamp_min(1.0)).mean()
            loss_t = loss_t + aux_hit_coef * aux
            tot_aux += float(aux.item())
        (loss_t * (loss_scale / T)).backward()
        tot_pg += float(pg.item()); tot_kl += float(kl.mean().item()); tot_ent += float(ent.mean().item())

    return {
        "loss": (tot_pg + kl_coef * tot_kl - ent_coef * tot_ent) / T,
        "pg_loss": tot_pg / T,
        "kl": tot_kl / T,
        "entropy": tot_ent / T,
        "reward_mean": float(R.mean().item()),
        "nvr_mean": float(nvr.mean().item()),
        "aux_hit_bce": tot_aux / T,
        "adv_absmax": float(adv.abs().max().item()),
        "ret_std_min": float(ret.std(0).clamp_min(0).min().item()),
    }


# ---------------------------------------------------------------------------
# Evaluation rollouts + explore/exploit baselines (no grad)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _policy_scores(
    net: RankerNet,
    ex: _TensorExample,
    obs_pos: list[int],
    *,
    temperature: float,
    rl_max_context: int,
    hits_only: bool = False,
) -> torch.Tensor:
    """Policy scores over all genes given observed positions (1-row group)."""
    device = ex.desc.device
    if hits_only:
        obs_pos = [p for p in obs_pos if ex.hits[p] > 0]
    cp = torch.tensor([obs_pos], dtype=torch.int64, device=device) if obs_pos \
        else torch.zeros(1, 0, dtype=torch.int64, device=device)
    desc_b = ex.desc.unsqueeze(0)
    gene_ids_b = ex.gene_ids.unsqueeze(0)
    hits_b = ex.hits.unsqueeze(0)
    scores = _encode_group(net, desc_b, gene_ids_b, hits_b, cp, rl_max_context)
    return (scores / temperature)[0]


@torch.no_grad()
def _knn_scores(net: RankerNet, ex: _TensorExample, obs_hit_pos: list[int]) -> torch.Tensor:
    """AL-kNN baseline over the *learned* gene embeddings: cosine similarity to
    the observed hits (cold start falls back to the per-gene bias)."""
    emb = net.gene_emb(ex.gene_ids)                            # (n, d)
    emb = torch.nn.functional.normalize(emb, dim=1)
    if not obs_hit_pos:
        return net.gene_bias[ex.gene_ids]
    h = emb[torch.tensor(obs_hit_pos, device=emb.device)]
    sim = emb @ h.t()                                          # (n, H)
    return sim.max(dim=1).values


@torch.no_grad()
def _rollout_metric(
    score_fn,
    ex: _TensorExample,
    *,
    n_steps: int,
    batch_size: int,
    stochastic: bool = False,
    temperature: float = 1.0,
    warm_pos: Sequence[int] | None = None,
    n_warm_rounds: int = 0,
    n_library: int | None = None,
    adj_budget: int | None = None,
) -> tuple[float, float]:
    """Greedy (or sampled) top-k rollout using ``score_fn(obs_pos, obs_hit_pos) ->
    scores (n,)``; returns ``(raw n_hits_vs_random, domain-adjusted NVR)``.

    The adjusted NVR matches the full-genome table metric: picks are classified
    in-library (positions ``< n_library``) vs out-of-library-but-in-universe
    (forgiven), charged against ``adj_budget`` (default ``n_steps * batch_size``),
    normalized by the library size. ``n_library`` defaults to ``ex.n`` (so
    non-full-genome examples fall back to a library-only number).

    Warm start: ``warm_pos`` genes (GLM's first ``n_warm_rounds`` acquisitions) are
    pre-observed with their true labels and the policy acts for the remaining
    ``n_steps - n_warm_rounds`` rounds; NVR is over the full budget."""
    device = ex.desc.device
    n = ex.n
    obs_mask = torch.zeros(n, dtype=torch.bool, device=device)
    obs_pos: list[int] = []
    obs_hit_pos: list[int] = []
    hits_observed = 0.0
    n_warm = 0
    if warm_pos is not None and len(warm_pos) > 0 and n_warm_rounds > 0:
        for c in warm_pos:
            c = int(c)
            if not bool(obs_mask[c]):
                obs_mask[c] = True
                obs_pos.append(c)
                if ex.hits[c] > 0:
                    obs_hit_pos.append(c)
        hits_observed = float(ex.hits[torch.tensor(obs_pos, device=device)].sum().item())
        n_warm = len(obs_pos)
    n_obs = n_warm
    for _ in range(max(0, n_steps - (n_warm_rounds if n_warm else 0))):
        remaining = n - n_obs
        if remaining <= 0:
            break
        k_eff = min(batch_size, remaining)
        scores = score_fn(obs_pos, obs_hit_pos).masked_fill(obs_mask, _NEG_INF)
        if stochastic:
            probs = torch.softmax(scores / temperature, dim=0).clamp_min(1e-12)
            chosen = torch.multinomial(probs, k_eff, replacement=False)
        else:
            chosen = torch.topk(scores, k_eff).indices
        for c in chosen.tolist():
            obs_mask[c] = True
            obs_pos.append(c)
            if ex.hits[c] > 0:
                obs_hit_pos.append(c)
        hits_observed += float(ex.hits[chosen].sum().item())
        n_obs += k_eff
    # Budget: charge GLM's REQUESTED warm budget (n_warm_rounds * batch_size), not
    # the actual gene count, so the random denominator matches the canonical
    # HitsAUC metric and GLM's own n_hits_vs_random (under-supply is penalized).
    warm_budget = (n_warm_rounds * batch_size) if n_warm else 0
    raw_budget = min(max((n_obs - n_warm) + warm_budget, n_obs), n) if n else 0
    frac_budget = raw_budget / n if n else 0.0
    raw = n_hits_vs_random_value(hits_observed, frac_budget, int(ex.total_hits))
    # Domain-adjusted NVR (paper/table metric): classify obs picks in-lib vs
    # out-of-lib-in-universe (forgiven), charged against the full requested budget.
    n_lib = ex.n if n_library is None else n_library
    n1 = sum(1 for p in obs_pos if p < n_lib)
    n2 = len(obs_pos) - n1
    bud = (n_steps * batch_size) if adj_budget is None else adj_budget
    adj = adjusted_ef_value(hits_observed, n1, n2, n_lib, int(ex.total_hits), bud)
    return raw, adj


@torch.no_grad()
def _mmr_rollout_metric(
    net: RankerNet, ex: _TensorExample, *, n_steps: int, batch_size: int,
    temperature: float, rl_max_context: int, lam: float = 0.5,
) -> float:
    """Diversified (MMR) greedy on the policy scores: within each batch, penalize
    genes similar (learned-embedding cosine) to those already picked this batch.
    A headroom probe for whether explore/exploit diversification helps."""
    device = ex.desc.device
    n = ex.n
    emb = torch.nn.functional.normalize(net.gene_emb(ex.gene_ids), dim=1)
    obs_mask = torch.zeros(n, dtype=torch.bool, device=device)
    obs_pos: list[int] = []
    hits_observed = 0.0
    n_obs = 0
    for _ in range(n_steps):
        remaining = n - n_obs
        if remaining <= 0:
            break
        k_eff = min(batch_size, remaining)
        base = _policy_scores(
            net, ex, obs_pos, temperature=temperature, rl_max_context=rl_max_context
        ).masked_fill(obs_mask, _NEG_INF)
        picked: list[int] = []
        cur = base.clone()
        for _j in range(k_eff):
            c = int(torch.argmax(cur).item())
            picked.append(c)
            cur[c] = _NEG_INF
            cur = cur - lam * (emb @ emb[c]).clamp_min(0.0)
            cur = cur.masked_fill(obs_mask, _NEG_INF)
        for c in picked:
            obs_mask[c] = True
            obs_pos.append(c)
        idx = torch.tensor(picked, device=device)
        hits_observed += float(ex.hits[idx].sum().item())
        n_obs += k_eff
    frac_budget = n_obs / n if n else 0.0
    return n_hits_vs_random_value(hits_observed, frac_budget, int(ex.total_hits))


@torch.no_grad()
def _policy_scores_text(
    net: RankerNet,
    ex: _TensorExample,
    obs_pos: list[int],
    *,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    hits_only: bool = False,
) -> torch.Tensor:
    """Text-encoder policy scores over all genes given observed positions."""
    if hits_only:
        obs_pos = [p for p in obs_pos if ex.hits[p] > 0]
    scores = _score_text(net, ex, [obs_pos], tokenizer, max_tokens)
    return (scores / temperature)[0]


@torch.no_grad()
def evaluate_policy(
    net: RankerNet,
    examples: list[_TensorExample],
    *,
    n_steps: int,
    batch_size: int,
    temperature: float,
    rl_max_context: int,
    seed: int = 0,
    baselines: bool = True,
    tokenizer: Any = None,
    max_tokens: int = 2048,
    warm_start: "WarmStart | None" = None,
    warm_n: int = 0,
) -> dict[str, float]:
    """Mean episodic ``n_hits_vs_random`` for the policy and explore/exploit
    baselines over ``examples``.

    When ``warm_start`` is given, each screen is seeded with GLM's first ``warm_n``
    acquired genes (true labels) and the policy/baselines continue for the
    remaining ``n_steps - warm_n`` rounds - the hybrid-handoff deployment metric.
    Val/test traces are single per screen so seeding is deterministic.
    """
    if not examples:
        return {}
    net.eval()
    is_text = net.is_text
    rng = random.Random(seed)

    def warm_of(ex):
        if warm_start is None or warm_n <= 0:
            return None
        wp = warm_start.sample_positions(ex.name, warm_n, rng)
        return None if wp is None else wp.tolist()

    def policy_score_fn(ex, hits_only=False):
        if is_text:
            return lambda op, ohp, ex=ex: _policy_scores_text(
                net, ex, op, tokenizer=tokenizer, max_tokens=max_tokens,
                temperature=temperature, hits_only=hits_only)
        return lambda op, ohp, ex=ex: _policy_scores(
            net, ex, op, temperature=temperature, rl_max_context=rl_max_context,
            hits_only=hits_only)

    pol_r, pol_a, pol_ho, rnd, knn, div = [], [], [], [], [], []
    bud = n_steps * batch_size
    for ex in examples:
        wp = warm_of(ex)
        r, a = _rollout_metric(
            policy_score_fn(ex), ex, n_steps=n_steps, batch_size=batch_size,
            warm_pos=wp, n_warm_rounds=warm_n, n_library=ex.n_library, adj_budget=bud,
        )
        pol_r.append(r); pol_a.append(a)
        if not baselines:
            continue
        # baselines kept in raw NVR terms (diagnostic continuity); [0] = raw
        pol_ho.append(_rollout_metric(
            policy_score_fn(ex, hits_only=True), ex, n_steps=n_steps, batch_size=batch_size,
            warm_pos=wp, n_warm_rounds=warm_n, n_library=ex.n_library, adj_budget=bud,
        )[0])
        rnd.append(_rollout_metric(
            lambda op, ohp, ex=ex: torch.rand(ex.n, device=ex.gene_ids.device),
            ex, n_steps=n_steps, batch_size=batch_size,
            warm_pos=wp, n_warm_rounds=warm_n, n_library=ex.n_library, adj_budget=bud,
        )[0])
        knn.append(_rollout_metric(
            lambda op, ohp, ex=ex: _knn_scores(net, ex, ohp),
            ex, n_steps=n_steps, batch_size=batch_size,
            warm_pos=wp, n_warm_rounds=warm_n, n_library=ex.n_library, adj_budget=bud,
        )[0])
        if not is_text and warm_n <= 0:
            div.append(_mmr_rollout_metric(
                net, ex, n_steps=n_steps, batch_size=batch_size,
                temperature=temperature, rl_max_context=rl_max_context,
            ))

    # nvr_adj = domain-adjusted policy NVR (the paper/table metric, used for
    # checkpoint selection); n_hits_vs_random kept for continuity.
    out = {"n_hits_vs_random": float(np.mean(pol_r)),
           "nvr_adj": float(np.mean(pol_a)),
           "n_screens": float(len(pol_r))}
    if baselines:
        out.update({
            "nvr_hits_only_ctx": float(np.mean(pol_ho)),
            "nvr_random": float(np.mean(rnd)),
            "nvr_knn": float(np.mean(knn)),
            "context_value_vs_hits_only": float(np.mean(pol_r) - np.mean(pol_ho)),
            "lift_vs_knn": float(np.mean(pol_r) - np.mean(knn)),
        })
        if div:
            out["nvr_diversified"] = float(np.mean(div))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_rl_training(
    *,
    init_checkpoint: str | Path | None = None,
    init_ckpt_file: str = "model.pt",
    from_scratch: bool = False,
    encoder_type: str = "transformer",
    d_model: int = 384,
    d_gene: int = 16,
    d_hit: int = 64,
    nhead: int = 2,
    num_layers: int = 3,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
    bert_model: str = "answerdotai/ModernBERT-base",
    disable_gene_bias: bool = False,
    aux_hit_coef: float = 0.0,
    out_dir: str | Path | None = None,
    run_name: str | None = None,
    train_screen_set: str = "train",
    train_size: int | None = None,
    eval_screen_set: str = "paper_validation",
    eval_size: int | None = None,
    eval_screens: int = 30,
    train_eval_screens: int = 30,
    test_eval_set: str | None = None,
    epochs: int = 3,
    screens_per_step: int = 1,
    group_size: int = 8,
    n_steps: int = 10,
    batch_size: int = 100,
    gamma: float = 1.0,
    temperature: float = 1.0,
    kl_coef: float = 0.1,
    ent_coef: float = 0.01,
    terminal_coef: float = 1.0,
    reward_mode: str = "telescope",
    ctx_reset_epochs: int = 25,
    adv_std_floor: float = 1e-6,
    adv_clip: float = 0.0,
    rl_max_context: int = 512,
    rl_max_tokens: int | None = None,
    lr: float = 1e-5,
    bert_lr: float = 1e-6,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    eval_every: int = 1,
    split_half_eval: bool = False,
    text_backend: str = "auto",
    use_description: bool | None = None,
    # None => wandb uses the account default (or $WANDB_ENTITY).
    wandb_entity: str | None = None,
    wandb_project: str = "assayloop-amortized-ranker",
    wandb_mode: str = "disabled",
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    reuse_wandb: bool = False,
    warm_start_glm_train: str | Path | None = None,
    warm_start_n: str | int = 0,
    warm_start_eval_dir: str | Path | None = None,
    warm_start_eval_prefix: str | None = None,
    eval_warm_n: int = 0,
    handoff_select: str = "off",
    handoff_val_prefix: str = "sweep-6cd2d623-",
    handoff_test_prefix: str = "sweep-a79fd5ce-",
    handoff_trace_dirs: list[str] | None = None,
    handoff_warm_n: int = 3,
    seed: int = 0,
    device: str | None = "auto",
    verbose: bool = True,
    full_genome: bool = False,
    save_every: int = 0,
) -> dict[str, Any]:
    """RL-fine-tune a warm-started ranker; save artifacts; return a summary.

    GLM warm-start handoff: if ``warm_start_glm_train`` is set, each training
    rollout pre-observes GLM-5.1's first ``n`` acquired genes (from stored traces,
    with true labels) and the policy continues for ``n_steps - n`` rounds.
    ``warm_start_n`` is either an int (fixed n) or ``"random"`` (V2: sample
    ``n ~ U{0..n_steps-1}`` per screen per epoch). For in-training validation under
    the handoff, set ``warm_start_eval_dir``/``warm_start_eval_prefix`` (the shared
    val GLM run dirs) and ``eval_warm_n`` (fixed n used for checkpoint selection).
    """
    from ..tasks import load_screens

    if reward_mode not in ("telescope", "context_delta", "context_delta_reset",
                           "adjusted", "nvr_terminal"):
        raise ValueError(
            f"Unknown reward_mode {reward_mode!r}; pick "
            "telescope|context_delta|context_delta_reset|adjusted|nvr_terminal.")
    _seed_all(seed)

    # Distributed (torchrun): shard screens across ranks, average grads each step.
    is_dist, rank, world_size, local_rank = _dist_info()
    is_main = rank == 0
    if is_dist:
        use_cuda = torch.cuda.is_available() and device != "cpu"
        backend = "nccl" if use_cuda else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = torch.device("cpu")
        log.info("[rank %d/%d] distributed RL (%s) on %s", rank, world_size, backend, dev)
    else:
        dev = _pick_device(device)

    from_scratch = bool(from_scratch) or init_checkpoint is None

    run_name = run_name or f"ranker-rl-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out_dir = Path(out_dir) if out_dir else (config.OUTPUT_PATH / "rankers" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_screens = _subsample(load_screens(target_set=train_screen_set), train_size, seed)
    eval_screens_all = _subsample(load_screens(target_set=eval_screen_set), eval_size, seed)

    if not full_genome:
        log.warning("--full-genome not set; auto-enabling it so the adjusted-NVR "
                    "selection metric matches the paper table (candidates = f2 universe).")
        full_genome = True
    universe_genes = None
    if full_genome:
        from ..tasks import gene_universe
        universe_genes = gene_universe(load_screens(target_set="paper_test"),
                                       min_screen_freq=2)
        log.info("full_genome: universe of %d genes (freq >= 2)", len(universe_genes))

    if from_scratch:
        # Cold start: build vocab + a fresh RankerNet from CLI arch args (no
        # warm-start weights). Used to learn the bilinear ranker by RL alone
        # (+ the auxiliary self-supervised hit loss).
        ckpt = None
        cfg_d = {}
        if use_description is None:
            use_description = True
        is_text = encoder_type == "modernbert_text"
        is_bert = encoder_type in ("modernbert_embed", "modernbert_text")
        text_max_tokens = int(rl_max_tokens) if rl_max_tokens else 2048
        # Vocab over train + eval (+ public test for cold gene slots), matching
        # supervised training so eval/transfer genes have ids.
        vocab_screens = [train_screens, eval_screens_all]
        try:
            vocab_screens.append(load_screens(target_set="paper_test"))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load paper test screens for vocab (%s).", e)
        vocab = GeneVocab.build(vocab_screens)
        if use_description and not is_text:
            text_dim = te.get_text_embedder(
                text_backend if text_backend != "auto" else "auto"
            ).dim
        else:
            text_dim = 1
        arch = dict(
            vocab_size=len(vocab), text_dim=text_dim, d_model=d_model, d_gene=d_gene,
            d_hit=d_hit, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, encoder_type=encoder_type,
            bert_model=bert_model, freeze_bert=False, bert_pool="cls",
            text_max_tokens=text_max_tokens, use_description=use_description,
            freeze_factors=False, disable_gene_bias=disable_gene_bias,
        )
        net = RankerNet(RankerConfig(**arch)).to(dev)
        if kl_coef != 0.0:
            log.info("from_scratch: forcing kl_coef=0 (a KL anchor to a random "
                     "init is meaningless/harmful).")
            kl_coef = 0.0
        ref = copy.deepcopy(net).to(dev)  # only used if a future reward needs it
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        log.info("Cold-start %s ranker (vocab=%d, d_gene=%d, disable_gene_bias=%s, "
                 "use_description=%s); RL from random init, aux_hit_coef=%g.",
                 encoder_type, len(vocab), d_gene, disable_gene_bias,
                 use_description, aux_hit_coef)
    else:
        ckpt = Path(init_checkpoint)
        cfg_d = json.loads((ckpt / "config.json").read_text())
        arch = cfg_d["arch"]
        # No-description ablation: inherit from the checkpoint unless overridden.
        ckpt_use_desc = bool(arch.get("use_description", True))
        if use_description is None:
            use_description = ckpt_use_desc
        elif use_description != ckpt_use_desc:
            log.warning("Overriding checkpoint use_description=%s with %s; warm-start "
                        "weights may not match this ablation.", ckpt_use_desc, use_description)
        arch["use_description"] = use_description
        encoder_type = cfg_d.get("encoder_type", arch.get("encoder_type", "transformer"))
        is_text = encoder_type == "modernbert_text"
        is_bert = encoder_type in ("modernbert_embed", "modernbert_text")
        # Cap the per-step token budget for RL: the checkpoint may have been trained
        # at a large context (e.g. 8192), but re-tokenizing that every step x group x
        # 2 passes is prohibitive. Truncation keeps the description + observed hits
        # (rendered first), so a smaller cap mainly drops the least-informative
        # non-hit symbols.
        text_max_tokens = int(rl_max_tokens) if rl_max_tokens else int(arch.get("text_max_tokens", 2048))
        # Warm start: reuse the checkpoint's vocab so gene ids match the table.
        vocab = GeneVocab.load(ckpt / "vocab.json")
        net = RankerNet(RankerConfig(**arch)).to(dev)
        net.load_state_dict(torch.load(ckpt / init_ckpt_file, map_location=dev))
        ref = copy.deepcopy(net).to(dev)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        log.info("Warm-started %s ranker from %s (vocab=%d)", encoder_type, ckpt, len(vocab))
        if arch.get("freeze_factors"):
            log.info("BPMF-anchored checkpoint: gene factors V + bias stay frozen; RL "
                     "optimizes only the encoder's screen-latent inference (û).")

    # No-context baseline for the context_delta rewards. For plain context_delta
    # it is the fixed reference; for context_delta_reset it is refreshed to the
    # current policy every ``ctx_reset_epochs`` epochs so the "vs static" target
    # tracks the policy (keeps rewards small + group-variance healthy, avoiding
    # the saturation-driven destructive update seen with a stale reference).
    baseline_ref = None
    if reward_mode in ("context_delta", "context_delta_reset"):
        baseline_ref = copy.deepcopy(net).to(dev)
        baseline_ref.eval()
        for p in baseline_ref.parameters():
            p.requires_grad_(False)
        if reward_mode == "context_delta_reset":
            log.info("context_delta_reset: baseline refreshed to the policy every "
                     "%d epoch(s).", max(1, ctx_reset_epochs))

    tokenizer = None
    if is_text:
        # Text encoder reads the rendered context directly; no description embeddings.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(arch.get("bert_model", "answerdotai/ModernBERT-base"))
        log.info("Text encoder; tokenizer %s, max_tokens=%d (no description embeddings).",
                 arch.get("bert_model"), text_max_tokens)
        train_examples = build_examples(train_screens, {}, vocab, require_desc_emb=False, universe_genes=universe_genes)
        eval_examples = build_examples(eval_screens_all, {}, vocab, require_desc_emb=False, universe_genes=universe_genes)
    elif not use_description:
        # Embedding encoder with the description ablated: the model uses a
        # constant query token, so skip the (remote) embedder entirely.
        log.info("use_description=False: skipping description embeddings; "
                 "the encoder uses a constant query token.")
        train_examples = build_examples(train_screens, {}, vocab, require_desc_emb=False, universe_genes=universe_genes)
        eval_examples = build_examples(eval_screens_all, {}, vocab, require_desc_emb=False, universe_genes=universe_genes)
    else:
        text_backend_use = text_backend if text_backend != "auto" else cfg_d.get("text_backend", "auto")
        embedder = te.get_text_embedder(
            text_backend_use,
            **({"model": cfg_d["text_model"]} if cfg_d.get("text_model") else {}),
        )
        log.info("Text embedder: %s (dim=%d)", embedder.name, embedder.dim)
        train_desc = te.embed_screens(train_screens, embedder)
        eval_desc = te.embed_screens(eval_screens_all, embedder)
        train_examples = build_examples(train_screens, train_desc, vocab, universe_genes=universe_genes)
        eval_examples = build_examples(eval_screens_all, eval_desc, vocab, universe_genes=universe_genes)
    if not train_examples:
        raise RuntimeError("No training examples built (check embeddings/screens).")

    train_t = [_TensorExample(ex, dev) for ex in train_examples]
    eval_t = [_TensorExample(ex, dev) for ex in eval_examples]
    rng = random.Random(seed)
    train_eval_t = (rng.sample(train_t, train_eval_screens)
                    if 0 < train_eval_screens < len(train_t) else train_t)
    eval_t_used = (rng.sample(eval_t, eval_screens)
                   if 0 < eval_screens < len(eval_t) else eval_t)

    # Split-half cross-validation for checkpoint selection (LOPO mode).
    eval_half_a: list = []
    eval_half_b: list = []
    if split_half_eval and is_main:
        sh_rng = random.Random(seed + 7)
        sh_list = list(eval_t_used)
        sh_rng.shuffle(sh_list)
        mid = len(sh_list) // 2
        eval_half_a = sh_list[:mid]
        eval_half_b = sh_list[mid:]
        log.info("split-half eval: %d screens -> half-A=%d, half-B=%d",
                 len(sh_list), len(eval_half_a), len(eval_half_b))

    # Handoff-based checkpoint selection (option B): evaluate an LLM->ranker
    # handoff on val (+ test) each epoch and save the best-handoff-val checkpoint.
    handoff_on = str(handoff_select).lower() != "off"
    if handoff_on and not test_eval_set:
        test_eval_set = "public"
        log.info("handoff eval on: defaulting test_eval_set='public' for handoff-test.")

    # Optional held-out *test* eval each epoch (e.g. test_eval_set="public") so
    # the val->test generalization can be tracked across the RL trajectory rather
    # than only at the best-val checkpoint.
    test_t_used: list = []
    if test_eval_set:
        test_screens_all = _subsample(load_screens(target_set=test_eval_set), eval_size, seed)
        if is_text or not use_description:
            test_examples = build_examples(
                test_screens_all, {}, vocab, require_desc_emb=False,
                universe_genes=universe_genes)
        else:
            test_desc = te.embed_screens(test_screens_all, embedder)
            test_examples = build_examples(test_screens_all, test_desc, vocab, universe_genes=universe_genes)
        test_t = [_TensorExample(ex, dev) for ex in test_examples]
        test_t_used = (rng.sample(test_t, eval_screens)
                       if 0 < eval_screens < len(test_t) else test_t)
        log.info("Per-epoch test eval enabled: %d screens from target_set=%r.",
                 len(test_t_used), test_eval_set)

    # GLM warm-start: load + index the per-screen acquisition traces.
    ws_train: WarmStart | None = None
    ws_train_random = False
    ws_train_n = 0
    if warm_start_glm_train:
        ws_train = WarmStart.from_train_jsonl(warm_start_glm_train).index_examples(train_t)
        if isinstance(warm_start_n, str) and warm_start_n.strip().lower() == "random":
            ws_train_random = True
        else:
            ws_train_n = int(warm_start_n)
        covered = sum(1 for ex in train_t if ws_train.has(ex.name))
        log.info("Warm-start (train): %d/%d screens have GLM traces; n=%s.",
                 covered, len(train_t), "random" if ws_train_random else ws_train_n)
    ws_eval: WarmStart | None = None
    if warm_start_eval_dir and warm_start_eval_prefix and eval_warm_n > 0:
        ws_eval = WarmStart.from_run_dirs(
            warm_start_eval_dir, warm_start_eval_prefix).index_examples(eval_t)
        covered = sum(1 for ex in eval_t_used if ws_eval.has(ex.name))
        log.info("Warm-start (eval): %d/%d val screens have GLM traces; eval n=%d.",
                 covered, len(eval_t_used), eval_warm_n)

    # Handoff WarmStarts (option B): LLM traces on val + test, merged across dirs.
    ws_handoff_val: WarmStart | None = None
    ws_handoff_test: WarmStart | None = None
    if handoff_on:
        _dirs = ([Path(d) for d in handoff_trace_dirs] if handoff_trace_dirs
                 else [config.OUTPUT_PATH / "runs", _SHARED_RUNS_DIR])

        def _merged_ws(prefix, examples):
            tr: dict = {}
            for d in _dirs:
                tr.update(load_run_traces(d, prefix))
            return WarmStart(tr).index_examples(examples)

        ws_handoff_val = _merged_ws(handoff_val_prefix, eval_t)
        cov_v = sum(1 for ex in eval_t_used if ws_handoff_val.has(ex.name))
        log.info("Handoff-val (%s, %s): %d/%d val screens covered; warm_n=%d.",
                 handoff_select, handoff_val_prefix, cov_v, len(eval_t_used), handoff_warm_n)
        if test_t_used:
            ws_handoff_test = _merged_ws(handoff_test_prefix, test_t)
            cov_t = sum(1 for ex in test_t_used if ws_handoff_test.has(ex.name))
            log.info("Handoff-test (%s, %s): %d/%d test screens covered.",
                     handoff_select, handoff_test_prefix, cov_t, len(test_t_used))

    def _warm_for(ex: _TensorExample, epoch: int) -> tuple[torch.Tensor | None, int]:
        """Per-screen warm-start positions + #rounds consumed for this epoch."""
        if ws_train is None:
            return None, 0
        avail = ws_train.max_rounds(ex.name)
        if avail <= 0:
            return None, 0
        r = random.Random(hash((ex.name, epoch, seed)) & 0xFFFFFFFF)
        n_req = r.randint(0, n_steps - 1) if ws_train_random else ws_train_n
        n_eff = max(0, min(n_req, avail, n_steps - 1))
        if n_eff <= 0:
            return None, 0
        pos = ws_train.sample_positions(ex.name, n_eff, r)
        if pos is None or pos.size == 0:
            return None, 0
        return torch.from_numpy(pos).to(dev), n_eff

    # Optimizer (discriminative LR for the BERT backbone, if any).
    if is_bert:
        bb = [p for n, p in net.named_parameters() if n.startswith("bert.") and p.requires_grad]
        head = [p for n, p in net.named_parameters() if not n.startswith("bert.") and p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": bb, "lr": bert_lr}, {"params": head, "lr": lr}],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay,
        )

    # wandb (optional; rank 0 only under DDP).
    wb = None
    if is_main:
        try:
            import wandb  # noqa: PLC0415

            _wb_cfg = {
                    "method": "rl_grpo",
                    "init_checkpoint": str(ckpt) if ckpt else "scratch",
                    "from_scratch": from_scratch, "aux_hit_coef": aux_hit_coef,
                    "disable_gene_bias": disable_gene_bias,
                    "arch": arch, "encoder_type": encoder_type, "world_size": world_size,
                    "epochs": epochs, "group_size": group_size, "n_steps": n_steps,
                    "batch_size": batch_size, "gamma": gamma, "temperature": temperature,
                    "kl_coef": kl_coef, "ent_coef": ent_coef, "terminal_coef": terminal_coef,
                    "reward_mode": reward_mode, "ctx_reset_epochs": ctx_reset_epochs,
                    "adv_std_floor": adv_std_floor, "adv_clip": adv_clip,
                    "lr": lr, "bert_lr": bert_lr, "weight_decay": weight_decay,
                    "rl_max_context": rl_max_context, "n_train": len(train_t),
                    "n_eval": len(eval_t_used), "seed": seed,
                    "train_screen_set": train_screen_set, "eval_screen_set": eval_screen_set,
            }
            if reuse_wandb and wandb.run is not None:
                # A sweep agent (or caller) already owns the run; attach to it
                # rather than starting a new one.
                wb = wandb.run
                wb.config.update(_wb_cfg, allow_val_change=True)
            else:
                wb = wandb.init(
                    entity=wandb_entity, project=wandb_project, name=run_name, mode=wandb_mode,
                    group=wandb_group, tags=(wandb_tags or None), config=_wb_cfg,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("wandb unavailable (%s); training without it.", e)

    def _eval(examples, baselines=True, warm=None, warm_n=0):
        return evaluate_policy(
            net, examples, n_steps=n_steps, batch_size=batch_size,
            temperature=temperature, rl_max_context=rl_max_context, seed=seed,
            baselines=baselines, tokenizer=tokenizer, max_tokens=text_max_tokens,
            warm_start=warm, warm_n=warm_n,
        )

    # Baseline (warm-start) eval before any updates. Eval is local (no
    # collectives) and identical across ranks, so only rank 0 runs it.
    base_eval: dict[str, float] = {}
    base_handoff_val: dict[str, float] = {}
    base_handoff_test: dict[str, float] = {}
    if is_main:
        base_eval = _eval(eval_t_used, warm=ws_eval, warm_n=eval_warm_n)
        base_train = _eval(train_eval_t, baselines=False)
        base_test = _eval(test_t_used, baselines=False) if test_t_used else {}
        if ws_handoff_val is not None:
            base_handoff_val = _eval(eval_t_used, baselines=False,
                                     warm=ws_handoff_val, warm_n=handoff_warm_n)
        if ws_handoff_test is not None:
            base_handoff_test = _eval(test_t_used, baselines=False,
                                      warm=ws_handoff_test, warm_n=handoff_warm_n)
        log.info("[init] eval nvr_adj=%.3f (raw=%.3f, knn=%.3f) | train raw=%.3f%s",
                 base_eval.get("nvr_adj", float("nan")),
                 base_eval.get("n_hits_vs_random", float("nan")),
                 base_eval.get("nvr_knn", float("nan")),
                 base_train.get("n_hits_vs_random", float("nan")),
                 (f" | handoff_val nvr_adj={base_handoff_val.get('nvr_adj', float('nan')):.3f}"
                  if base_handoff_val else ""))
        if wb is not None:
            wb.log({**{f"eval/init_{k}": v for k, v in base_eval.items()},
                    **{f"handoff_val/init_{k}": v for k, v in base_handoff_val.items()},
                    **{f"handoff_test/init_{k}": v for k, v in base_handoff_test.items()}})

    init_metric = base_eval.get("nvr_adj", -float("inf"))   # selection on adjusted NVR
    best_val = init_metric
    best_state = copy.deepcopy(net.state_dict()) if is_main else None
    # Handoff-val selection (option B): best-handoff-val checkpoint.
    init_handoff = base_handoff_val.get("nvr_adj", -float("inf"))
    best_handoff_val = init_handoff
    best_handoff_state = (copy.deepcopy(net.state_dict())
                          if (is_main and ws_handoff_val is not None) else None)
    # Split-half best-checkpoint tracking.
    best_val_a = -float("inf")
    best_val_b = -float("inf")
    best_state_a = copy.deepcopy(net.state_dict()) if (is_main and split_half_eval) else None
    best_state_b = copy.deepcopy(net.state_dict()) if (is_main and split_half_eval) else None
    history: list[dict[str, Any]] = (
        [{"phase": "init", "eval": base_eval, "train": base_train,
          **({"eval_test": base_test} if base_test else {}),
          **({"handoff_val": base_handoff_val} if base_handoff_val else {}),
          **({"handoff_test": base_handoff_test} if base_handoff_test else {})}]
        if is_main else [])
    global_step = 0

    rl_kw = dict(
        group_size=group_size, n_steps=n_steps, batch_size=batch_size, gamma=gamma,
        temperature=temperature, kl_coef=kl_coef, ent_coef=ent_coef,
        terminal_coef=terminal_coef, reward_mode=reward_mode, aux_hit_coef=aux_hit_coef,
        baseline_net=baseline_ref, adv_std_floor=adv_std_floor, adv_clip=adv_clip,
    )

    trainable = [p for p in net.parameters() if p.requires_grad]

    if is_main and verbose:
        sample_ex = next((train_t[j] for j in range(len(train_t))
                          if train_t[j].n >= 2 and train_t[j].total_hits >= 1), None)
        if sample_ex is not None:
            try:
                _log_sample_io(
                    net, sample_ex, n_steps=n_steps, batch_size=batch_size,
                    rl_max_context=rl_max_context, tokenizer=tokenizer,
                    max_tokens=text_max_tokens, is_text=is_text, log=log)
            except Exception as e:  # noqa: BLE001
                log.warning("sample I/O dump failed (%s); continuing.", e)

    def _accum_window(idxs: list[int], epoch: int) -> list[dict[str, float]]:
        """Zero grads and accumulate gradients for a window of screens (no
        optimizer step). The text encoder uses a memory-safe two-pass rollout
        that backwards internally; the embedding encoders stack per-screen losses
        and backward once. The caller does the (optionally all-reduced) step."""
        opt.zero_grad()
        screens = [train_t[j] for j in idxs
                   if train_t[j].n >= 2 and train_t[j].total_hits >= 1]
        if not screens:
            return []
        stats_list: list[dict[str, float]] = []
        if is_text:
            for ex in screens:
                wp, n_ws = _warm_for(ex, epoch)
                st = rollout_group_text(
                    net, ref, ex, tokenizer=tokenizer, max_tokens=text_max_tokens,
                    loss_scale=1.0 / len(screens), warm_pos=wp, n_warm_rounds=n_ws,
                    **rl_kw)
                if st:
                    stats_list.append(st)
        else:
            losses: list[torch.Tensor] = []
            for ex in screens:
                wp, n_ws = _warm_for(ex, epoch)
                loss, st = rollout_group(
                    net, ref, ex, rl_max_context=rl_max_context,
                    warm_pos=wp, n_warm_rounds=n_ws, **rl_kw)
                if st:
                    losses.append(loss)
                    stats_list.append(st)
            if losses:
                torch.stack(losses).mean().backward()
        return stats_list

    for epoch in range(epochs):
        net.train()
        # Refresh the context_delta_reset baseline to the current policy. All
        # ranks hold identical weights (grads are all-reduced), so this stays in
        # lockstep without a broadcast.
        if (reward_mode == "context_delta_reset" and baseline_ref is not None
                and epoch > 0 and ctx_reset_epochs > 0 and epoch % ctx_reset_epochs == 0):
            baseline_ref.load_state_dict(net.state_dict())
            baseline_ref.eval()
            for p in baseline_ref.parameters():
                p.requires_grad_(False)
            if is_main:
                log.info("[e%d] context_delta_reset: baseline refreshed to policy.", epoch)
        # Every rank shuffles identically, then takes its own contiguous shard.
        # Dropping the remainder keeps the per-rank screen (and thus window/step)
        # count identical, so the per-window all-reduce stays in lockstep.
        order = list(range(len(train_t)))
        random.Random(seed + epoch).shuffle(order)
        if is_dist:
            per = len(order) // world_size
            order = order[rank * per:(rank + 1) * per]
        ep_stats: list[dict[str, float]] = []
        ep_gn: list[float] = []
        for w in range(0, len(order), screens_per_step):
            stats_list = _accum_window(order[w:w + screens_per_step], epoch)
            # Under DDP every rank must step every window (collective symmetry),
            # even if its window was empty (zero-filled grads contribute nothing).
            # clip_grad_norm_ returns the pre-clip total grad norm (logged so the
            # destructive-update spikes that can wreck an unanchored policy are
            # observable).
            gn: float | None = None
            if is_dist:
                _all_reduce_grads(trainable, world_size)
                gn = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip))
                opt.step()
            elif stats_list:
                gn = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip))
                opt.step()
            ep_stats.extend(stats_list)
            if gn is not None and np.isfinite(gn):
                ep_gn.append(gn)
            if stats_list:
                global_step += 1
                if wb is not None:
                    last = stats_list[-1]
                    logd = {f"train/{k}": last[k] for k in last}
                    if gn is not None:
                        logd["train/grad_norm"] = gn
                    wb.log(logd | {"step": global_step})

        row: dict[str, Any] = {"epoch": epoch}
        if ep_gn:
            row["train_grad_norm"] = float(np.mean(ep_gn))
            row["train_grad_norm_max"] = float(np.max(ep_gn))
        if ep_stats:
            row["train_reward_mean"] = float(np.mean([s["reward_mean"] for s in ep_stats]))
            row["train_nvr_mean"] = float(np.mean([s["nvr_mean"] for s in ep_stats]))
            row["train_kl"] = float(np.mean([s["kl"] for s in ep_stats]))

        # Eval + checkpoint selection on rank 0 only (weights are identical across
        # ranks after the averaged step). Other ranks wait at the barrier.
        if is_main and ((epoch % eval_every) == 0 or epoch == epochs - 1):
            ev = _eval(eval_t_used, warm=ws_eval, warm_n=eval_warm_n)
            tr = _eval(train_eval_t, baselines=False)
            row["eval"] = ev
            row["train_eval_nvr"] = tr.get("n_hits_vs_random")
            te_ev = _eval(test_t_used, baselines=False) if test_t_used else {}
            if te_ev:
                row["eval_test"] = te_ev
            # Handoff (option B): LLM->ranker handoff eval on val (+ test), select
            # the best-handoff-VAL checkpoint separately from the standalone best.
            hv: dict[str, float] = {}
            ht: dict[str, float] = {}
            if ws_handoff_val is not None:
                hv = _eval(eval_t_used, baselines=False, warm=ws_handoff_val, warm_n=handoff_warm_n)
                row["handoff_val"] = hv
                if ws_handoff_test is not None:
                    ht = _eval(test_t_used, baselines=False, warm=ws_handoff_test, warm_n=handoff_warm_n)
                    row["handoff_test"] = ht
                m_h = hv.get("nvr_adj", -float("inf"))
                if m_h > best_handoff_val:
                    best_handoff_val = m_h
                    best_handoff_state = copy.deepcopy(net.state_dict())
                    row["is_best_handoff"] = True
            metric = ev.get("nvr_adj", -float("inf"))
            if metric > best_val:
                best_val = metric
                best_state = copy.deepcopy(net.state_dict())
                row["is_best"] = True
            if save_every > 0 and epoch % save_every == 0:
                torch.save(copy.deepcopy(net.state_dict()),
                           out_dir / f"model_epoch_{epoch}.pt")
            # Split-half checkpoint tracking.
            if split_half_eval and eval_half_a and eval_half_b:
                ev_a = _eval(eval_half_a, baselines=False)
                ev_b = _eval(eval_half_b, baselines=False)
                m_a = ev_a.get("nvr_adj", -float("inf"))
                m_b = ev_b.get("nvr_adj", -float("inf"))
                if m_a > best_val_a:
                    best_val_a = m_a
                    best_state_a = copy.deepcopy(net.state_dict())
                if m_b > best_val_b:
                    best_val_b = m_b
                    best_state_b = copy.deepcopy(net.state_dict())
                row["eval_half_a_nvr"] = m_a
                row["eval_half_b_nvr"] = m_b
            wb_extra = {}
            if split_half_eval and "eval_half_a_nvr" in row:
                wb_extra["eval/half_a_nvr"] = row["eval_half_a_nvr"]
                wb_extra["eval/half_b_nvr"] = row["eval_half_b_nvr"]
            if wb is not None:
                wb.log({**{f"eval/{k}": v for k, v in ev.items()},
                        **{f"eval_test/{k}": v for k, v in te_ev.items()},
                        **{f"handoff_val/{k}": v for k, v in hv.items()},
                        **{f"handoff_test/{k}": v for k, v in ht.items()},
                        "train/eval_n_hits_vs_random": tr.get("n_hits_vs_random"),
                        **wb_extra,
                        "epoch": epoch, "step": global_step})
            if verbose:
                log.info(
                    "[e%d] train_reward=%.3f eval_n_vs_rand=%.3f (knn=%.3f, ho=%.3f) "
                    "train_n_vs_rand=%.3f%s",
                    epoch, row.get("train_reward_mean", float("nan")),
                    ev.get("n_hits_vs_random", float("nan")),
                    ev.get("nvr_knn", float("nan")),
                    ev.get("nvr_hits_only_ctx", float("nan")),
                    tr.get("n_hits_vs_random", float("nan")),
                    (f" test_n_vs_rand={te_ev.get('n_hits_vs_random', float('nan')):.3f}"
                     if te_ev else ""),
                )
        if is_main:
            history.append(row)
        if is_dist:
            dist.barrier()

    # Non-main ranks have done their share of the gradient work; only rank 0
    # evaluated / tracked best and writes artifacts. Tear down and return.
    if not is_main:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return {"rank": rank, "world_size": world_size}

    # Persist the best checkpoint + artifacts (eval-ranker / dashboard compatible).
    # Also keep the final-epoch weights: the held-out metric can dip below the
    # init early, so "best" may equal the init -- model_last.pt lets you eval the
    # actually-updated policy regardless.
    final_state = copy.deepcopy(net.state_dict())
    improved = best_val > init_metric + 1e-9
    if not improved:
        log.warning(
            "Best held-out n_hits_vs_random (%.4f) did not beat the warm-start init "
            "(%.4f): model.pt == the INIT weights. Eval model_last.pt to see the "
            "updated policy.", best_val, init_metric,
        )
    torch.save(final_state, out_dir / "model_last.pt")
    net.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    if best_handoff_state is not None:
        # best-on-handoff-val checkpoint (option B); may differ from model.pt
        torch.save(best_handoff_state, out_dir / "model_handoff.pt")
        log.info("Saved model_handoff.pt (best handoff-val nvr_adj=%.4f, init=%.4f).",
                 best_handoff_val, init_handoff)
    vocab.save(out_dir / "vocab.json")
    np.save(out_dir / "gene_embeddings.npy", net.gene_emb.weight.detach().cpu().numpy())
    (out_dir / "config.json").write_text(json.dumps({
        "arch": arch,
        "text_backend": cfg_d.get("text_backend", "none" if is_text else "auto"),
        "text_model": cfg_d.get("text_model"),
        "encoder_type": encoder_type,
        "bert_model": arch.get("bert_model", "answerdotai/ModernBERT-base"),
        "bert_pool": arch.get("bert_pool", "cls"),
        "text_max_tokens": arch.get("text_max_tokens", 2048),
        "rl": {
            "init_checkpoint": str(ckpt) if ckpt else "scratch",
            "from_scratch": from_scratch, "aux_hit_coef": aux_hit_coef,
            "disable_gene_bias": disable_gene_bias,
            "epochs": epochs, "group_size": group_size,
            "n_steps": n_steps, "batch_size": batch_size, "gamma": gamma,
            "temperature": temperature, "kl_coef": kl_coef, "ent_coef": ent_coef,
            "terminal_coef": terminal_coef, "reward_mode": reward_mode,
            "ctx_reset_epochs": ctx_reset_epochs,
            "adv_std_floor": adv_std_floor, "adv_clip": adv_clip,
            "lr": lr, "bert_lr": bert_lr,
            "rl_max_context": rl_max_context, "seed": seed,
            "train_screen_set": train_screen_set, "eval_screen_set": eval_screen_set,
            "wandb_group": wandb_group, "wandb_tags": wandb_tags or [],
            "full_genome": full_genome,
            "selection_metric": "nvr_adj",
            "warm_start": {
                "glm_train": str(warm_start_glm_train) if warm_start_glm_train else None,
                "n": "random" if ws_train_random else (ws_train_n if ws_train else 0),
                "eval_dir": str(warm_start_eval_dir) if warm_start_eval_dir else None,
                "eval_prefix": warm_start_eval_prefix,
                "eval_warm_n": eval_warm_n,
            },
            "handoff": {
                "select": handoff_select,
                "val_prefix": handoff_val_prefix if handoff_on else None,
                "test_prefix": handoff_test_prefix if handoff_on else None,
                "warm_n": handoff_warm_n if handoff_on else 0,
            },
        },
    }, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Split-half cross-validated NVR: evaluate best-on-A checkpoint on half-B and
    # vice versa; average is an unbiased held-out metric with no val leakage.
    split_half_nvr = None
    if split_half_eval and best_state_a is not None and best_state_b is not None:
        net.load_state_dict(best_state_a)
        net.eval()
        score_a_on_b = _eval(eval_half_b, baselines=False).get("nvr_adj", float("nan"))
        net.load_state_dict(best_state_b)
        net.eval()
        score_b_on_a = _eval(eval_half_a, baselines=False).get("nvr_adj", float("nan"))
        split_half_nvr = float(np.mean([score_a_on_b, score_b_on_a]))
        log.info("split-half CV: best-A-on-B=%.4f, best-B-on-A=%.4f, mean=%.4f",
                 score_a_on_b, score_b_on_a, split_half_nvr)
        # Restore best_state (full eval) for the saved model.pt.
        net.load_state_dict(best_state)
        net.eval()

    # Stability/perf summaries from the eval series (useful as sweep objectives:
    # final-window mean rewards configs that *stay* high; max_drop flags the kind
    # of single-epoch collapse seen with unanchored from-scratch RL).
    eval_series = [
        h["eval"]["nvr_adj"] for h in history
        if isinstance(h.get("eval"), dict) and h["eval"].get("nvr_adj") is not None
    ]
    last5_eval_mean = float(np.mean(eval_series[-5:])) if eval_series else None
    final_eval = float(eval_series[-1]) if eval_series else None
    max_eval_drop = (
        float(max((eval_series[i] - eval_series[i + 1] for i in range(len(eval_series) - 1)),
                  default=0.0))
        if len(eval_series) > 1 else 0.0
    )

    summary = {
        "out_dir": str(out_dir),
        "run_name": run_name,
        "init_checkpoint": str(ckpt) if ckpt else "scratch",
        "from_scratch": from_scratch,
        "aux_hit_coef": aux_hit_coef,
        "encoder_type": encoder_type,
        "reward_mode": reward_mode,
        # selection metric = domain-adjusted NVR (matches the paper table)
        "init_eval_nvr_adj": base_eval.get("nvr_adj"),
        "best_eval_nvr_adj": best_val if best_val != -float("inf") else None,
        "final_eval_nvr_adj": final_eval,
        "last5_eval_nvr_adj": last5_eval_mean,
        "init_eval_n_hits_vs_random": base_eval.get("n_hits_vs_random"),  # raw, reference
        "max_eval_drop": max_eval_drop,
        "best_beat_init": bool(improved),
        "init_eval_knn": base_eval.get("nvr_knn"),
        # handoff (option B) selection, if enabled
        "handoff_select": handoff_select,
        "init_handoff_val_nvr_adj": base_handoff_val.get("nvr_adj") if base_handoff_val else None,
        "best_handoff_val_nvr_adj": (best_handoff_val if best_handoff_state is not None
                                     and best_handoff_val != -float("inf") else None),
        "n_train": len(train_t),
        "n_eval": len(eval_t_used),
        "split_half_nvr": split_half_nvr,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wb is not None:
        try:
            wb.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            wb.finish()
        except Exception:  # noqa: BLE001
            pass
    log.info("Saved RL ranker to %s (best eval nvr_adj=%s, init=%s, beat_init=%s); "
             "model.pt=best, model_last.pt=final-epoch",
             out_dir, summary["best_eval_nvr_adj"],
             summary["init_eval_nvr_adj"], improved)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return summary


@torch.no_grad()
def per_screen_handoff_nvr(
    net: RankerNet,
    examples: list[_TensorExample],
    *,
    warm_start: "WarmStart",
    n_values: Sequence[int],
    n_steps: int,
    batch_size: int,
    temperature: float = 1.0,
    rl_max_context: int = 1024,
    tokenizer: Any = None,
    max_tokens: int = 2048,
    seed: int = 0,
) -> dict[str, dict[int, float]]:
    """Per-screen ``{name: {n: n_hits_vs_random}}`` for the GLM->transformer
    handoff over the requested ``n_values`` (greedy policy rollout). Used to build
    handoff-predictor labels (best n per screen / "continue GLM?" per round)."""
    net.eval()
    is_text = net.is_text
    rng = random.Random(seed)
    out: dict[str, dict[int, float]] = {}
    for ex in examples:
        if is_text:
            fn = lambda op, ohp, ex=ex: _policy_scores_text(
                net, ex, op, tokenizer=tokenizer, max_tokens=max_tokens,
                temperature=temperature)
        else:
            fn = lambda op, ohp, ex=ex: _policy_scores(
                net, ex, op, temperature=temperature, rl_max_context=rl_max_context)
        row: dict[int, float] = {}
        for n in n_values:
            wp = warm_start.sample_positions(ex.name, n, rng) if n > 0 else None
            wpl = wp.tolist() if wp is not None else None
            row[int(n)] = _rollout_metric(
                fn, ex, n_steps=n_steps, batch_size=batch_size,
                warm_pos=wpl, n_warm_rounds=int(n))[0]  # raw NVR (handoff-predictor labels)
        out[ex.name] = row
    return out


@torch.no_grad()
def screen_latent_after_n(
    net: RankerNet,
    ex: _TensorExample,
    warm_pos: Sequence[int] | None,
    *,
    rl_max_context: int = 1024,
    tokenizer: Any = None,
    max_tokens: int = 2048,
) -> np.ndarray:
    """The encoder's screen latent û (d_gene) after observing ``warm_pos`` genes -
    the per-round context feature for the sequential handoff predictor (B)."""
    device = ex.gene_ids.device
    obs = list(warm_pos) if warm_pos is not None else []
    if net.is_text:
        from .text_embed import render_context_text
        desc_text = ex.desc_text if net.cfg.use_description else ""
        txt = render_context_text(
            desc_text, [ex.genes[p] for p in obs], [int(ex.hits[p].item()) for p in obs])
        enc = tokenizer([txt], padding=True, truncation=True,
                        max_length=max_tokens, return_tensors="pt")
        u = net.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        return u[0].float().cpu().numpy()
    if obs:
        cp = torch.tensor([obs], dtype=torch.long, device=device)
        if cp.shape[1] > rl_max_context:
            keep = torch.randperm(cp.shape[1], device=device)[:rl_max_context]
            cp = cp[:, keep]
        ctx_idx = ex.gene_ids.unsqueeze(0).gather(1, cp)
        ctx_hit = ex.hits.unsqueeze(0).gather(1, cp).long()
        pad = torch.zeros(1, cp.shape[1], dtype=torch.bool, device=device)
    else:
        ctx_idx = torch.zeros(1, 0, dtype=torch.long, device=device)
        ctx_hit = torch.zeros(1, 0, dtype=torch.long, device=device)
        pad = torch.zeros(1, 0, dtype=torch.bool, device=device)
    u = net.encode(ex.desc.unsqueeze(0), ctx_idx, ctx_hit, pad)
    return u[0].float().cpu().numpy()


#: number of features produced per round by :func:`per_screen_round_evidence`.
ROUND_EVIDENCE_DIM = 6


@torch.no_grad()
def per_screen_round_evidence(
    examples: list[_TensorExample],
    *,
    warm_start: "WarmStart",
    n_max: int,
    batch_size: int,
    trace_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Per-screen ``{name: array[n_max, 6]}`` of *realized GLM hit performance*
    features for handoff rounds ``1..n_max`` (net-free; the Bayesian predictor's
    per-round evidence).

    Features per round ``j`` (computed on the cumulative-unique observed set, to
    match :func:`per_screen_handoff_nvr` / :func:`_rollout_metric`):

    0. round index ``j / n_max``
    1. round hit rate ``hits_j / batch_size``
    2. cumulative hit rate ``H_j / |U_j|``
    3. trend ``rate_j - rate_{j-1}``
    4. fraction of total hits found ``H_j / total_hits``
    5. remaining budget fraction ``(10*batch_size - |U_j|) / (10*batch_size)``

    Rounds beyond the trace length are padded as a "stall" (no new genes): the
    cumulative stats carry forward, round rate/trend are 0.
    """
    full_budget = float(max(10 * batch_size, 1))
    out: dict[str, np.ndarray] = {}
    for ex in examples:
        rounds = warm_start.round_positions(ex.name, trace_idx)
        feats = np.zeros((n_max, ROUND_EVIDENCE_DIM), dtype=np.float32)
        cum_obs = 0
        cum_hits = 0.0
        prev_rate = 0.0
        total_hits = float(ex.total_hits) or 1.0
        for j in range(n_max):
            new = rounds[j] if j < len(rounds) else np.empty(0, dtype=np.int64)
            if new.size:
                hits_j = float(ex.hits[torch.as_tensor(new, device=ex.hits.device)].sum().item())
            else:
                hits_j = 0.0
            cum_obs += int(new.size)
            cum_hits += hits_j
            round_rate = hits_j / float(batch_size) if batch_size else 0.0
            cum_rate = cum_hits / float(cum_obs) if cum_obs else 0.0
            feats[j, 0] = (j + 1) / float(max(n_max, 1))
            feats[j, 1] = round_rate
            feats[j, 2] = cum_rate
            feats[j, 3] = round_rate - prev_rate
            feats[j, 4] = cum_hits / total_hits
            feats[j, 5] = max(0.0, (full_budget - cum_obs) / full_budget)
            prev_rate = round_rate
        out[ex.name] = feats
    return out


__all__ = [
    "run_rl_training", "rollout_group", "evaluate_policy",
    "per_screen_handoff_nvr", "screen_latent_after_n",
    "per_screen_round_evidence", "ROUND_EVIDENCE_DIM",
]
