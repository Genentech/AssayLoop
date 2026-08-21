"""MAML meta-learning baseline for gene ranking.

Gradient-based adaptation at test time: takes K inner-loop gradient steps
on the revealed (gene, hit) observations to adapt the screen latent û,
then scores candidates via the bilinear head û · V_g + bias_g.

The meta-learned initialization (screen MLP + gene factors) is trained
by ``scripts/train_maml.py``.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation

log = logging.getLogger(__name__)


class ScreenMLP(nn.Module):
    """Maps screen description embedding to initial screen latent û₀."""

    def __init__(self, text_dim: int, d_gene: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_gene),
        )

    def forward(self, desc_emb: torch.Tensor) -> torch.Tensor:
        return self.net(desc_emb)


class MAMLRankerModel(Model):
    """MAML baseline: gradient-based adaptation of the screen latent.

    At each ``predict()`` call:
      1. Compute û₀ = screen_mlp(description_embedding)
      2. Take ``inner_steps`` gradient steps on û using BCE loss on
         the cumulative observations
      3. Score all candidates via û_adapted · V_g + bias_g

    Args:
        checkpoint: path to a directory with model.pt, vocab.json, config.json
        device: torch device
        inner_lr: learning rate for inner-loop adaptation
        inner_steps: number of gradient steps per predict() call
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "auto",
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        ckpt = Path(checkpoint)
        cfg = json.loads((ckpt / "config.json").read_text())

        from ..amortized.data import GeneVocab
        self.vocab = GeneVocab.load(ckpt / "vocab.json")

        state = torch.load(ckpt / "model.pt", map_location=self.device,
                          weights_only=False)

        self.d_gene = cfg["d_gene"]
        text_dim = cfg.get("text_dim", 1536)

        # Gene factors V (frozen)
        self.V = torch.tensor(state["V"], dtype=torch.float32,
                             device=self.device)
        self.bias = torch.tensor(state["bias"], dtype=torch.float32,
                                device=self.device)

        # Screen MLP (meta-learned initialization)
        self.screen_mlp = ScreenMLP(text_dim, self.d_gene,
                                    hidden=cfg.get("hidden", 128))
        self.screen_mlp.load_state_dict(state["screen_mlp"])
        self.screen_mlp.to(self.device)
        self.screen_mlp.eval()

        # Keep a copy of the meta-learned MLP weights for reset
        self._init_mlp_state = copy.deepcopy(self.screen_mlp.state_dict())

        # Text embedder
        self._text_backend = cfg.get("text_backend", "auto")
        self._embedder = None
        self._desc_cache: dict[str, torch.Tensor] = {}

        log.info("MAMLRankerModel: d_gene=%d, inner_steps=%d, inner_lr=%.4f, "
                 "vocab=%d, device=%s",
                 self.d_gene, self.inner_steps, self.inner_lr,
                 len(self.vocab), self.device)

    def _ensure_embedder(self):
        if self._embedder is None:
            import assayloop.amortized.text_embed as te
            self._embedder = te.get_text_embedder(self._text_backend)

    def _get_desc_emb(self, task_context: dict) -> torch.Tensor:
        name = task_context.get("dataset_name", "")
        if name in self._desc_cache:
            return self._desc_cache[name]
        self._ensure_embedder()
        from assayloop.amortized.text_embed import screen_description_text, embed_texts
        text = screen_description_text(task_context)
        emb = embed_texts([text], self._embedder)[0]
        t = torch.tensor(emb, dtype=torch.float32, device=self.device)
        self._desc_cache[name] = t
        return t

    def name(self) -> str:
        return "maml_ranker"

    def reset(self) -> None:
        self.screen_mlp.load_state_dict(self._init_mlp_state)

    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        ctx = task_context or {}

        # 1. Get description embedding and compute û₀
        desc_emb = self._get_desc_emb(ctx).unsqueeze(0)  # (1, text_dim)
        u0 = self.screen_mlp(desc_emb).squeeze(0)  # (d_gene,)

        # 2. Inner-loop adaptation on observations
        if observations and self.inner_steps > 0:
            obs_ids = []
            obs_hits = []
            for o in observations:
                vid = self.vocab.to_idx(str(o.candidate))
                if vid == 0:
                    continue
                label = o.label
                hit = bool(label.get("hit")) if isinstance(label, dict) else bool(label)
                obs_ids.append(vid)
                obs_hits.append(float(hit))

            if obs_ids:
                obs_ids_t = torch.tensor(obs_ids, device=self.device)
                obs_hits_t = torch.tensor(obs_hits, device=self.device)
                V_obs = self.V[obs_ids_t]  # (n_obs, d_gene)
                bias_obs = self.bias[obs_ids_t]  # (n_obs,)

                u = u0.clone().detach().requires_grad_(True)

                for _ in range(self.inner_steps):
                    logits = V_obs @ u + bias_obs  # (n_obs,)
                    loss = F.binary_cross_entropy_with_logits(logits, obs_hits_t)
                    grad = torch.autograd.grad(loss, u)[0]
                    u = u - self.inner_lr * grad
                u0 = u.detach()

        # 3. Score all candidates
        cand_ids = [self.vocab.to_idx(str(c)) for c in candidates]
        cand_ids_t = torch.tensor(cand_ids, device=self.device)
        V_cand = self.V[cand_ids_t]  # (n_cand, d_gene)
        bias_cand = self.bias[cand_ids_t]  # (n_cand,)
        scores_t = V_cand @ u0 + bias_cand  # (n_cand,)
        scores_np = scores_t.detach().cpu().numpy()

        score_map = {c: float(scores_np[i]) for i, c in enumerate(candidates)}
        return ModelPrediction(
            scores=score_map,
            metadata={"name": self.name(), "inner_steps": self.inner_steps,
                      "n_obs": len(observations)},
        )


__all__ = ["MAMLRankerModel"]
