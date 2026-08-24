"""Inference-time Model wrapper for the amortized gene ranker.

Loads a trained :class:`~assayloop.amortized.model.RankerNet` (from a
checkpoint directory) and exposes it through the assayloop
:class:`~assaybench.core.Model` interface so it can be driven by any
acquisition (greedy/UCB) inside the AL loop and evaluated like every other
run type.

It can also be built from in-memory components (used by the trainer to
evaluate the live model on the validation screens without round-tripping to
disk).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from assaybench.core.model import Model
from assaybench.core.types import ModelPrediction, Observation

log = logging.getLogger("assayloop.models.amortized_ranker")


def _pick_device(device: str | None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_ckpt(ckpt: Path, ckpt_file: str) -> Path:
    """The requested weights file, or an error that says what is there instead.

    Deliberately not a fallback: a directory holding only ``model_last.pt`` is
    not a directory holding ``model.pt``, and loading the final epoch when the
    caller asked for the best-on-validation one would change a reported number
    without saying so. But the two names differ by a suffix and mean different
    things, and the released checkpoint on HuggingFace ships only the second, so
    the bare ``FileNotFoundError`` torch raises is a bad first experience. Name
    the alternatives and let the caller choose.
    """
    path = ckpt / ckpt_file
    if path.exists():
        return path
    others = sorted(p.name for p in ckpt.glob("*.pt"))
    raise FileNotFoundError(
        f"{path} does not exist. " + (
            f"This checkpoint directory holds {', '.join(others)} -- pass "
            f"ckpt_file='{others[0]}' if that is the one you want. model.pt is "
            "the best-on-validation epoch and model_last.pt the final one; the "
            "paper's AssayLoop rows, and the released Genentech/assayformer "
            "checkpoint, use model_last.pt."
            if others else
            f"There are no .pt files in {ckpt} at all."
        )
    )


class AmortizedRankerModel(Model):
    """Score candidates with a trained context-conditioned ranker.

    Construct either from a checkpoint dir (``checkpoint=...``) or from
    in-memory components (``net``, ``vocab``, ``embedder``).

    Args:
        checkpoint: path to a directory containing ``model.pt``, ``vocab.json``,
            ``config.json`` (as written by ``amortized.train``).
        ckpt_file: which weights file in that directory to load. ``model.pt`` is
            the best-on-validation epoch, ``model_last.pt`` the final one; the
            paper's AssayLoop rows and the released ``Genentech/assayformer``
            checkpoint use ``model_last.pt``, which has to be asked for.
        net / vocab / embedder: in-memory alternative to ``checkpoint``.
        desc_emb_by_name: optional ``{dataset_name: vector}`` to avoid
            recomputing description embeddings during evaluation.
        device: "auto" | "cpu" | "cuda".
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        ckpt_file: str = "model.pt",
        net: Any = None,
        vocab: Any = None,
        embedder: Any = None,
        desc_emb_by_name: dict[str, np.ndarray] | None = None,
        desc_text_by_name: dict[str, str] | None = None,
        tokenizer: Any = None,
        ignore_context: bool = False,
        device: str | None = "auto",
    ):
        from ..amortized.data import GeneVocab
        from ..amortized.model import RankerConfig, RankerNet

        self.device = _pick_device(device)
        self._desc_emb_by_name = desc_emb_by_name or {}
        self._desc_text_by_name = desc_text_by_name or {}
        self._emb_cache = None
        self._meta: dict[str, Any] = {}
        self.tokenizer = tokenizer
        # Diagnostic ablation: when True, the model ignores the accumulated AL
        # observations and always scores from the description alone (a static
        # one-shot ranking). Lets eval-ranker measure the marginal value of
        # the context.
        self.ignore_context = bool(ignore_context)

        if checkpoint is not None:
            ckpt = Path(checkpoint)
            cfg_d = json.loads((ckpt / "config.json").read_text())
            arch = cfg_d["arch"]
            self.vocab = GeneVocab.load(ckpt / "vocab.json")
            self.net = RankerNet(RankerConfig(**arch))
            state = torch.load(_resolve_ckpt(ckpt, ckpt_file),
                               map_location=self.device)
            self.net.load_state_dict(state)
            self._text_backend = cfg_d.get("text_backend", "auto")
            self._text_model = cfg_d.get("text_model")
            self.embedder = embedder
            self._ckpt = str(ckpt)
        else:
            if net is None or vocab is None:
                raise ValueError("Provide either checkpoint= or (net=, vocab=).")
            self.net = net
            self.vocab = vocab
            self.embedder = embedder
            self._text_backend = "auto"
            self._text_model = None
            self._ckpt = ""

        self.encoder_type = getattr(self.net.cfg, "encoder_type", "transformer")
        # The text encoder needs a tokenizer; load it lazily from the saved
        # backbone name if one was not supplied by the trainer.
        if self.encoder_type == "modernbert_text" and self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.net.cfg.bert_model)

        self.net.to(self.device)
        self.net.eval()

    # -- lazy text embedder (only needed when desc emb not precomputed) -----

    def _ensure_embedder(self):
        if self.embedder is None:
            from ..amortized import text_embed as te

            kwargs = {}
            if self._text_model:
                kwargs["model"] = self._text_model
            self.embedder = te.get_text_embedder(self._text_backend, **kwargs)
        return self.embedder

    def _desc_emb(self, ctx: dict[str, Any]) -> np.ndarray:
        name = ctx.get("dataset_name")
        if name and name in self._desc_emb_by_name:
            return np.asarray(self._desc_emb_by_name[name], dtype=np.float32)
        from ..amortized import text_embed as te

        if self._emb_cache is None:
            self._emb_cache = te.EmbeddingCache()
        text = te.screen_description_text(ctx)
        emb = te.embed_texts([text], self._ensure_embedder(), cache=self._emb_cache)
        return emb[0]

    def _desc_text(self, ctx: dict[str, Any]) -> str:
        name = ctx.get("dataset_name")
        if name and name in self._desc_text_by_name:
            return self._desc_text_by_name[name]
        from ..amortized import text_embed as te

        return te.screen_description_text(ctx)

    def _encode_text_context(
        self, ctx: dict[str, Any], observations: list[Observation]
    ) -> torch.Tensor:
        """Render description + observations to text, tokenize, encode -> (1, d_gene)."""
        from ..amortized.text_embed import render_context_text

        syms, hits = [], []
        for o in observations:
            lbl = o.label
            hit = bool(lbl.get("hit")) if isinstance(lbl, dict) else bool(lbl)
            syms.append(str(o.candidate))
            hits.append(1 if hit else 0)
        desc_text = self._desc_text(ctx) if self.net.cfg.use_description else ""
        text = render_context_text(desc_text, syms, hits)
        enc = self.tokenizer(
            [text], padding=True, truncation=True,
            max_length=self.net.cfg.text_max_tokens, return_tensors="pt",
        )
        return self.net.encode_text(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        )

    def name(self) -> str:
        return "amortized_ranker"

    def reset(self) -> None:
        self._meta = {}

    @torch.no_grad()
    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        ctx = task_context or {}
        if not candidates:
            return ModelPrediction(scores={}, metadata={"name": self.name()})

        # Diagnostic ablation: drop the AL context entirely (static ranking).
        obs = [] if self.ignore_context else observations
        n_context = len(obs)
        if self.encoder_type == "modernbert_text":
            desc_repr = self._encode_text_context(ctx, obs)  # (1, d_gene)
        else:
            if not self.net.cfg.use_description:
                # No-description ablation: the model ignores the desc embedding,
                # so skip the (potentially remote) embedding call and feed a zero.
                text_dim = self.net.desc_proj.in_features
                desc = torch.zeros(1, text_dim, device=self.device)
            else:
                desc = torch.from_numpy(self._desc_emb(ctx)).float().unsqueeze(0).to(self.device)

            # Build the observed-gene context from cumulative observations.
            ctx_ids: list[int] = []
            ctx_hits: list[int] = []
            for o in obs:
                lbl = o.label
                hit = bool(lbl.get("hit")) if isinstance(lbl, dict) else bool(lbl)
                ctx_ids.append(self.vocab.to_idx(str(o.candidate)))
                ctx_hits.append(1 if hit else 0)

            if ctx_ids:
                ci = torch.tensor([ctx_ids], dtype=torch.int64, device=self.device)
                ch = torch.tensor([ctx_hits], dtype=torch.int64, device=self.device)
                cpad = torch.zeros(1, len(ctx_ids), dtype=torch.bool, device=self.device)
            else:
                ci = torch.zeros(1, 0, dtype=torch.int64, device=self.device)
                ch = torch.zeros(1, 0, dtype=torch.int64, device=self.device)
                cpad = torch.zeros(1, 0, dtype=torch.bool, device=self.device)

            desc_repr = self.net.encode(desc, ci, ch, cpad)  # (1, d_gene)

        cand_ids = torch.tensor(
            [[self.vocab.to_idx(str(c)) for c in candidates]],
            dtype=torch.int64, device=self.device,
        )
        scores = self.net.score_ids(desc_repr, cand_ids)[0].float().cpu().numpy()

        score_map = {c: float(scores[i]) for i, c in enumerate(candidates)}
        self._meta = {"n_context": n_context, "n_candidates": len(candidates),
                      "ignore_context": self.ignore_context}
        return ModelPrediction(
            scores=score_map,
            metadata={"name": self.name(), "checkpoint": self._ckpt, **self._meta},
        )


__all__ = ["AmortizedRankerModel"]
