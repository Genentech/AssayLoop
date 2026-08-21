"""RankerNet: amortized context-conditioned gene ranker.

Three interchangeable encoders produce a single screen/context representation
that is projected into gene-embedding space and scored by dot-product against a
**tied** learned gene-embedding table (+ per-gene bias). The scoring head is
shared, so AL eval / analyze work identically regardless of encoder.

Encoders (``RankerConfig.encoder_type``):

- ``transformer`` (default): a small ``nn.TransformerEncoder`` over the token
  set ``[DESC, obs_1..obs_Lc]`` where DESC = projected text embedding and each
  obs = projected ``concat(gene_emb, hit_emb)``. No positional encoding.
- ``modernbert_embed``: the same embedding tokens, but the encoder stack is a
  pretrained ModernBERT consuming ``inputs_embeds`` (projected to its hidden
  size). Read the position-0 (DESC) hidden state.
- ``modernbert_text``: ModernBERT over **tokenized text** (screen description +
  observed hits/non-hits), pooled (CLS or mean). The gene-id input side is
  unused, but the gene-embedding table is still learned through the scoring head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

ENCODER_TYPES = ("transformer", "modernbert_embed", "modernbert_text")


@dataclass
class RankerConfig:
    vocab_size: int
    text_dim: int = 1536
    d_model: int = 256
    d_gene: int = 256
    d_hit: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    # ModernBERT variants.
    encoder_type: str = "transformer"
    bert_model: str = "answerdotai/ModernBERT-base"
    freeze_bert: bool = False
    bert_pool: str = "cls"          # cls | mean (text variant only)
    text_max_tokens: int = 2048
    # Ablation: when False the screen description is removed from the model
    # input. Embedding encoders replace the desc token with a learned constant;
    # the text encoder renders an empty description (observed genes only).
    use_description: bool = True
    # Bilinear amortized-inference head: when True the gene-embedding table and
    # per-gene bias are initialised from a BPMF teacher and *frozen*, so the
    # encoder output ``û`` must approximate the BPMF screen latent (score =
    # ``û · V_g``). Wired up by the trainer via ``set_gene_factors``.
    freeze_factors: bool = False
    # Teacher-free bilinear bottleneck: when True the per-gene bias is zeroed and
    # frozen, removing the static-prior shortcut so the ranking must come from
    # ``û · V_g`` (gene factors ``V`` stay *trainable*, unlike ``freeze_factors``).
    disable_gene_bias: bool = False
    # Variational amortized inference (BPMF objective): when True the encoder
    # additionally produces a per-dimension log-variance for the screen latent
    # û, so training can sample û ~ N(µ, σ_q²) (reparam) and add a KL to the
    # Gaussian prior. Inference still uses the posterior mean µ (== the plain
    # ``encode``), so the AL loop / RL stage are unchanged.
    variational: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _load_bert(model_name: str):
    """Load a pretrained encoder (ModernBERT or any BERT-like) for embeddings or
    token inputs. Uses SDPA attention and disables the compile path for
    CPU/multi-thread stability."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(
            model_name, attn_implementation="sdpa", reference_compile=False,
        )
    except TypeError:
        # Non-ModernBERT models don't accept ``reference_compile``.
        return AutoModel.from_pretrained(model_name, attn_implementation="sdpa")


class RankerNet(nn.Module):
    def __init__(self, cfg: RankerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.encoder_type not in ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type {cfg.encoder_type!r}; pick {ENCODER_TYPES}.")

        # Shared scoring head: learned gene embeddings (id 0 = <unk>/pad at zero)
        # + per-gene bias. Used by every encoder variant.
        self.gene_emb = nn.Embedding(cfg.vocab_size, cfg.d_gene, padding_idx=0)
        self.gene_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        # Observed-gene hit embedding (used by the embedding-token encoders).
        self.hit_emb = nn.Embedding(2, cfg.d_hit)

        if cfg.encoder_type == "transformer":
            self._init_transformer(cfg)
        elif cfg.encoder_type == "modernbert_embed":
            self._init_bert_embed(cfg)
        else:  # modernbert_text
            self._init_bert_text(cfg)

        # Variational head: a second projection from the encoder's pooled hidden
        # to a per-dimension log-variance for û (parallel to ``out_proj``, which
        # is the posterior mean µ). Only built when requested.
        if cfg.variational:
            self.out_logvar = nn.Linear(self.out_proj.in_features, cfg.d_gene)

        self._reset_parameters()

        # Bilinear anchor head: when reconstructing from a config (e.g. RL
        # warm-start / inference) keep the BPMF-anchored factors frozen so only
        # the encoder's screen-latent inference is trainable. The actual factor
        # values come from the loaded state dict (or set_gene_factors at train).
        if cfg.freeze_factors:
            self.gene_emb.weight.requires_grad_(False)
            self.gene_bias.requires_grad_(False)

        # Teacher-free bilinear: drop the static per-gene bias so the only path
        # to the ranking is û · V_g (V stays trainable).
        if cfg.disable_gene_bias:
            with torch.no_grad():
                self.gene_bias.zero_()
            self.gene_bias.requires_grad_(False)

    # -- per-encoder construction -----------------------------------------

    def _init_transformer(self, cfg: RankerConfig) -> None:
        self.desc_proj = nn.Linear(cfg.text_dim, cfg.d_model)
        self.ctx_proj = nn.Linear(cfg.d_gene + cfg.d_hit, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)  # 0 = desc, 1 = obs
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.num_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_gene)

    def _init_bert_embed(self, cfg: RankerConfig) -> None:
        self.bert = _load_bert(cfg.bert_model)
        h = self.bert.config.hidden_size
        self.desc_proj = nn.Linear(cfg.text_dim, h)
        self.ctx_proj = nn.Linear(cfg.d_gene + cfg.d_hit, h)
        self.type_emb = nn.Embedding(2, h)
        self.out_proj = nn.Linear(h, cfg.d_gene)
        if cfg.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def _init_bert_text(self, cfg: RankerConfig) -> None:
        self.bert = _load_bert(cfg.bert_model)
        h = self.bert.config.hidden_size
        self.out_proj = nn.Linear(h, cfg.d_gene)
        if cfg.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        with torch.no_grad():
            self.gene_emb.weight[0].zero_()
        nn.init.normal_(self.hit_emb.weight, std=0.02)
        if hasattr(self, "type_emb"):
            nn.init.normal_(self.type_emb.weight, std=0.02)
        # Start the variational head at log-variance 0 (σ_q ≈ 1) for stability.
        if hasattr(self, "out_logvar"):
            nn.init.zeros_(self.out_logvar.weight)
            nn.init.zeros_(self.out_logvar.bias)

    def set_gene_factors(
        self, V: torch.Tensor, bias: torch.Tensor, *, freeze: bool = True
    ) -> None:
        """Load (and optionally freeze) the scoring head's gene factors.

        ``V`` is ``(vocab_size, d_gene)`` and ``bias`` is ``(vocab_size,)``,
        built by :mod:`assayloop.amortized.gene_factors` (paper §4.1.1). With
        ``freeze=True`` the gene factors / bias stop receiving gradients, so the
        only trainable path to fit the (context-varying) targets is the
        encoder's ``û`` -- i.e. the model is forced to infer the screen latent
        from the revealed context instead of leaning on a static prior.
        """
        V = torch.as_tensor(V, dtype=self.gene_emb.weight.dtype)
        bias = torch.as_tensor(bias, dtype=self.gene_bias.dtype)
        if V.shape != self.gene_emb.weight.shape:
            raise ValueError(
                f"gene factor shape {tuple(V.shape)} != "
                f"{tuple(self.gene_emb.weight.shape)}; set d_gene to the BPMF K."
            )
        with torch.no_grad():
            self.gene_emb.weight.copy_(V)
            self.gene_emb.weight[0].zero_()  # keep padding id at 0
            self.gene_bias.copy_(bias)
        if freeze:
            self.gene_emb.weight.requires_grad_(False)
            self.gene_bias.requires_grad_(False)

    @property
    def is_text(self) -> bool:
        return self.cfg.encoder_type == "modernbert_text"

    # -- encoding: embedding-token variants (transformer / modernbert_embed)

    def _encode_hidden(
        self,
        desc_emb: torch.Tensor,   # (B, text_dim)
        ctx_idx: torch.Tensor,    # (B, Lc) long
        ctx_hit: torch.Tensor,    # (B, Lc) long
        ctx_pad: torch.Tensor,    # (B, Lc) bool (True = padding)
    ) -> torch.Tensor:
        """Pooled DESC hidden (pre ``out_proj``): (B, H). Shared by ``encode``
        (mean head) and ``encode_var`` (mean + log-variance heads)."""
        if self.is_text:
            raise RuntimeError("encode() is not valid for the text encoder; use encode_text().")
        B = desc_emb.shape[0]
        Lc = ctx_idx.shape[1] if ctx_idx.dim() == 2 else 0

        if self.cfg.use_description:
            desc_tok = self.desc_proj(desc_emb) + self.type_emb.weight[0]
        else:
            # Ablation: a learned constant query token, independent of any
            # description embedding passed in.
            desc_tok = self.type_emb.weight[0].unsqueeze(0).expand(B, -1)
        desc_tok = desc_tok.unsqueeze(1)  # (B, 1, H)

        if Lc > 0:
            g = self.gene_emb(ctx_idx)
            h = self.hit_emb(ctx_hit.clamp(0, 1))
            obs_tok = self.ctx_proj(torch.cat([g, h], dim=-1)) + self.type_emb.weight[1]
            tokens = torch.cat([desc_tok, obs_tok], dim=1)  # (B, 1+Lc, H)
            desc_pad = torch.zeros(B, 1, dtype=torch.bool, device=desc_emb.device)
            key_pad = torch.cat([desc_pad, ctx_pad], dim=1)
        else:
            tokens = desc_tok
            key_pad = torch.zeros(B, 1, dtype=torch.bool, device=desc_emb.device)

        if self.cfg.encoder_type == "transformer":
            enc = self.encoder(tokens, src_key_padding_mask=key_pad)
            desc_out = self.norm(enc[:, 0, :])
        else:  # modernbert_embed: attention_mask is 1 for valid tokens
            attn = (~key_pad).long()
            out = self.bert(inputs_embeds=tokens, attention_mask=attn)
            desc_out = out.last_hidden_state[:, 0, :]
        return desc_out

    def encode(
        self,
        desc_emb: torch.Tensor,
        ctx_idx: torch.Tensor,
        ctx_hit: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> torch.Tensor:
        """DESC representation projected into gene space: (B, d_gene). Valid for
        the ``transformer`` and ``modernbert_embed`` encoders."""
        return self.out_proj(self._encode_hidden(desc_emb, ctx_idx, ctx_hit, ctx_pad))

    def encode_var(
        self,
        desc_emb: torch.Tensor,
        ctx_idx: torch.Tensor,
        ctx_hit: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (µ, logσ²) over the screen latent û: each (B, d_gene).
        Requires ``cfg.variational``."""
        if not hasattr(self, "out_logvar"):
            raise RuntimeError("encode_var() requires RankerConfig.variational=True.")
        h = self._encode_hidden(desc_emb, ctx_idx, ctx_hit, ctx_pad)
        return self.out_proj(h), self.out_logvar(h)

    # -- encoding: text variant -------------------------------------------

    def _encode_text_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled text hidden (pre ``out_proj``): (B, H)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state                       # (B, T, H)
        if self.cfg.bert_pool == "mean":
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:  # cls
            pooled = hs[:, 0, :]
        return pooled

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled text representation projected into gene space: (B, d_gene)."""
        return self.out_proj(self._encode_text_hidden(input_ids, attention_mask))

    def encode_text_var(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (µ, logσ²) over û from text: each (B, d_gene)."""
        if not hasattr(self, "out_logvar"):
            raise RuntimeError("encode_text_var() requires RankerConfig.variational=True.")
        h = self._encode_text_hidden(input_ids, attention_mask)
        return self.out_proj(h), self.out_logvar(h)

    # -- scoring -----------------------------------------------------------

    def score_ids(self, desc_repr: torch.Tensor, gene_ids: torch.Tensor) -> torch.Tensor:
        emb = self.gene_emb(gene_ids)
        scores = torch.einsum("bd,bld->bl", desc_repr, emb)
        scores = scores + self.gene_bias[gene_ids]
        return scores

    def score_all(self, desc_repr: torch.Tensor) -> torch.Tensor:
        scores = desc_repr @ self.gene_emb.weight.t()
        scores = scores + self.gene_bias.unsqueeze(0)
        return scores

    def forward(
        self,
        desc_emb: torch.Tensor,
        ctx_idx: torch.Tensor,
        ctx_hit: torch.Tensor,
        ctx_pad: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        desc_repr = self.encode(desc_emb, ctx_idx, ctx_hit, ctx_pad)
        return self.score_ids(desc_repr, tgt_idx)


__all__ = ["RankerConfig", "RankerNet", "ENCODER_TYPES"]
