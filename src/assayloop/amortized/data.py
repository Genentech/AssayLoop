"""Data plumbing for the amortized ranker.

- ``GeneVocab``: maps gene symbol -> integer id (id 0 is a shared ``<unk>``),
  built from the union of all genes across the supplied screen lists so that
  eval-only genes still get a (cold, untrained) slot.
- ``ScreenExample``: a precomputed per-screen bundle (description embedding,
  in-vocab gene ids, hit flags, unmasked relevance-score targets).
- ``RankerDataset``: yields, per access, a randomly sampled observed-gene
  *context* plus the supervised *target* genes/scores (on-the-fly so contexts
  vary across epochs).
- ``collate``: pads ragged context/target sequences into batched tensors.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger("assayloop.amortized.data")

UNK = "<unk>"


# ---------------------------------------------------------------------------
# Gene vocabulary
# ---------------------------------------------------------------------------


class GeneVocab:
    """Gene symbol <-> id. Id 0 is reserved for ``<unk>`` (OOV / padding)."""

    def __init__(self, genes: Sequence[str]):
        # ``genes`` should NOT include the unk token; we prepend it.
        self.itos: list[str] = [UNK] + list(genes)
        self.stoi: dict[str, int] = {g: i for i, g in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def to_idx(self, gene: str) -> int:
        i = self.stoi.get(gene)
        if i is None:
            i = self.stoi.get(str(gene).upper(), 0)
        return i

    @classmethod
    def build(cls, screen_lists: Sequence[Sequence[Any]]) -> "GeneVocab":
        """Build from the union of ``screen.genes`` across several lists.

        Sorted for determinism so the same data always yields the same ids
        (important for reusing a saved checkpoint's embedding table).
        """
        seen: set[str] = set()
        for screens in screen_lists:
            for s in screens:
                for g in s.genes:
                    seen.add(g)
        return cls(sorted(seen))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeneVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        # Drop the leading UNK; __init__ re-adds it.
        v = cls(itos[1:] if itos and itos[0] == UNK else itos)
        return v

    def coverage(self, screens: Sequence[Any]) -> dict[str, float]:
        """Fraction of (screen, gene) entries present in the vocab (non-unk)."""
        total = 0
        known = 0
        for s in screens:
            for g in s.genes:
                total += 1
                if self.to_idx(g) != 0:
                    known += 1
        return {"known": known, "total": total,
                "frac": (known / total) if total else 0.0}


# ---------------------------------------------------------------------------
# Precomputed per-screen example
# ---------------------------------------------------------------------------


@dataclass
class ScreenExample:
    name: str
    desc_emb: np.ndarray          # (D_text,)
    gene_idx: np.ndarray          # (G,) int64 vocab ids of in-screen genes
    hit: np.ndarray               # (G,) int64 {0,1}
    score: np.ndarray             # (G,) float32 unmasked relevance score
    genes: list[str]              # (G,) gene symbols (for the text encoder)
    desc_text: str = ""           # rendered description (for the text encoder)
    n_library: int = 0            # # in-screen-library genes (first n_library entries);
                                  # any beyond are appended universe-extras (hit=0)


def build_examples(
    screens: Sequence[Any],
    desc_embeddings: dict[str, np.ndarray],
    vocab: GeneVocab,
    *,
    require_desc_emb: bool = True,
    universe_genes: list[str] | None = None,
) -> list[ScreenExample]:
    """Bundle screens into ``ScreenExample``s.

    Uses ``unmasked_relevance_scores`` as the target, falling back to
    ``relevance_scores`` if the unmasked version is empty.

    ``require_desc_emb``: the embedding-token encoders need a precomputed
    description embedding (screens without one are skipped). The ``modernbert_text``
    encoder works purely from text, so pass ``require_desc_emb=False`` to keep
    every screen and substitute a zero placeholder embedding.
    """
    from .text_embed import screen_description_text

    out: list[ScreenExample] = []
    for s in screens:
        scores = s.unmasked_relevance_scores or s.relevance_scores
        n = min(len(s.genes), len(scores), len(s.hits))
        if n == 0:
            continue
        genes = [str(g) for g in s.genes[:n]]
        gene_idx = np.fromiter((vocab.to_idx(g) for g in genes), dtype=np.int64, count=n)
        hit = np.fromiter((1 if h else 0 for h in s.hits[:n]), dtype=np.int64, count=n)
        score = np.asarray(scores[:n], dtype=np.float32)
        emb = desc_embeddings.get(s.name if hasattr(s, "name") else s.dataset_name)
        if emb is None:
            emb = desc_embeddings.get(s.dataset_name)
        if emb is None:
            if require_desc_emb:
                log.warning("No description embedding for screen %s; skipping.", s.dataset_name)
                continue
            emb = np.zeros(1, dtype=np.float32)
        if universe_genes is not None:
            screen_set = set(genes)
            extra = [g for g in universe_genes if g not in screen_set]
            if extra:
                extra_idx = np.fromiter((vocab.to_idx(g) for g in extra),
                                        dtype=np.int64, count=len(extra))
                gene_idx = np.concatenate([gene_idx, extra_idx])
                hit = np.concatenate([hit, np.zeros(len(extra), dtype=np.int64)])
                score = np.concatenate([score, np.zeros(len(extra), dtype=np.float32)])
                genes = genes + extra
        out.append(ScreenExample(
            name=s.dataset_name,
            desc_emb=np.asarray(emb, dtype=np.float32),
            gene_idx=gene_idx,
            hit=hit,
            score=score,
            genes=genes,
            desc_text=screen_description_text(s),
            n_library=n,
        ))
    return out


# ---------------------------------------------------------------------------
# Dataset with on-the-fly context sampling
# ---------------------------------------------------------------------------


class RankerDataset(Dataset):
    """Each access samples a random observed-gene context for one screen.

    Args:
        examples: precomputed ScreenExamples (training screens).
        max_context: cap on # observed genes fed as context.
        min_context / context sizing: context size is drawn uniformly in
            ``[0, min(max_context, G-1)]`` each access, so the model sees the
            full cold-start..warm spectrum.
        max_targets: cap on # supervised genes per example (all hits are
            always kept; non-hits sampled to fill). ``None`` = all in-screen.
        hit_context_frac: with this probability bias the context sample to
            include at least some hits (otherwise random subsets are almost
            all non-hits, and the model never sees informative context).
        leave_out: if True (default), supervise only the genes NOT in the
            sampled context (held-out). This makes the observed context
            causally necessary -- the model must use the observed hits to
            predict the unseen genes -- instead of being free to ignore it
            (the target score is otherwise context-independent). Cold-start
            (empty context) still supervises all genes.
        seed: RNG seed.
    """

    def __init__(
        self,
        examples: list[ScreenExample],
        *,
        max_context: int = 1024,
        max_targets: int | None = 4096,
        hit_context_frac: float = 0.5,
        leave_out: bool = True,
        seed: int = 0,
        target_mode: str = "relevance",
        warm_start: Any = None,
        warm_start_prob: float = 0.0,
        warm_start_n: str | int = "random",
    ):
        self.examples = examples
        self.max_context = int(max_context)
        self.max_targets = max_targets
        self.hit_context_frac = float(hit_context_frac)
        self.leave_out = bool(leave_out)
        self._rng = random.Random(seed)
        # GLM warm-start contexts: with prob ``warm_start_prob`` the sampled
        # context is GLM-5.1's first n acquired genes (a realistic handoff
        # context) instead of a random subset, so the model learns to continue
        # from GLM-quality observations. ``warm_start_n`` = "random"
        # (n~U{1..rounds}) or a fixed int.
        self.warm_start_prob = float(warm_start_prob)
        self.warm_start_n = warm_start_n
        self.warm_start = None
        if warm_start is not None and warm_start_prob > 0.0:
            warm_start.index_examples(examples)
            self.warm_start = warm_start
        # ``relevance`` -> static unmasked relevance score (MSE);
        # ``hits`` -> the genes' own binary hit labels (self-supervised
        # in-context hit prediction, soft-target BCE).
        if target_mode not in ("relevance", "hits"):
            raise ValueError(f"Unknown target_mode {target_mode!r}; pick relevance|hits.")
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.examples)

    def _warm_context(self, ex: ScreenExample) -> np.ndarray | None:
        """GLM warm-start context for ``ex`` (first n acquired genes), or None."""
        ws = self.warm_start
        if ws is None or not ws.has(ex.name):
            return None
        rounds = ws.max_rounds(ex.name)
        if rounds <= 0:
            return None
        if isinstance(self.warm_start_n, str) and self.warm_start_n.lower() == "random":
            n = self._rng.randint(1, rounds)
        else:
            n = max(1, min(int(self.warm_start_n), rounds))
        pos = ws.sample_positions(ex.name, n, self._rng)
        if pos is None or pos.size == 0:
            return None
        # Keep <=1 hit held out so leave-out targets have a positive to find.
        g = len(ex.gene_idx)
        if pos.size >= g:
            pos = pos[: g - 1]
        if pos.size > self.max_context:
            pos = np.array(self._rng.sample(list(pos), self.max_context), dtype=np.int64)
        return pos.astype(np.int64)

    def _sample_context(self, ex: ScreenExample) -> np.ndarray:
        """Return indices (into ex arrays) chosen as the observed context."""
        if self.warm_start is not None and self._rng.random() < self.warm_start_prob:
            wc = self._warm_context(ex)
            if wc is not None:
                return wc
        g = len(ex.gene_idx)
        hi = min(self.max_context, max(0, g - 1))
        if hi <= 0:
            return np.empty(0, dtype=np.int64)
        n_ctx = self._rng.randint(0, hi)
        if n_ctx == 0:
            return np.empty(0, dtype=np.int64)
        all_pos = range(g)
        # Optionally guarantee some hits in the context so the model learns
        # to use active-learning feedback (random subsets are mostly misses).
        hit_pos = np.nonzero(ex.hit == 1)[0]
        if (
            len(hit_pos) > 0
            and self._rng.random() < self.hit_context_frac
        ):
            # Under leave-out, always keep >=1 hit held-out so the target set
            # has a positive to "find"; cap context hits at len(hit_pos) - 1.
            hit_cap = len(hit_pos)
            if self.leave_out and len(hit_pos) >= 2:
                hit_cap = len(hit_pos) - 1
            n_hit = self._rng.randint(1, min(hit_cap, max(1, n_ctx)))
            chosen_hits = self._rng.sample(list(hit_pos), n_hit)
            remaining = n_ctx - n_hit
            if remaining > 0:
                pool = [p for p in all_pos if p not in set(chosen_hits)]
                chosen_rest = self._rng.sample(pool, min(remaining, len(pool)))
            else:
                chosen_rest = []
            ctx = np.array(chosen_hits + chosen_rest, dtype=np.int64)
        else:
            ctx = np.array(self._rng.sample(list(all_pos), n_ctx), dtype=np.int64)
        return ctx

    def _sample_targets(self, ex: ScreenExample, ctx_pos: np.ndarray) -> np.ndarray:
        """Target positions, capped via ``max_targets`` (always keeping hits).

        Under ``leave_out`` the observed context positions are excluded so the
        model is supervised only on the held-out genes (predict the unseen from
        the observed). Cold-start (empty context) supervises all genes.
        """
        g = len(ex.gene_idx)
        if self.leave_out and len(ctx_pos):
            ctx_set = {int(p) for p in ctx_pos}
            cand = np.array([p for p in range(g) if p not in ctx_set], dtype=np.int64)
        else:
            cand = np.arange(g)
        if cand.size == 0:
            # Degenerate (all genes observed); fall back to all positions.
            cand = np.arange(g)
        if self.max_targets is None or cand.size <= self.max_targets:
            return cand
        is_hit = ex.hit[cand] == 1
        hit_pos = cand[is_hit]
        non_hit = cand[~is_hit]
        budget = max(0, self.max_targets - len(hit_pos))
        sampled = (
            np.array(self._rng.sample(list(non_hit), min(budget, len(non_hit))),
                     dtype=np.int64)
            if budget > 0 else np.empty(0, dtype=np.int64)
        )
        return np.concatenate([hit_pos, sampled])

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = self.examples[i]
        ctx_pos = self._sample_context(ex)
        tgt_pos = self._sample_targets(ex, ctx_pos)
        ctx_hit_arr = ex.hit[ctx_pos] if len(ctx_pos) else np.empty(0, dtype=np.int64)
        tgt_ids = ex.gene_idx[tgt_pos]
        if self.target_mode == "hits":
            tgt_score = ex.hit[tgt_pos].astype(np.float32)
        else:
            tgt_score = ex.score[tgt_pos]
        return {
            "desc_emb": torch.from_numpy(ex.desc_emb),
            "ctx_idx": torch.from_numpy(ex.gene_idx[ctx_pos]) if len(ctx_pos)
            else torch.empty(0, dtype=torch.int64),
            "ctx_hit": torch.from_numpy(ctx_hit_arr) if len(ctx_pos)
            else torch.empty(0, dtype=torch.int64),
            "tgt_idx": torch.from_numpy(tgt_ids),
            "tgt_score": torch.from_numpy(tgt_score),
            # Text-encoder inputs (ignored by the embedding collate).
            "desc_text": ex.desc_text,
            "ctx_sym": [ex.genes[p] for p in ctx_pos],
            "ctx_hit_list": [int(x) for x in ctx_hit_arr],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad ragged context/target sequences. Returns a dict of tensors.

    - ``desc_emb``  : (B, D_text)
    - ``ctx_idx``   : (B, Lc) long   (0 padded)
    - ``ctx_hit``   : (B, Lc) long   (0 padded)
    - ``ctx_pad``   : (B, Lc) bool   True where padding (key_padding_mask)
    - ``tgt_idx``   : (B, Lt) long   (0 padded)
    - ``tgt_score`` : (B, Lt) float
    - ``tgt_mask``  : (B, Lt) bool   True where valid
    """
    B = len(batch)
    desc = torch.stack([b["desc_emb"] for b in batch], dim=0)
    Lc = max((b["ctx_idx"].numel() for b in batch), default=0)
    Lt = max((b["tgt_idx"].numel() for b in batch), default=1)
    Lc = max(Lc, 0)
    Lt = max(Lt, 1)

    ctx_idx = torch.zeros(B, Lc, dtype=torch.int64)
    ctx_hit = torch.zeros(B, Lc, dtype=torch.int64)
    ctx_pad = torch.ones(B, Lc, dtype=torch.bool)  # True = padding
    tgt_idx = torch.zeros(B, Lt, dtype=torch.int64)
    tgt_score = torch.zeros(B, Lt, dtype=torch.float32)
    tgt_mask = torch.zeros(B, Lt, dtype=torch.bool)

    for i, b in enumerate(batch):
        c = b["ctx_idx"].numel()
        if c:
            ctx_idx[i, :c] = b["ctx_idx"]
            ctx_hit[i, :c] = b["ctx_hit"]
            ctx_pad[i, :c] = False
        t = b["tgt_idx"].numel()
        tgt_idx[i, :t] = b["tgt_idx"]
        tgt_score[i, :t] = b["tgt_score"]
        tgt_mask[i, :t] = True

    return {
        "desc_emb": desc,
        "ctx_idx": ctx_idx,
        "ctx_hit": ctx_hit,
        "ctx_pad": ctx_pad,
        "tgt_idx": tgt_idx,
        "tgt_score": tgt_score,
        "tgt_mask": tgt_mask,
    }


def _pad_targets(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad just the supervised target tensors (shared by both collators)."""
    B = len(batch)
    Lt = max((b["tgt_idx"].numel() for b in batch), default=1)
    Lt = max(Lt, 1)
    tgt_idx = torch.zeros(B, Lt, dtype=torch.int64)
    tgt_score = torch.zeros(B, Lt, dtype=torch.float32)
    tgt_mask = torch.zeros(B, Lt, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["tgt_idx"].numel()
        tgt_idx[i, :t] = b["tgt_idx"]
        tgt_score[i, :t] = b["tgt_score"]
        tgt_mask[i, :t] = True
    return {"tgt_idx": tgt_idx, "tgt_score": tgt_score, "tgt_mask": tgt_mask}


class TextCollator:
    """Collate for the ``modernbert_text`` encoder.

    Renders each example's description + observed hits/non-hits to text, tokenizes
    the batch (right-truncated at ``text_max_tokens``), and pads the supervised
    target tensors. Hits are rendered before non-hits so truncation drops the
    least-informative non-hits first.
    """

    def __init__(self, tokenizer: Any, text_max_tokens: int = 2048,
                 use_description: bool = True):
        self.tokenizer = tokenizer
        self.text_max_tokens = int(text_max_tokens)
        self.use_description = bool(use_description)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        from .text_embed import render_context_text

        texts = [
            render_context_text(
                b["desc_text"] if self.use_description else "",
                b["ctx_sym"], b["ctx_hit_list"],
            )
            for b in batch
        ]
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.text_max_tokens, return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        out.update(_pad_targets(batch))
        return out


__all__ = [
    "UNK",
    "GeneVocab",
    "ScreenExample",
    "build_examples",
    "RankerDataset",
    "collate",
    "TextCollator",
]
