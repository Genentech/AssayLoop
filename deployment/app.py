"""Public AssayFormer inference service for the website's Try It page.

This service intentionally has one inference path: the released s19
AssayFormer checkpoint. It contains no BPMF model, LLM client, handoff logic,
internal screen, or internal annotation source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

from assaybench.core.types import Observation
from assayloop.models.amortized_ranker import AmortizedRankerModel

log = logging.getLogger("assayloop.demo")

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
GENE_UNIVERSE_PATH = Path(__file__).resolve().parent / "data" / "f2_genes.json"
ANNOTATIONS_PATH = DOCS_DIR / "assets" / "data" / "gene_umap.json"

# This Hub artifact is the renamed public release of the paper's s19 run.
MODEL_REPO = os.environ.get("ASSAYFORMER_MODEL_REPO", "Genentech/assayformer")
MODEL_REVISION = os.environ.get(
    "ASSAYFORMER_MODEL_REVISION", "474c4e338a9e85ae22b71ca8612d00fd7a216c5c"
)
MODEL_FILE = "model_last.pt"
MODEL_SHA256 = "c5a58d77c43916cd1261410cc0085fc45877dd7888db09aed9022002738d39f4"
MAX_CONTEXT = 1024
DEFAULT_PROBABILITY_SAMPLES = 4000


class ScreenContext(BaseModel):
    phenotype: str = Field("", max_length=500)
    cleaned_phenotype: str = Field("", max_length=200)
    cell_line: str = Field("", max_length=200)
    cell_type: str = Field("", max_length=200)
    organism: str = Field("Homo sapiens", max_length=100)
    library_methodology: str = Field("", max_length=100)
    condition_clause: str = Field("", max_length=500)
    contrast_label: str = Field("", max_length=200)
    description: str = Field("", max_length=1500)


class ObservedGene(BaseModel):
    gene: str = Field(min_length=1, max_length=80)
    hit: bool


class RankRequest(BaseModel):
    screen: ScreenContext
    observations: Annotated[list[ObservedGene], Field(max_length=MAX_CONTEXT)] = Field(
        default_factory=list
    )
    batch_size: int = Field(100, ge=1, le=500)
    exclude_common_essential: bool = False
    example_id: str = Field("", max_length=80)


class _Runtime:
    def __init__(self, checkpoint: Path):
        _validate_release_checkpoint(checkpoint)
        self.model = AmortizedRankerModel(
            checkpoint=checkpoint,
            ckpt_file=MODEL_FILE,
            device=os.environ.get("ASSAYFORMER_DEVICE", "auto"),
        )
        vocab_genes = [g for g in self.model.vocab.itos if g != "<unk>"]
        # HGNC symbols are mostly uppercase but a few (for example C10orf105)
        # are not. Match user input case-insensitively while always passing the
        # checkpoint's exact vocabulary spelling to the model.
        self.gene_by_upper = {g.upper(): g for g in vocab_genes}
        universe = json.loads(GENE_UNIVERSE_PATH.read_text(encoding="utf-8"))
        self.genes = [self.gene_by_upper[g.upper()] for g in universe]
        if len(self.genes) != 21147 or len(set(self.genes)) != len(self.genes):
            raise RuntimeError("The released f2 candidate universe is invalid.")
        self.annotations = _load_annotations()
        self.lock = threading.Lock()


def _validate_release_checkpoint(checkpoint: Path) -> None:
    """Refuse a mislabeled or internal model before any weights are loaded."""
    required = {"config.json", "vocab.json", MODEL_FILE}
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise RuntimeError(f"AssayFormer release is missing: {', '.join(missing)}")

    digest = hashlib.sha256()
    with (checkpoint / MODEL_FILE).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError("Checkpoint weights do not match the released paper s19 artifact.")

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    rl = cfg.get("rl") or {}
    arch = cfg.get("arch") or {}
    checks = {
        "rl.seed": rl.get("seed") == 19,
        "rl.train_screen_set": rl.get("train_screen_set") == "public_train",
        "arch.vocab_size": arch.get("vocab_size") == 24217,
        "arch.text_dim": arch.get("text_dim") == 1536,
        "text_model": cfg.get("text_model") == "text-embedding-3-small",
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "Checkpoint is not the released paper s19 AssayFormer (failed "
            + ", ".join(failed)
            + ")."
        )


def _checkpoint_dir() -> Path:
    configured = os.environ.get("ASSAYFORMER_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(
        os.environ.get(
            "ASSAYFORMER_CACHE_DIR", str(Path(__file__).resolve().parent / ".model-cache")
        )
    )
    return Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir=cache,
            allow_patterns=["config.json", "vocab.json", MODEL_FILE],
            token=os.environ.get("HF_TOKEN"),
        )
    )


@lru_cache(maxsize=1)
def _runtime() -> _Runtime:
    return _Runtime(_checkpoint_dir())


@lru_cache(maxsize=1)
def _load_annotations() -> dict[str, dict]:
    if not ANNOTATIONS_PATH.is_file():
        return {}
    payload = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    records = payload.get("genes", []) if isinstance(payload, dict) else payload
    return {str(row["gene"]).upper(): row for row in records}


def _examples() -> list[dict]:
    examples = []
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # Explicit public-data allowlist. Adding an example requires naming its
        # BioGRID/AssayBench screen here; an arbitrary file cannot leak through.
        if record.get("screen_id") not in {
            "U_1733_merged",
            "U_1736_dec",
            "1953",
            "2462",
            "2466",
        }:
            continue
        examples.append(record)
    return sorted(examples, key=lambda record: record.get("order", 999))


def _batch_inclusion_probabilities(
    scores: list[float],
    batch_size: int,
    *,
    device: torch.device,
    n_samples: int = DEFAULT_PROBABILITY_SAMPLES,
    temperature: float = 1.0,
) -> list[float]:
    """Monte-Carlo Plackett-Luce marginals via the paper's Gumbel-top-k policy."""
    values = torch.as_tensor(scores, dtype=torch.float32, device=device)
    n_candidates = values.numel()
    k = min(batch_size, n_candidates)
    counts = torch.zeros(n_candidates, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    # Keep the temporary Gumbel matrix near four million floats (~16 MB).
    chunk_size = max(1, min(n_samples, 4_000_000 // max(n_candidates, 1)))
    scaled = values / max(temperature, 1e-6)
    completed = 0
    while completed < n_samples:
        chunk = min(chunk_size, n_samples - completed)
        uniform = torch.rand(
            (chunk, n_candidates), generator=generator, device=device
        ).clamp_min(1e-12)
        gumbel = -torch.log(-torch.log(uniform))
        selected = torch.topk(scaled.unsqueeze(0) + gumbel, k, dim=1).indices
        counts.scatter_add_(
            0,
            selected.reshape(-1),
            torch.ones(chunk * k, dtype=torch.float32, device=device),
        )
        completed += chunk
    return (counts / n_samples).cpu().tolist()


def _probability_sample_count() -> int:
    try:
        configured = int(
            os.environ.get(
                "ASSAYFORMER_PROBABILITY_SAMPLES", str(DEFAULT_PROBABILITY_SAMPLES)
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "ASSAYFORMER_PROBABILITY_SAMPLES must be an integer."
        ) from exc
    return min(20_000, max(500, configured))


def _matching_example(req: RankRequest, ctx: dict) -> dict | None:
    """Return public retrospective labels only when metadata is unmodified."""
    if not req.example_id:
        return None
    for example in _examples():
        if example.get("screen_id") != req.example_id:
            continue
        example_ctx = example.get("screen") or {}
        if all(
            str(ctx.get(key, "")).strip() == str(value).strip()
            for key, value in example_ctx.items()
        ):
            return example
    return None


def _rank(req: RankRequest) -> dict:
    rt = _runtime()
    ctx = req.screen.model_dump()
    if not any(str(v).strip() for v in ctx.values()):
        raise ValueError("Describe the screen before ranking genes.")

    seen: set[str] = set()
    known_observations: list[Observation] = []
    unknown: list[str] = []
    for row in req.observations:
        supplied = row.gene.strip()
        key = supplied.upper()
        gene = rt.gene_by_upper.get(key)
        if not supplied or gene in seen:
            continue
        if gene is None:
            unknown.append(key)
            continue
        seen.add(gene)
        known_observations.append(Observation(candidate=gene, label={"hit": row.hit}))

    candidates = [gene for gene in rt.genes if gene not in seen]
    if req.exclude_common_essential:
        candidates = [
            gene
            for gene in candidates
            if not bool((rt.annotations.get(gene.upper()) or {}).get("essential"))
        ]
    if not candidates:
        raise ValueError("No candidate genes remain after exclusions.")

    with rt.lock:
        prediction = rt.model.predict(known_observations, candidates, ctx)
        score_values = [prediction.scores[gene] for gene in candidates]
        n_probability_samples = _probability_sample_count()
        inclusion_probabilities = _batch_inclusion_probabilities(
            score_values,
            req.batch_size,
            device=rt.model.device,
            n_samples=n_probability_samples,
        )
    inclusion_by_gene = dict(zip(candidates, inclusion_probabilities, strict=True))
    matched_example = _matching_example(req, ctx)
    next_batch_labels = (matched_example or {}).get("next_batch_labels") or {}
    ranked = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
    rows = []
    for rank, (gene, score) in enumerate(ranked[: req.batch_size], start=1):
        ann = rt.annotations.get(gene.upper()) or {}
        rows.append(
            {
                "rank": rank,
                "gene": gene,
                "score": score,
                "batch_inclusion_probability": inclusion_by_gene[gene],
                "retrospective_hit": next_batch_labels.get(gene),
                "public_screens": ann.get("screens"),
                "public_hits": ann.get("hits"),
                "common_essential": bool(ann.get("essential", False)),
                "pathway": ann.get("pathway") or "",
                "complex": ann.get("complex") or "",
            }
        )
    return {
        "model": "AssayFormer (released paper checkpoint, s19)",
        "score_label": "Acquisition score",
        "probability_label": "Probability of inclusion in the next batch",
        "probability_method": "Monte Carlo Gumbel-top-k (Plackett-Luce)",
        "probability_samples": n_probability_samples,
        "probability_temperature": 1.0,
        "n_context": len(known_observations),
        "n_candidates": len(candidates),
        "ignored_unknown_genes": unknown,
        "example_id": matched_example.get("screen_id") if matched_example else None,
        "results": rows,
    }


app = FastAPI(
    title="AssayFormer Try It API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

origins = [
    item.strip()
    for item in os.environ.get(
        "ASSAYLOOP_ALLOWED_ORIGINS",
        "https://genentech.github.io,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict:
    # Deliberately do not expose local paths, keys, or endpoint configuration.
    return {
        "ok": True,
        "ready": _runtime.cache_info().currsize > 0,
        "model": "Genentech/assayformer",
        "checkpoint": "model_last.pt (paper s19)",
        "custom_screens_enabled": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/api/examples")
def examples() -> dict:
    return {"examples": _examples()}


@app.post("/api/rank")
async def rank(req: RankRequest) -> dict:
    try:
        return await run_in_threadpool(_rank, req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError:
        log.exception("AssayFormer inference failed")
        raise HTTPException(
            status_code=503,
            detail="AssayFormer is unavailable. Check the service configuration.",
        ) from None


# The same process can serve the complete website. API routes must be mounted
# first so the static site cannot shadow them.
if DOCS_DIR.is_dir():
    app.mount("/", StaticFiles(directory=DOCS_DIR, html=True), name="site")
