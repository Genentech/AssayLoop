"""Screen-description text embeddings for the amortized ranker.

The ranker conditions on a single text embedding of the screen description.
We use OpenAI ``text-embedding-3-small`` (1536-d) and cache every embedding on
disk keyed by a hash of ``(model, exact text)``, so re-runs and the DAgger
rounds never re-pay the API cost.

Three things are worth knowing before you change anything here.

**The cache is keyed on the model, not the endpoint.** ``text-embedding-3-small``
is the same model whether you reach it through the public OpenAI API or through
an Azure OpenAI deployment. The research runs went through an internal Azure
deployment; that backend is gone from this release, but because the keys name
only the model, the vectors it produced are exactly the ones shipped here — see
``data/text_embeddings/PROVENANCE.md``. Checkpoints that recorded
``text_backend: "azure"`` therefore load unchanged.

**A cache hit needs no credentials.** API clients are constructed lazily, on the
first *miss*. Every screen in every public screen set ships pre-embedded, so
reproducing the paper requires no API key at all. Point the code at a screen we
never embedded and you get a hard error naming the variable to set — never a
silently different embedding.

**There is no silent fallback.** ``LocalTextEmbedder`` (MiniLM, 384-d) lives in a
completely different vector space and will quietly wreck any number computed
against an ASSAYFORMER checkpoint trained on the 1536-d space. It is reachable
only by asking for ``backend="local"`` explicitly; ``"auto"`` will never choose
it, and a missing API key will never demote you into it.

Credentials: ``OPENAI_API_KEY``, plus optional ``OPENAI_BASE_URL`` if you route
through a compatible proxy. Nothing else.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .. import config

log = logging.getLogger("assayloop.amortized.text_embed")

# The embedding model, and the only one the released checkpoints understand.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Back-compat aliases: checkpoint configs and older scripts refer to these.
AZURE_EMBED_MODEL = EMBED_MODEL
AZURE_EMBED_DIM = EMBED_DIM

# A local sentence-transformer, for ablations only. NOT a fallback -- see the
# module docstring. 384-d, incompatible with every released checkpoint.
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pre-computed embeddings for every screen in the public screen sets, shipped
# with the package so that reproducing the paper needs no API key.
SHIPPED_CACHE_RESOURCE = ("assayloop.data", "text_embeddings/screen_descriptions.npz")


# ---------------------------------------------------------------------------
# Canonical screen text
# ---------------------------------------------------------------------------


def screen_description_text(screen) -> str:
    """Build the canonical description string we embed for a screen.

    Public screens often leave ``description`` empty, so we compose a rich,
    stable string from the available metadata fields. The exact text is what
    the on-disk cache is keyed on, so keep this deterministic.
    """
    ctx = screen.context() if hasattr(screen, "context") else dict(screen)
    parts: list[str] = []
    for label, key in [
        ("Phenotype", "phenotype"),
        ("Cleaned phenotype", "cleaned_phenotype"),
        ("Cell line", "cell_line"),
        ("Cell type", "cell_type"),
        ("Organism", "organism"),
        ("Library methodology", "library_methodology"),
        ("Condition", "condition_clause"),
        ("Contrast", "contrast_label"),
        ("Description", "description"),
    ]:
        v = ctx.get(key)
        if v and str(v).strip() and str(v).strip().lower() != "not specified":
            parts.append(f"{label}: {str(v).strip()}")
    if not parts:
        # Last resort: the rendered ranking question, or the screen name.
        q = ctx.get("question") or ctx.get("dataset_name") or ""
        return str(q).strip()
    return "\n".join(parts)


def render_context_text(
    desc_text: str,
    ctx_symbols: Sequence[str],
    ctx_hits: Sequence[int],
) -> str:
    """Render a screen description + observed AL feedback into a single string
    for the ``modernbert_text`` encoder.

    Hits are listed before non-hits so that right-truncation (when the tokenized
    text exceeds ``text_max_tokens``) drops the least-informative non-hits first
    while keeping the description and observed hits intact.
    """
    hits = [str(g) for g, h in zip(ctx_symbols, ctx_hits) if h]
    non = [str(g) for g, h in zip(ctx_symbols, ctx_hits) if not h]
    parts = [desc_text.strip()]
    parts.append(
        f"Observed hit genes ({len(hits)}): " + (", ".join(hits) if hits else "none")
    )
    parts.append(
        f"Observed non-hit genes ({len(non)}): " + (", ".join(non) if non else "none")
    )
    return "\n".join(parts)


def _text_key(text: str, *, model: str) -> str:
    """Cache key for ``text`` under ``model``.

    ``model`` is the bare model id (``text-embedding-3-small``), deliberately
    *not* the backend-qualified ``embedder.name``: the vectors are identical
    whichever endpoint served them, so keying on the endpoint would fork the
    cache into two copies of the same numbers.
    """
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------


class TextEmbedder:
    """Base class: ``_embed_uncached(list[str]) -> np.ndarray (N, dim)``."""

    dim: int = 0
    name: str = "base"
    model: str = ""

    def _embed_uncached(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class OpenAITextEmbedder(TextEmbedder):
    """Public OpenAI embeddings (``text-embedding-3-small`` by default).

    The client is built lazily, on the first cache miss, so an embedder can be
    constructed -- and every cached screen served -- with no API key present.
    """

    def __init__(
        self,
        *,
        model: str = EMBED_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 256,
    ):
        self.model = model
        self.dim = EMBED_DIM
        self.batch_size = int(batch_size)
        self.name = f"openai:{model}"
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Text not found in the embedding cache, and no OpenAI API key is "
                "configured, so it cannot be embedded. Set OPENAI_API_KEY to embed "
                "new screens. (Every screen in the public screen sets ships "
                "pre-embedded; you only need a key for screens of your own.)"
            )
        from openai import OpenAI  # lazy import

        base_url = self._base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
        return self._client

    def _embed_uncached(self, texts: Sequence[str]) -> np.ndarray:
        client = self._ensure_client()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i : i + self.batch_size])
            resp = client.embeddings.create(model=self.model, input=chunk)
            for d in resp.data:
                out.append(np.asarray(d.embedding, dtype=np.float32))
        return np.stack(out, axis=0) if out else np.zeros((0, self.dim), np.float32)


class LocalTextEmbedder(TextEmbedder):
    """Local sentence-transformer, for ablations only.

    384-d, a different vector space from ``text-embedding-3-small``. Feeding it
    to a released ASSAYFORMER checkpoint produces meaningless rankings. Only
    reachable via an explicit ``backend="local"``.
    """

    def __init__(self, *, model: str = LOCAL_EMBED_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = int(batch_size)
        self.model = model
        self.name = f"local:{model}"

    def _embed_uncached(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return np.asarray(vecs, dtype=np.float32)


# Backend strings recorded by checkpoints trained before the release. Both meant
# "text-embedding-3-small", which is what ``openai`` now means, and the cache
# keys never mentioned the endpoint -- so these load with identical vectors.
_LEGACY_BACKENDS = {"azure", "none"}


def get_text_embedder(backend: str = "auto", **kwargs) -> TextEmbedder:
    """Resolve a text embedder.

    ``backend``:

    - ``"auto"`` (default) / ``"openai"`` -- ``text-embedding-3-small`` through
      the public OpenAI API. Constructing it always succeeds; a missing
      ``OPENAI_API_KEY`` surfaces only on a genuine cache miss, so the shipped
      screen descriptions are served without credentials.
    - ``"local"`` -- the 384-d MiniLM ablation. Never selected implicitly,
      because feeding it to a released checkpoint silently produces nonsense.

    ``"azure"`` and ``"none"`` are accepted for checkpoints trained before the
    release and resolve to ``"openai"``.
    """
    backend = (backend or "auto").lower()
    if backend == "local":
        return LocalTextEmbedder(**kwargs)
    if backend in ("auto", "openai") or backend in _LEGACY_BACKENDS:
        return OpenAITextEmbedder(**kwargs)
    raise ValueError(
        f"Unknown text backend {backend!r}; expected one of "
        "'auto', 'openai', 'local'."
    )


# ---------------------------------------------------------------------------
# Cached embedding store
# ---------------------------------------------------------------------------


def _shipped_cache_path() -> Path | None:
    """Locate the packaged screen-description embeddings, or ``None``."""
    package, relative = SHIPPED_CACHE_RESOURCE
    try:
        from importlib.resources import files

        path = Path(str(files(package))) / relative
    except (ImportError, ModuleNotFoundError, TypeError):  # pragma: no cover
        return None
    return path if path.is_file() else None


class EmbeddingCache:
    """``sha1(model, text) -> vector``, over two layers.

    The read-only *shipped* layer holds the screen-description embeddings that
    come with the package. The writable *user* layer is a single ``.npz`` under
    the output directory and holds anything embedded since. Lookups check the
    user layer first; ``save()`` only ever writes the user layer, so the shipped
    vectors stay byte-identical to what the paper used.
    """

    def __init__(self, path: str | Path | None = None, *, shipped: bool = True):
        self.path = Path(path) if path else (
            config.OUTPUT_PATH / "rankers" / "text_emb_cache" / "cache.npz"
        )
        self._shipped: dict[str, np.ndarray] = {}
        self._store: dict[str, np.ndarray] = {}
        if shipped:
            self._shipped = self._read(_shipped_cache_path())
        self._store = self._read(self.path)

    @staticmethod
    def _read(path: Path | None) -> dict[str, np.ndarray]:
        if path is None or not path.is_file():
            return {}
        try:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read embedding cache %s: %s", path, e)
            return {}

    def __len__(self) -> int:
        return len(set(self._store) | set(self._shipped))

    def get(self, key: str) -> np.ndarray | None:
        v = self._store.get(key)
        return self._shipped.get(key) if v is None else v

    def put(self, key: str, vec: np.ndarray) -> None:
        self._store[key] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically to avoid a torn cache on interruption. NOTE: the
        # temp name must end in ``.npz`` or np.savez appends it, breaking the
        # rename target. Use a unique temp name so concurrent savers (parallel
        # eval workers) don't clobber each other's temp file mid-rename.
        uniq = f"{os.getpid()}.{threading.get_ident()}.{int(time.time() * 1e6)}"
        tmp = self.path.with_name(f"{self.path.stem}.{uniq}.tmp.npz")
        try:
            np.savez(tmp, **self._store)
            os.replace(tmp, self.path)
        except OSError as e:  # best-effort cache; never fail a run over it
            log.warning("Could not persist embedding cache %s: %s", self.path, e)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def embed_texts(
    texts: Sequence[str],
    embedder: TextEmbedder,
    *,
    cache: EmbeddingCache | None = None,
    save: bool = True,
) -> np.ndarray:
    """Embed ``texts`` (de-duplicated) with on-disk caching. Returns (N, dim)."""
    cache = cache if cache is not None else EmbeddingCache()
    keys = [_text_key(t, model=embedder.model) for t in texts]
    # Determine which unique texts are missing.
    missing_idx: list[int] = []
    missing_texts: list[str] = []
    seen_missing: set[str] = set()
    for i, (t, k) in enumerate(zip(texts, keys)):
        if cache.get(k) is None and k not in seen_missing:
            seen_missing.add(k)
            missing_idx.append(i)
            missing_texts.append(t)
    if missing_texts:
        log.info("Embedding %d new text(s) via %s (cache has %d).",
                 len(missing_texts), embedder.name, len(cache))
        vecs = embedder._embed_uncached(missing_texts)
        for t, v in zip(missing_texts, vecs):
            cache.put(_text_key(t, model=embedder.model), v)
        if save:
            cache.save()
    return np.stack([cache.get(k) for k in keys], axis=0)


def embed_screens(
    screens: Iterable,
    embedder: TextEmbedder,
    *,
    cache: EmbeddingCache | None = None,
    save: bool = True,
) -> dict[str, np.ndarray]:
    """Return ``{dataset_name: embedding}`` for a list of ScreenRecords."""
    screens = list(screens)
    texts = [screen_description_text(s) for s in screens]
    mat = embed_texts(texts, embedder, cache=cache, save=save)
    return {s.dataset_name: mat[i] for i, s in enumerate(screens)}


__all__ = [
    "EMBED_MODEL",
    "EMBED_DIM",
    "AZURE_EMBED_MODEL",
    "AZURE_EMBED_DIM",
    "screen_description_text",
    "render_context_text",
    "TextEmbedder",
    "OpenAITextEmbedder",
    "LocalTextEmbedder",
    "get_text_embedder",
    "EmbeddingCache",
    "embed_texts",
    "embed_screens",
]
