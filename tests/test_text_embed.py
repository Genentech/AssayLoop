"""Contracts for the text-embedding layer.

The three properties worth locking down, because breaking any of them silently
changes published numbers rather than raising:

1. The cache is keyed on the *model*, so the same text embeds to the same key
   whichever endpoint served it.
2. A credential-free machine can serve every shipped screen description.
3. A cache miss on a credential-free machine *raises*. It never degrades to a
   different model, which is what the old ``auto`` backend did.
"""

from __future__ import annotations

import numpy as np
import pytest

from assayloop.amortized import text_embed as te

CREDENTIAL_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture
def no_credentials(monkeypatch):
    """A machine with no embedding credentials.

    Deletes rather than blanks: ``assayloop.config`` loads a ``.env`` at import
    time, so a developer's real key is already in ``os.environ`` by the time
    these tests run.
    """
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def shipped_cache():
    path = te._shipped_cache_path()
    if path is None:
        pytest.skip("shipped screen-description embeddings not present")
    return path


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------


def test_key_is_model_keyed_not_backend_keyed():
    """The released key must match the one the research runs wrote.

    The research cache was written by an Azure backend whose ``name`` was
    ``azure:text-embedding-3-small``. Keying on ``model`` rather than ``name``
    is what lets those vectors ship as-is -- see
    ``scripts/build_text_embedding_cache.py``.
    """
    embedder = te.OpenAITextEmbedder()
    assert embedder.model == te.EMBED_MODEL
    assert embedder.name == f"openai:{te.EMBED_MODEL}"
    text = "Phenotype: proliferation\nCell line: K562"
    assert te._text_key(text, model=embedder.model) != te._text_key(
        text, model=embedder.name
    )


def test_key_depends_on_text_and_model():
    a = te._text_key("x", model="m1")
    assert a != te._text_key("y", model="m1")
    assert a != te._text_key("x", model="m2")


def test_key_has_no_delimiter_collision():
    """The NUL separator must keep (model, text) splits unambiguous."""
    assert te._text_key("b", model="a") != te._text_key("", model="a\x00b")


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_auto_never_resolves_to_local(no_credentials):
    """The old ``auto`` fell back to 384-d MiniLM when the API was unreachable."""
    embedder = te.get_text_embedder("auto")
    assert isinstance(embedder, te.OpenAITextEmbedder)
    assert embedder.dim == te.EMBED_DIM


def test_auto_prefers_openai_when_keyed(no_credentials, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert te.get_text_embedder("auto").name.startswith("openai:")


@pytest.mark.parametrize("legacy", ["azure", "none"])
def test_legacy_checkpoint_backends_still_load(no_credentials, legacy):
    """Checkpoints recorded "azure" (and "none") before the release.

    Both meant text-embedding-3-small, and the cache keys never named the
    endpoint, so they resolve to the public client with identical vectors.
    """
    embedder = te.get_text_embedder(legacy)
    assert isinstance(embedder, te.OpenAITextEmbedder)
    assert embedder.model == te.EMBED_MODEL


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown text backend"):
        te.get_text_embedder("bogus")


def test_api_embedders_construct_without_credentials(no_credentials):
    """Clients are lazy, so a cache hit needs no key."""
    assert te.OpenAITextEmbedder().dim == te.EMBED_DIM


# ---------------------------------------------------------------------------
# Misses are loud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["auto", "openai", "azure", "none"])
def test_miss_without_credentials_raises_naming_the_variable(
    no_credentials, tmp_path, backend
):
    cache = te.EmbeddingCache(tmp_path / "cache.npz", shipped=False)
    embedder = te.get_text_embedder(backend)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        te.embed_texts(["a screen that was never run"], embedder, cache=cache, save=False)


# ---------------------------------------------------------------------------
# The two-layer cache
# ---------------------------------------------------------------------------


def test_shipped_layer_is_loaded(shipped_cache):
    cache = te.EmbeddingCache(shipped_cache.parent / "does-not-exist.npz")
    assert len(cache._shipped) > 1000
    assert not cache._store
    assert len(cache) == len(cache._shipped)


def test_shipped_vectors_are_the_right_shape(shipped_cache):
    with np.load(shipped_cache, allow_pickle=False) as z:
        for key in list(z.files)[:50]:
            vec = z[key]
            assert vec.shape == (te.EMBED_DIM,)
            assert vec.dtype == np.float32
            assert np.isfinite(vec).all()
            assert np.linalg.norm(vec) > 0


def test_user_layer_shadows_shipped_layer(shipped_cache, tmp_path):
    cache = te.EmbeddingCache(tmp_path / "cache.npz")
    key = next(iter(cache._shipped))
    original = cache.get(key).copy()
    override = np.full(te.EMBED_DIM, 0.5, dtype=np.float32)
    cache.put(key, override)
    assert np.array_equal(cache.get(key), override)
    assert np.array_equal(cache._shipped[key], original)  # shipped layer intact


def test_save_writes_only_the_user_layer(shipped_cache, tmp_path):
    """The shipped vectors must stay byte-identical to what the paper used."""
    path = tmp_path / "cache.npz"
    cache = te.EmbeddingCache(path)
    cache.put("deadbeef", np.ones(te.EMBED_DIM, dtype=np.float32))
    cache.save()
    with np.load(path, allow_pickle=False) as z:
        assert list(z.files) == ["deadbeef"]
    # ...but a fresh cache still reads through to the shipped layer.
    reloaded = te.EmbeddingCache(path)
    assert reloaded.get("deadbeef") is not None
    assert len(reloaded) == len(cache._shipped) + 1


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    cache = te.EmbeddingCache(tmp_path / "cache.npz", shipped=False)
    cache.put("k", np.zeros(4, dtype=np.float32))
    cache.save()
    assert [p.name for p in tmp_path.iterdir()] == ["cache.npz"]


def test_unreadable_cache_degrades_to_empty(tmp_path, caplog):
    """A torn cache is a performance problem, not a correctness one."""
    path = tmp_path / "cache.npz"
    path.write_bytes(b"not an npz")
    cache = te.EmbeddingCache(path, shipped=False)
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# embed_texts / embed_screens against a stub backend
# ---------------------------------------------------------------------------


class _StubEmbedder(te.TextEmbedder):
    """Counts API calls so we can assert the cache actually prevents them."""

    def __init__(self):
        self.model = te.EMBED_MODEL
        self.name = f"stub:{te.EMBED_MODEL}"
        self.dim = 4
        self.calls: list[list[str]] = []

    def _embed_uncached(self, texts):
        self.calls.append(list(texts))
        return np.stack([np.full(4, float(len(t)), dtype=np.float32) for t in texts])


class _Screen:
    def __init__(self, name, **ctx):
        self.dataset_name = name
        self._ctx = ctx

    def context(self):
        return self._ctx


def test_embed_texts_deduplicates(tmp_path):
    cache = te.EmbeddingCache(tmp_path / "c.npz", shipped=False)
    stub = _StubEmbedder()
    out = te.embed_texts(["a", "b", "a"], stub, cache=cache, save=False)
    assert out.shape == (3, 4)
    assert stub.calls == [["a", "b"]]  # "a" embedded once
    assert np.array_equal(out[0], out[2])


def test_embed_texts_second_call_hits_cache(tmp_path):
    cache = te.EmbeddingCache(tmp_path / "c.npz", shipped=False)
    stub = _StubEmbedder()
    first = te.embed_texts(["a", "b"], stub, cache=cache, save=False)
    second = te.embed_texts(["b", "a"], stub, cache=cache, save=False)
    assert len(stub.calls) == 1
    assert np.array_equal(second, first[::-1])


def test_embed_screens_keys_by_dataset_name(tmp_path):
    cache = te.EmbeddingCache(tmp_path / "c.npz", shipped=False)
    screens = [_Screen("s1", phenotype="growth"), _Screen("s2", phenotype="death")]
    out = te.embed_screens(screens, _StubEmbedder(), cache=cache, save=False)
    assert set(out) == {"s1", "s2"}


# ---------------------------------------------------------------------------
# Canonical text
# ---------------------------------------------------------------------------


def test_screen_text_field_order_is_fixed():
    """The cache is keyed on this string; field order must not drift."""
    text = te.screen_description_text(
        _Screen("s", cell_line="K562", phenotype="proliferation", organism="human")
    )
    assert text == "Phenotype: proliferation\nCell line: K562\nOrganism: human"


def test_screen_text_skips_empty_and_not_specified():
    text = te.screen_description_text(
        _Screen("s", phenotype="growth", cell_line="", cell_type="not specified",
                organism="  ", condition_clause="Not Specified")
    )
    assert text == "Phenotype: growth"


def test_screen_text_falls_back_to_question_then_name():
    assert te.screen_description_text(_Screen("s", question="rank genes")) == "rank genes"
    assert te.screen_description_text(_Screen("s", dataset_name="s")) == "s"


def test_render_context_text_puts_hits_first():
    out = te.render_context_text("desc", ["A", "B", "C"], [0, 1, 0])
    assert out.splitlines() == [
        "desc",
        "Observed hit genes (1): B",
        "Observed non-hit genes (2): A, C",
    ]


def test_render_context_text_handles_empty_sides():
    out = te.render_context_text("d", [], [])
    assert "Observed hit genes (0): none" in out
    assert "Observed non-hit genes (0): none" in out
