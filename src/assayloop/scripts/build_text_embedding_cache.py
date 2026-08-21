"""Build (or verify) the shipped screen-description embedding cache.

The released package ships ``text-embedding-3-small`` vectors for every screen
in every public screen set, so reproducing the paper needs no API key. This
script is how that file is produced.

Two modes:

**Re-key** (``--from-cache``) -- take the research cache, which was written by
the Azure backend under keys ``sha1("azure:text-embedding-3-small" + text)``,
and rewrite it under the released keys ``sha1("text-embedding-3-small" + text)``.
No API calls, and the vectors are bit-identical to the ones the paper's numbers
were computed from. This is how the shipped file was made.

**Embed** (default) -- embed every public screen description through whichever
backend ``--backend`` resolves to. Needs ``OPENAI_API_KEY``. Use this to rebuild
from scratch, or to extend the cache to screen sets of your own.

Either way the script only ever writes texts it can reconstruct from the public
screen sets, so internal screens cannot leak into the released artifact.

``--verify N`` checks the premise the re-key mode rests on -- that the public
OpenAI API and the Azure deployment return the same vectors for the same model.
It re-embeds N cached texts through the public API and reports the cosine
similarity distribution. Needs ``OPENAI_API_KEY``.

Usage::

    # what produced the shipped file
    uv run python -m assayloop.scripts.build_text_embedding_cache \
        --from-cache output/rankers/text_emb_cache/cache.npz

    # rebuild from scratch through the public API
    OPENAI_API_KEY=sk-... uv run python -m assayloop.scripts.build_text_embedding_cache

    # confirm Azure and public OpenAI agree before trusting the re-key
    OPENAI_API_KEY=sk-... uv run python -m assayloop.scripts.build_text_embedding_cache \
        --from-cache output/rankers/text_emb_cache/cache.npz --verify 25 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from assayloop.amortized import text_embed as te
from assayloop.tasks import load_screens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_text_embedding_cache")

# Every screen set a public user can ask for. The union of their descriptions is
# exactly what we are allowed to ship.
PUBLIC_SCREEN_SETS = (
    "public",
    "public_validation",
    "public_train",
    "public_val",
)

# The key prefix the research cache was written under, before keys moved from
# the backend-qualified name to the bare model id.
LEGACY_KEY_NAME = f"azure:{te.EMBED_MODEL}"

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "data" / "text_embeddings" / "screen_descriptions.npz"
)


def _legacy_key(text: str) -> str:
    return hashlib.sha1(f"{LEGACY_KEY_NAME}\x00{text}".encode("utf-8")).hexdigest()


def collect_public_texts(screen_sets=PUBLIC_SCREEN_SETS) -> dict[str, list[str]]:
    """``{screen_set: [description text, ...]}`` for the public screen sets."""
    out: dict[str, list[str]] = {}
    for name in screen_sets:
        screens = load_screens(target_set=name)
        out[name] = [te.screen_description_text(s) for s in screens]
        log.info("%s: %d screens", name, len(screens))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_provenance(out: Path, *, texts: int, source: str, per_set: dict[str, int]) -> None:
    lines = [
        "# Shipped screen-description embeddings",
        "",
        f"`{out.name}` -- {texts} embeddings of `{te.EMBED_MODEL}`, {te.EMBED_DIM}-d float32,",
        "keyed by `sha1(model + \"\\x00\" + text)` (see",
        "`assayloop.amortized.text_embed._text_key`).",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{te.EMBED_MODEL}` |",
        f"| Dimensionality | {te.EMBED_DIM} |",
        f"| Entries | {texts} |",
        f"| Bytes | {out.stat().st_size:,} |",
        f"| sha256 | `{_sha256(out)}` |",
        f"| Built by | `python -m assayloop.scripts.build_text_embedding_cache` ({source}) |",
        "",
        "## Coverage",
        "",
        "One entry per distinct screen description across the public screen sets:",
        "",
        "| Screen set | Screens |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in per_set.items()]
    lines += [
        "",
        "Distinct texts total fewer than the sum of the rows because the smaller",
        "evaluation sets are subsets of the training fold.",
        "",
        "## Why this ships",
        "",
        "ASSAYFORMER conditions on a 1536-d embedding of the screen description.",
        "Shipping the vectors means reproducing every number in the paper needs no",
        "API key and costs nothing. Screens outside these sets are not cached; the",
        "embedder raises and names the credential to set rather than substituting a",
        "different model.",
        "",
        "## Provenance of the vectors",
        "",
        "The research runs called `text-embedding-3-small` through an Azure OpenAI",
        "deployment. These are those exact vectors, re-keyed from the",
        "backend-qualified cache key to the bare model id -- not re-embedded -- so",
        "they are bit-identical to what the published results used. The public",
        "OpenAI API serves the same model. To confirm the two endpoints agree",
        "before relying on that, with your own key:",
        "",
        "```",
        "OPENAI_API_KEY=sk-... python -m assayloop.scripts.build_text_embedding_cache \\",
        "    --from-cache <research cache>.npz --verify 25 --dry-run",
        "```",
    ]
    (out.parent / "PROVENANCE.md").write_text("\n".join(lines) + "\n")


def verify_against_public_api(store: dict[str, np.ndarray], texts: list[str], n: int) -> None:
    """Re-embed ``n`` cached texts via the public OpenAI API and compare."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--verify needs OPENAI_API_KEY set.")
    rng = np.random.default_rng(0)
    have = [t for t in texts if te._text_key(t, model=te.EMBED_MODEL) in store]
    pick = [have[i] for i in rng.choice(len(have), size=min(n, len(have)), replace=False)]
    fresh = te.OpenAITextEmbedder()._embed_uncached(pick)
    cos = []
    for t, v in zip(pick, fresh):
        c = store[te._text_key(t, model=te.EMBED_MODEL)]
        cos.append(
            float(np.dot(c, v) / (np.linalg.norm(c) * np.linalg.norm(v)))
        )
    cos_arr = np.asarray(cos)
    log.info(
        "verify: n=%d  cosine min=%.6f  median=%.6f  max=%.6f",
        len(cos_arr), cos_arr.min(), float(np.median(cos_arr)), cos_arr.max(),
    )
    if cos_arr.min() < 0.999:
        log.error(
            "Public OpenAI and the cached vectors DISAGREE (min cosine %.6f). "
            "The shipped cache cannot stand in for the public API.", cos_arr.min()
        )
    else:
        log.info("Public OpenAI matches the cached vectors.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-cache", type=Path, default=None,
        help="Re-key an existing research cache instead of calling an API.",
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output .npz (default: {DEFAULT_OUT}).",
    )
    ap.add_argument(
        "--backend", default="auto",
        help="Embedder backend when not re-keying: auto|openai.",
    )
    ap.add_argument(
        "--verify", type=int, default=0, metavar="N",
        help="Re-embed N texts through the public OpenAI API and compare cosines.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report coverage; write nothing.")
    args = ap.parse_args()

    by_set = collect_public_texts()
    per_set = {k: len(v) for k, v in by_set.items()}
    texts = sorted({t for v in by_set.values() for t in v})
    log.info("%d distinct screen descriptions across %d sets", len(texts), len(by_set))

    store: dict[str, np.ndarray] = {}
    if args.from_cache is not None:
        with np.load(args.from_cache, allow_pickle=False) as z:
            legacy = {k: z[k] for k in z.files}
        log.info("research cache %s: %d entries", args.from_cache, len(legacy))
        missing = []
        for t in texts:
            v = legacy.get(_legacy_key(t))
            if v is None:
                missing.append(t)
            else:
                store[te._text_key(t, model=te.EMBED_MODEL)] = np.asarray(v, dtype=np.float32)
        source = f"re-keyed from {args.from_cache.name}"
        if missing:
            # Not a warning to shrug at: a public screen we cannot serve from the
            # cache is a public screen that needs an API key at eval time.
            raise SystemExit(
                f"{len(missing)} of {len(texts)} public screen descriptions are absent "
                f"from {args.from_cache}. Re-keying would ship an incomplete cache. "
                f"First missing text:\n{missing[0][:300]}"
            )
    else:
        embedder = te.get_text_embedder(args.backend)
        log.info("embedding %d texts via %s", len(texts), embedder.name)
        vecs = embedder._embed_uncached(texts)
        store = {
            te._text_key(t, model=te.EMBED_MODEL): np.asarray(v, dtype=np.float32)
            for t, v in zip(texts, vecs)
        }
        source = f"embedded via {embedder.name}"

    dims = {tuple(v.shape) for v in store.values()}
    if dims != {(te.EMBED_DIM,)}:
        raise SystemExit(f"Unexpected vector shapes in cache: {sorted(dims)}")
    log.info("collected %d/%d vectors", len(store), len(texts))

    if args.verify:
        verify_against_public_api(store, texts, args.verify)

    if args.dry_run:
        log.info("--dry-run: not writing %s", args.out)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.npz")
    np.savez(tmp, **store)
    os.replace(tmp, args.out)
    _write_provenance(args.out, texts=len(store), source=source, per_set=per_set)
    log.info("wrote %s (%.1f MB) + PROVENANCE.md", args.out, args.out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
