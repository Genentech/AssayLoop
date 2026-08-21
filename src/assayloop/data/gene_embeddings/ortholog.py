"""MGI -> HGNC symbol mapping for mouse-organism screens.

Human-only gene-embedding sources (PRESAGE, GenePT) need a human symbol.
Mouse screens arrive with MGI symbols, so we map before lookup.

**This is a naming convention, not an ortholog database.** What it
actually does is uppercase the symbol -- which is correct for the large
majority of mouse genes, whose MGI symbol differs from the human HGNC
symbol only in case (``Brca1`` -> ``BRCA1``) -- plus a hand-written table
for the handful of common genes where the two nomenclatures genuinely
diverge (``Trp53`` -> ``TP53``). It does not consult HomoloGene, BioMart,
or any orthology assignment, and it will happily return a string that is
not a human gene at all.

That is survivable because the failure is caught one layer down: an
unrecognised symbol misses in the embedding source, and the KNN / RF /
MLP models substitute a zero vector with an ``is_unknown`` flag rather
than a wrong neighbour. Mouse screens are a small minority of the corpus
and none are in the paper's 20-screen test set, so this is a convenience
for running the loop on mouse data, not a result-bearing component. A
real cross-species join is the obvious upgrade if that changes.

Note this is a *different* problem from ``assaybench``'s
``utils/gene_mapper.py``, which reconciles aliases and deprecated symbols
*within* one species; the two do not overlap and neither subsumes the
other.

Resolved symbols are cached under ``$PRESAGE_CACHE/../ortholog_cache/``.
The uppercase fallback is deliberately not cached, so that dropping in a
real mapping table later takes effect immediately.
"""

from __future__ import annotations

import json
from typing import Mapping

from ... import config


_CACHE_FILE = config.ORTHOLOG_CACHE_PATH / "mgi_to_hgnc.json"
_BUILTIN: Mapping[str, str] = {
    # Genes where uppercasing the MGI symbol does NOT give the HGNC symbol.
    "Trp53": "TP53",
    "Brca1": "BRCA1",
    "Brca2": "BRCA2",
    "Myc": "MYC",
    "Pten": "PTEN",
    "Cdkn1a": "CDKN1A",
    "Cdkn2a": "CDKN2A",
    "Rb1": "RB1",
    "Ezh2": "EZH2",
    "Apc": "APC",
}


def _load_cache() -> dict[str, str]:
    if _CACHE_FILE.is_file():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(mapping: dict[str, str]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _CACHE_FILE.write_text(json.dumps(mapping, sort_keys=True, indent=0))
    except OSError:
        pass


def map_mgi_to_hgnc(mgi_symbol: str) -> str | None:
    """Best-effort MGI -> HGNC mapping; ``None`` only for an empty symbol.

    Note this never reports failure for a *non-empty* symbol: unmappable
    ones come back uppercased and then miss downstream. See the module
    docstring for why that is the intended contract here.
    """
    if not mgi_symbol:
        return None
    cache = _load_cache()
    if mgi_symbol in cache:
        return cache[mgi_symbol]
    if mgi_symbol.upper() in cache:
        return cache[mgi_symbol.upper()]
    # Try built-ins.
    if mgi_symbol in _BUILTIN:
        cache[mgi_symbol] = _BUILTIN[mgi_symbol]
        _save_cache(cache)
        return _BUILTIN[mgi_symbol]
    if mgi_symbol.upper() in {k.upper(): v for k, v in _BUILTIN.items()}:
        v = next(v for k, v in _BUILTIN.items() if k.upper() == mgi_symbol.upper())
        cache[mgi_symbol] = v
        _save_cache(cache)
        return v
    # Last resort: most MGI symbols are HGNC-compatible once uppercased.
    # Not cached, so a real mapping table can take over later.
    return mgi_symbol.upper()


__all__ = ["map_mgi_to_hgnc"]
