#!/usr/bin/env python3
"""Build the STRING / CORUM / SIGNOR interaction tables from public downloads.

The network-recovery analysis (paper Figure 9) and the CORUM-labelled UMAPs
originally read parquet tables produced by an internal data platform, which
cannot be redistributed. This rebuilds tables with the same layout from the
public releases, so a clean checkout can run those scripts.

Run ``scripts/fetch_ground_truth.sh`` first; it downloads the raw files and
then calls this. To convert already-downloaded files:

    python3 scripts/convert_ground_truth.py --raw <dir>/_raw --out <dir>

Nothing here degrades quietly. A missing source, an unexpected column, or an
empty edge set raises: a half-built ground truth turns into a low recall
number that reads as a result.

REPRODUCTION FIDELITY

The internal tables were built from STRING v11.5 (2021-08-12), CORUM 4.1
(2022-11-28) and SIGNOR 3.0 -- all three CC BY 4.0 at those versions. This
script pins those releases, and its output was compared against the internal
tables directly:

  * STRING interactions reproduce exactly -- all 593,222 protein pairs match,
    as does the score >= 400 subset. 42 of 593,222 scores differ by one unit
    from rounding.
  * CORUM and SIGNOR reproduce exactly: same complexes, same entities, same
    edge sets.

Two harmless differences remain, both reported in the summary:

  * Protein dimension. The internal STRING table listed only the 17,848
    proteins touched by an edge; this one lists all 19,566 in the release.
    The extra 1,718 are isolated and contribute no interactions.
  * Gene symbols. The internal mapping supplied a symbol for 17,565 of its
    17,848 proteins; public STRING supplies one for all of them. Gene-pair
    counts therefore come out slightly *higher* here (+2.9% at score >= 400):
    the paper's counts were deflated by those missing symbols, not by a
    narrower edge set. Where both have a symbol they agree 97.0% of the time;
    the rest are HGNC renames postdating v11.5 (KIAA2022 -> NEXMIF,
    C12orf5 -> TIGAR). No public alias source scored better -- BioMart_HUGO
    and Ensembl_HGNC both reach 97.1%.

Expect network-recovery numbers very close to the paper's, differing only
through that wider symbol coverage.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

# STRING combines its two experimental channels ("experiments" and
# "experiments_transferred") with a noisy-OR after removing a uniform prior of
# 0.041, then adds the prior back. The internal table kept pairs whose combined
# score was >= 182 on that scale; that rule reproduces its 593,222 rows and its
# minimum score exactly, so it is used verbatim here.
STRING_PRIOR = 0.041
STRING_MIN_COMBINED = 182

# Counts produced by the internal tables the paper used. Printed next to what
# this script builds so the drift is explicit.
PAPER_REFERENCE = {
    "STRING proteins": 17848,
    "STRING gene symbols": 17565,
    "STRING interaction rows": 593222,
    "STRING gene pairs (score >= 400)": 100348,
    "STRING gene pairs (all)": 574628,
    "CORUM complexes": 3691,
    "CORUM gene pairs": 46090,
    "SIGNOR protein entities": 7718,
    "SIGNOR gene pairs": 16839,
}

RAW_FILES = {
    "string_links": "9606.protein.links.full.v11.5.txt.gz",
    "string_info": "9606.protein.info.v11.5.txt.gz",
    "string_aliases": "9606.protein.aliases.v11.5.txt.gz",
    "corum": "allComplexes.txt",
    "signor": "signor_human_all.tsv",
}

HOWTO = {
    "corum": (
        "CORUM 4.1 'All complexes' table. Download allComplexes.txt (or the\n"
        "  .zip and unpack it) from https://mips.helmholtz-muenchen.de/corum/\n"
        "  -> Download, and place it at the path above."
    ),
    "signor": (
        "SIGNOR 3.0 human interactions, tab-separated. Download 'All human\n"
        "  data' from https://signor.uniroma2.it/downloads.php (or\n"
        "  https://signor.uniroma2.it/getData.php?organism=9606) and save it\n"
        "  at the path above."
    ),
}


class MissingSource(FileNotFoundError):
    """A raw download this script cannot proceed without."""


def _need(path: Path, key: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        extra = HOWTO.get(key)
        raise MissingSource(
            f"Missing raw input: {path}\n  "
            + (extra or "Run scripts/fetch_ground_truth.sh to download it.")
        )
    return path


def _require_columns(have, want, what: str) -> None:
    missing = [c for c in want if c not in have]
    if missing:
        raise ValueError(
            f"{what}: expected column(s) {missing}, got {sorted(have)[:14]}... "
            "The upstream format changed; fix the converter rather than "
            "dropping the source."
        )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------

def _string_combined(experiments: int, transferred: int) -> int:
    """STRING's prior-corrected noisy-OR of the two experimental channels."""
    p = STRING_PRIOR
    a = max((experiments / 1000.0 - p) / (1 - p), 0.0)
    b = max((transferred / 1000.0 - p) / (1 - p), 0.0)
    return round(1000 * ((1 - (1 - a) * (1 - b)) * (1 - p) + p))


def convert_string(raw: Path, out: Path) -> dict:
    import pandas as pd

    info_path = _need(raw / RAW_FILES["string_info"], "string_info")
    alias_path = _need(raw / RAW_FILES["string_aliases"], "string_aliases")
    links_path = _need(raw / RAW_FILES["string_links"], "string_links")

    # Protein dimension, in file order, so ids are stable across runs.
    ensps: list[str] = []
    symbols: list[str] = []
    with _open_text(info_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        _require_columns(header, ["#string_protein_id", "preferred_name"], info_path.name)
        i_id, i_name = header.index("#string_protein_id"), header.index("preferred_name")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_id, i_name):
                continue
            ensps.append(parts[i_id])
            symbols.append(parts[i_name])
    if not ensps:
        raise ValueError(f"{info_path} yielded no proteins.")
    index = {e: i for i, e in enumerate(ensps)}

    # UniProt accessions. Curated mappings win over BLAST-inferred ones.
    curated: dict[str, str] = {}
    inferred: dict[str, str] = {}
    with _open_text(alias_path) as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, alias, source = parts[0], parts[1], parts[2]
            if pid not in index:
                continue
            if "UniProt_AC" not in source:
                continue
            target = inferred if source.startswith("BLAST_") else curated
            target.setdefault(pid, alias)
    uniprot = {**inferred, **curated}

    # Interactions. One row per undirected pair, matching the internal table.
    pairs: dict[tuple[int, int], int] = {}
    with _open_text(links_path) as fh:
        header = fh.readline().split()
        _require_columns(
            header, ["protein1", "protein2", "experiments", "experiments_transferred"],
            links_path.name,
        )
        i_p1, i_p2 = header.index("protein1"), header.index("protein2")
        i_e, i_t = header.index("experiments"), header.index("experiments_transferred")
        width = len(header)
        for line in fh:
            parts = line.split()
            if len(parts) != width:
                continue
            e, t = int(parts[i_e]), int(parts[i_t])
            if e == 0 and t == 0:
                continue
            score = _string_combined(e, t)
            if score < STRING_MIN_COMBINED:
                continue
            a, b = index.get(parts[i_p1]), index.get(parts[i_p2])
            if a is None or b is None or a == b:
                continue
            pairs[(min(a, b), max(a, b))] = score
    if not pairs:
        raise ValueError(
            "STRING produced no interactions. Check that the links file is the "
            "'full' variant (it must contain an 'experiments' column)."
        )

    dest = out / "STRING-HUMAN"
    dest.mkdir(parents=True, exist_ok=True)
    ids = pd.Series(range(len(ensps)), dtype="int16")
    pd.DataFrame({"string_human_id": ids,
                  "string_protein": pd.Series(ensps, dtype="category")}
                 ).to_parquet(dest / "fact_StringProtein.parquet")
    pd.DataFrame({"string_human_id": ids,
                  "gene_symbol": pd.Series(symbols, dtype="str")}
                 ).to_parquet(dest / "dimension_GeneSymbol.parquet")
    pd.DataFrame({"string_human_id": ids,
                  "uniprot_id": pd.Series([uniprot.get(e) for e in ensps], dtype="str")}
                 ).to_parquet(dest / "dimension_UniprotId.parquet")
    items = sorted(pairs.items())
    pd.DataFrame({
        "string_human_id_1": pd.Series([a for (a, _), _ in items], dtype="int16"),
        "string_human_id_2": pd.Series([b for (_, b), _ in items], dtype="int16"),
        "experiment_score": pd.Series([s for _, s in items], dtype="int64"),
    }).to_parquet(dest / "dimension_Interactions.parquet")

    sym_of = {i: s for i, s in enumerate(symbols) if s}
    gene_pairs = {(min(x, y), max(x, y))
                  for (a, b) in pairs
                  for x, y in [(sym_of.get(a), sym_of.get(b))]
                  if x and y and x != y}
    strong = {(min(x, y), max(x, y))
              for (a, b), s in pairs.items() if s >= 400
              for x, y in [(sym_of.get(a), sym_of.get(b))]
              if x and y and x != y}
    return {
        "STRING proteins": len(ensps),
        "STRING gene symbols": len(sym_of),
        "STRING interaction rows": len(pairs),
        "STRING gene pairs (score >= 400)": len(strong),
        "STRING gene pairs (all)": len(gene_pairs),
    }


# ---------------------------------------------------------------------------
# CORUM
# ---------------------------------------------------------------------------

CORUM_COLS = {
    "complex_id": ["ComplexID"],
    "complex_name": ["ComplexName"],
    "organism": ["Organism"],
    "comment": ["Complex comment", "ComplexComment"],
    "uniprot": ["subunits(UniProt IDs)"],
    "entrez": ["subunits(Entrez IDs)"],
    "protein_name": ["subunits(Protein name)"],
    "gene_name": ["subunits(Gene name)"],
    "synonyms": ["subunits(Gene name syn)"],
}


def _corum_pick(header: list[str], names: list[str]) -> int | None:
    for n in names:
        if n in header:
            return header.index(n)
    return None


def _split_subunits(cell: str) -> list[str]:
    return [p.strip() for p in (cell or "").split(";")]


def convert_corum(raw: Path, out: Path) -> dict:
    import pandas as pd

    path = raw / RAW_FILES["corum"]
    if not path.is_file():
        zipped = path.with_suffix(".txt.zip")
        if zipped.is_file():
            with zipfile.ZipFile(zipped) as z:
                name = next(n for n in z.namelist() if n.endswith(".txt"))
                path.write_bytes(z.read(name))
    _need(path, "corum")

    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {k: _corum_pick(header, v) for k, v in CORUM_COLS.items()}
        for key in ("complex_id", "complex_name", "organism", "gene_name"):
            if idx[key] is None:
                _require_columns(header, CORUM_COLS[key], path.name)
        rows = [line.rstrip("\n").split("\t") for line in fh]

    def cell(parts, key):
        i = idx[key]
        return parts[i] if i is not None and i < len(parts) else ""

    complexes, subunits = [], []
    for parts in rows:
        if not parts or not cell(parts, "complex_id"):
            continue
        if "human" not in cell(parts, "organism").lower():
            continue
        try:
            cid = int(cell(parts, "complex_id"))
        except ValueError:
            continue
        complexes.append((cid, cell(parts, "complex_name"), cell(parts, "comment") or None))
        genes = _split_subunits(cell(parts, "gene_name"))
        ups = _split_subunits(cell(parts, "uniprot"))
        ents = _split_subunits(cell(parts, "entrez"))
        pns = _split_subunits(cell(parts, "protein_name"))
        syns = _split_subunits(cell(parts, "synonyms"))
        for i, gene in enumerate(genes):
            if not gene or gene.lower() in ("none", "null"):
                continue
            try:
                entrez = float(ents[i]) if i < len(ents) and ents[i].isdigit() else None
            except (ValueError, IndexError):
                entrez = None
            subunits.append({
                "complex_id": cid,
                "uniprot_id": ups[i] if i < len(ups) else None,
                "protein_name": pns[i] if i < len(pns) else None,
                "swissprot_organism": "Homo sapiens (Human)",
                "entrez_gene_id": entrez,
                "gene_name": gene,
                "gene_name_synonyms": syns[i] if i < len(syns) else None,
                "ensembl_gene_id": None,
                "subunit_comment": None,
            })
    if not complexes or not subunits:
        raise ValueError(
            f"{path} yielded no human complexes. Check that this is CORUM's "
            "'All complexes' table and that its Organism column says 'Human'."
        )

    dest = out / "CORUM-HUMAN"
    dest.mkdir(parents=True, exist_ok=True)
    cdf = pd.DataFrame(complexes, columns=["complex_id", "complex_name", "complex_comment"])
    cdf = cdf.drop_duplicates("complex_id").set_index("complex_id")
    cdf.to_parquet(dest / "fact_Complex.parquet")
    sdf = pd.DataFrame(subunits)
    sdf.index = pd.RangeIndex(1, len(sdf) + 1, name="subunit_id")
    sdf.to_parquet(dest / "dimension_Subunit.parquet")

    members = defaultdict(set)
    for s in subunits:
        members[s["complex_id"]].add(s["gene_name"].strip().upper())
    pairs = set()
    for group in members.values():
        g = sorted(group)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.add((g[i], g[j]))
    return {"CORUM complexes": len(cdf), "CORUM gene pairs": len(pairs)}


# ---------------------------------------------------------------------------
# SIGNOR
# ---------------------------------------------------------------------------

def convert_signor(raw: Path, out: Path) -> dict:
    import pandas as pd

    path = _need(raw / RAW_FILES["signor"], "signor")
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        _require_columns(
            header, ["ENTITYA", "TYPEA", "IDA", "ENTITYB", "TYPEB", "IDB",
                     "EFFECT", "MECHANISM"], path.name,
        )
        col = {c: i for i, c in enumerate(header)}
        rows = [line.rstrip("\n").split("\t") for line in fh]

    # Entities are keyed by (name, id) so a symbol reused across databases does
    # not silently collapse two distinct entities into one.
    keys: dict[tuple[str, str], int] = {}
    entities: list[dict] = []
    facts: list[dict] = []

    def key_for(name, ent_id, etype, database):
        k = (name, ent_id)
        if k not in keys:
            keys[k] = len(entities)
            entities.append({"ENTITY_KEY": len(entities), "ENTITY_ID": ent_id,
                             "ENTITY_NAME": name, "TYPE": etype, "DATABASE": database})
        return keys[k]

    def get(parts, name):
        i = col.get(name)
        return parts[i].strip() if i is not None and i < len(parts) else ""

    for parts in rows:
        if not parts or not get(parts, "ENTITYA") or not get(parts, "ENTITYB"):
            continue
        a = key_for(get(parts, "ENTITYA"), get(parts, "IDA"), get(parts, "TYPEA"),
                    get(parts, "DATABASEA"))
        b = key_for(get(parts, "ENTITYB"), get(parts, "IDB"), get(parts, "TYPEB"),
                    get(parts, "DATABASEB"))
        facts.append({"SIGNOR_KEY": len(facts), "ENTITYA_KEY": a, "ENTITYB_KEY": b,
                      "EFFECT": get(parts, "EFFECT"), "MECHANISM": get(parts, "MECHANISM")})
    if not facts:
        raise ValueError(
            f"{path} yielded no interactions. Check that it is the tab-separated "
            "'all data' export rather than a filtered pathway download."
        )
    if len(entities) > 65535 or len(facts) > 65535:
        raise ValueError(
            "SIGNOR is larger than the uint16 key space the consumers expect "
            f"({len(entities)} entities, {len(facts)} interactions). Widen the "
            "dtype here and in the readers together."
        )

    dest = out / "SIGNOR-HUMAN"
    dest.mkdir(parents=True, exist_ok=True)
    edf = pd.DataFrame(entities)
    edf["ENTITY_KEY"] = edf["ENTITY_KEY"].astype("uint16")
    for c in ("TYPE", "DATABASE"):
        edf[c] = edf[c].astype("category")
    edf.to_parquet(dest / "Dimension_Entities.parquet", index=False)
    fdf = pd.DataFrame(facts)
    for c in ("SIGNOR_KEY", "ENTITYA_KEY", "ENTITYB_KEY"):
        fdf[c] = fdf[c].astype("uint16")
    for c in ("EFFECT", "MECHANISM"):
        fdf[c] = fdf[c].astype("category")
    fdf.to_parquet(dest / "Fact_Protein_Interactions.parquet", index=False)

    prot = {e["ENTITY_KEY"]: e["ENTITY_NAME"].upper()
            for e in entities if e["TYPE"] == "protein" and e["ENTITY_NAME"]}
    pairs = {(min(x, y), max(x, y))
             for f in facts
             for x, y in [(prot.get(f["ENTITYA_KEY"]), prot.get(f["ENTITYB_KEY"]))]
             if x and y and x != y}
    return {"SIGNOR protein entities": len(prot), "SIGNOR gene pairs": len(pairs)}


# ---------------------------------------------------------------------------

CONVERTERS = {"string": convert_string, "corum": convert_corum, "signor": convert_signor}


def summarise(stats: dict) -> None:
    print("\n" + "=" * 68)
    print(f"{'':44s}{'built':>10s}{'paper':>10s}")
    print("-" * 68)
    for k, v in stats.items():
        ref = PAPER_REFERENCE.get(k)
        if ref:
            delta = 100.0 * (v - ref) / ref
            print(f"{k:44s}{v:10d}{ref:10d}   {delta:+6.1f}%")
        else:
            print(f"{k:44s}{v:10d}{'-':>10s}")
    print("=" * 68)
    print("The 'paper' column is what the internal tables contained. Differences\n"
          "come from the wider public protein universe and from HGNC symbol\n"
          "drift; see this script's docstring.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="directory of downloaded sources")
    ap.add_argument("--out", type=Path, required=True,
                    help="directory to write STRING-HUMAN/, CORUM-HUMAN/, SIGNOR-HUMAN/")
    ap.add_argument("--sources", default="string,corum,signor",
                    help="comma-separated subset to build")
    args = ap.parse_args()

    wanted = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in CONVERTERS]
    if unknown:
        ap.error(f"unknown source(s) {unknown}; choose from {sorted(CONVERTERS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    failed: list[str] = []
    for name in wanted:
        print(f"[{name}] converting...")
        try:
            stats.update(CONVERTERS[name](args.raw, args.out))
        except MissingSource as exc:
            failed.append(f"{name}: {exc}")
            print(f"[{name}] MISSING -- {exc}", file=sys.stderr)
    if stats:
        summarise(stats)
    if failed:
        print("\nNot built:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        print("\nThe scripts that read these will raise until every source is "
              "present. They do not fall back to a partial ground truth.",
              file=sys.stderr)
        return 1
    print(f"\nWrote {args.out}. Export it:\n  "
          f"export ASSAYLOOP_GROUND_TRUTH={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
