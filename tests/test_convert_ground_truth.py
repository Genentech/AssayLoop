"""The ground-truth converter must fail loudly, not build a partial table.

The scripts that read STRING / CORUM / SIGNOR raise when a source is missing,
precisely so a thin ground truth never shows up as a low recall number. That
guarantee is only worth anything if the converter upholds the same rule: a
truncated download, a renamed upstream column, or a filtered export must stop
the build rather than write a table that parses.

These run on hand-written fixtures, so they check the parsing contract and the
output schema the readers depend on -- not the real edge counts, which are
verified by running the converter against the actual releases.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "convert_ground_truth",
    Path(__file__).resolve().parent.parent / "scripts" / "convert_ground_truth.py",
)
cgt = importlib.util.module_from_spec(_SPEC)
sys.modules["convert_ground_truth"] = cgt
_SPEC.loader.exec_module(cgt)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

PROTEINS = [
    ("9606.ENSP00000000001", "AAA", "P00001"),
    ("9606.ENSP00000000002", "BBB", "P00002"),
    ("9606.ENSP00000000003", "CCC", "P00003"),
]

LINK_COLS = [
    "protein1", "protein2", "neighborhood", "neighborhood_transferred", "fusion",
    "cooccurence", "homology", "coexpression", "coexpression_transferred",
    "experiments", "experiments_transferred", "database", "database_transferred",
    "textmining", "textmining_transferred", "combined_score",
]


def _link(p1, p2, experiments, transferred=0):
    row = dict.fromkeys(LINK_COLS, 0)
    row.update(protein1=p1, protein2=p2, experiments=experiments,
               experiments_transferred=transferred, combined_score=900)
    return " ".join(str(row[c]) for c in LINK_COLS)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "_raw"
    d.mkdir()

    with gzip.open(d / cgt.RAW_FILES["string_info"], "wt") as fh:
        fh.write("#string_protein_id\tpreferred_name\tprotein_size\tannotation\n")
        for ensp, sym, _ in PROTEINS:
            fh.write(f"{ensp}\t{sym}\t100\tsome annotation\n")

    with gzip.open(d / cgt.RAW_FILES["string_aliases"], "wt") as fh:
        fh.write("#string_protein_id\talias\tsource\n")
        for ensp, _, acc in PROTEINS:
            # A BLAST-inferred hit first, so the curated one must win.
            fh.write(f"{ensp}\tWRONG{acc}\tBLAST_UniProt_AC\n")
            fh.write(f"{ensp}\t{acc}\tUniProt_AC\n")

    a, b, c = (p[0] for p in PROTEINS)
    with gzip.open(d / cgt.RAW_FILES["string_links"], "wt") as fh:
        fh.write(" ".join(LINK_COLS) + "\n")
        for p, q in ((a, b), (b, a)):          # strong, both directions
            fh.write(_link(p, q, 900) + "\n")
        for p, q in ((a, c), (c, a)):          # above threshold, below 400
            fh.write(_link(p, q, 200) + "\n")
        for p, q in ((b, c), (c, b)):          # below threshold, dropped
            fh.write(_link(p, q, 100) + "\n")

    (d / cgt.RAW_FILES["corum"]).write_text(
        "ComplexID\tComplexName\tOrganism\tSynonyms\tCell line\t"
        "subunits(UniProt IDs)\tsubunits(Entrez IDs)\tsubunits(Protein name)\t"
        "subunits(Gene name)\tsubunits(Gene name syn)\tPubMed ID\tFunCat ID\t"
        "Complex comment\n"
        "1\tAlpha complex\tHuman\t\t\tP00001;P00002\t11;22\tPa;Pb\tAAA;BBB\t\t1\t\tnote\n"
        "2\tBeta complex\tHuman\t\t\tP00002;P00003\t22;33\tPb;Pc\tBBB;CCC\t\t2\t\t\n"
        "3\tMouse complex\tMouse\t\t\tQ1;Q2\t44;55\tPx;Py\tXXX;YYY\t\t3\t\t\n",
        encoding="utf-8",
    )

    (d / cgt.RAW_FILES["signor"]).write_text(
        "ENTITYA\tTYPEA\tIDA\tDATABASEA\tENTITYB\tTYPEB\tIDB\tDATABASEB\t"
        "EFFECT\tMECHANISM\n"
        "AAA\tprotein\tP00001\tUNIPROT\tBBB\tprotein\tP00002\tUNIPROT\tup\tbinding\n"
        "BBB\tprotein\tP00002\tUNIPROT\tAAA\tprotein\tP00001\tUNIPROT\tdown\tbinding\n"
        "AAA\tprotein\tP00001\tUNIPROT\tDRUG\tchemical\tC1\tCHEBI\tdown\t\n",
        encoding="utf-8",
    )
    return d


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------

def test_string_tables_match_the_reader_contract(raw, tmp_path):
    out = tmp_path / "gt"
    stats = cgt.convert_string(raw, out)

    d = out / "STRING-HUMAN"
    sym = pd.read_parquet(d / "dimension_GeneSymbol.parquet")
    uni = pd.read_parquet(d / "dimension_UniprotId.parquet")
    edges = pd.read_parquet(d / "dimension_Interactions.parquet")

    assert list(sym.columns) == ["string_human_id", "gene_symbol"]
    assert list(edges.columns) == [
        "string_human_id_1", "string_human_id_2", "experiment_score"]
    assert sym.string_human_id.dtype == "int16"
    assert list(sym.gene_symbol) == ["AAA", "BBB", "CCC"]
    # Curated UniProt accession beats the BLAST-inferred one.
    assert list(uni.uniprot_id) == ["P00001", "P00002", "P00003"]

    # One row per undirected pair, and the sub-threshold pair is gone.
    assert len(edges) == 2
    assert set(zip(edges.string_human_id_1, edges.string_human_id_2)) == {(0, 1), (0, 2)}
    assert stats["STRING gene pairs (score >= 400)"] == 1
    assert stats["STRING gene pairs (all)"] == 2


def test_corum_keeps_human_complexes_only(raw, tmp_path):
    out = tmp_path / "gt"
    stats = cgt.convert_corum(raw, out)

    d = out / "CORUM-HUMAN"
    sub = pd.read_parquet(d / "dimension_Subunit.parquet")
    cx = pd.read_parquet(d / "fact_Complex.parquet")

    assert cx.index.name == "complex_id"
    assert "complex_name" in cx.columns
    assert {"complex_id", "gene_name"} <= set(sub.columns)
    assert list(cx.complex_name) == ["Alpha complex", "Beta complex"]
    assert set(sub.gene_name) == {"AAA", "BBB", "CCC"}
    assert stats["CORUM complexes"] == 2
    # AAA-BBB and BBB-CCC; AAA-CCC share no complex.
    assert stats["CORUM gene pairs"] == 2


def test_signor_keys_entities_and_skips_non_proteins(raw, tmp_path):
    out = tmp_path / "gt"
    stats = cgt.convert_signor(raw, out)

    d = out / "SIGNOR-HUMAN"
    ents = pd.read_parquet(d / "Dimension_Entities.parquet")
    facts = pd.read_parquet(d / "Fact_Protein_Interactions.parquet")

    assert list(ents.columns) == [
        "ENTITY_KEY", "ENTITY_ID", "ENTITY_NAME", "TYPE", "DATABASE"]
    assert list(facts.columns) == [
        "SIGNOR_KEY", "ENTITYA_KEY", "ENTITYB_KEY", "EFFECT", "MECHANISM"]
    assert ents.ENTITY_KEY.dtype == "uint16"
    assert facts.ENTITYA_KEY.dtype == "uint16"
    # AAA, BBB and the chemical -- repeated entities are keyed once.
    assert len(ents) == 3
    assert len(facts) == 3
    assert stats["SIGNOR protein entities"] == 2
    # Both directions of AAA-BBB collapse; the chemical edge is not a gene pair.
    assert stats["SIGNOR gene pairs"] == 1


# --------------------------------------------------------------------------
# failure paths -- the point of the exercise
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["string_info", "string_aliases", "string_links"])
def test_missing_string_input_raises_naming_the_file(raw, tmp_path, key):
    (raw / cgt.RAW_FILES[key]).unlink()
    with pytest.raises(cgt.MissingSource, match=cgt.RAW_FILES[key]):
        cgt.convert_string(raw, tmp_path / "gt")


@pytest.mark.parametrize("key,fn", [("corum", "convert_corum"), ("signor", "convert_signor")])
def test_missing_source_explains_how_to_get_it(raw, tmp_path, key, fn):
    (raw / cgt.RAW_FILES[key]).unlink()
    with pytest.raises(cgt.MissingSource) as exc:
        getattr(cgt, fn)(raw, tmp_path / "gt")
    assert "http" in str(exc.value)


def test_truncated_download_is_not_treated_as_present(raw, tmp_path):
    (raw / cgt.RAW_FILES["corum"]).write_text("", encoding="utf-8")
    with pytest.raises(cgt.MissingSource):
        cgt.convert_corum(raw, tmp_path / "gt")


def test_links_file_without_experiments_column_raises(raw, tmp_path):
    # The 'links' file rather than 'links.full' -- it has no per-channel scores.
    with gzip.open(raw / cgt.RAW_FILES["string_links"], "wt") as fh:
        fh.write("protein1 protein2 combined_score\n")
        fh.write("9606.ENSP00000000001 9606.ENSP00000000002 900\n")
    with pytest.raises(ValueError, match="experiments"):
        cgt.convert_string(raw, tmp_path / "gt")


def test_corum_export_without_human_rows_raises(raw, tmp_path):
    text = (raw / cgt.RAW_FILES["corum"]).read_text()
    (raw / cgt.RAW_FILES["corum"]).write_text(
        text.replace("\tHuman\t", "\tMouse\t"), encoding="utf-8")
    with pytest.raises(ValueError, match="no human complexes"):
        cgt.convert_corum(raw, tmp_path / "gt")


def test_signor_single_pathway_export_raises(raw, tmp_path):
    header = (raw / cgt.RAW_FILES["signor"]).read_text().splitlines()[0]
    (raw / cgt.RAW_FILES["signor"]).write_text(header + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no interactions"):
        cgt.convert_signor(raw, tmp_path / "gt")


def test_renamed_upstream_column_raises_rather_than_dropping_the_source(raw, tmp_path):
    with gzip.open(raw / cgt.RAW_FILES["string_info"], "wt") as fh:
        fh.write("#string_protein_id\tdisplay_name\tprotein_size\n")
        fh.write("9606.ENSP00000000001\tAAA\t100\n")
    with pytest.raises(ValueError, match="preferred_name"):
        cgt.convert_string(raw, tmp_path / "gt")
