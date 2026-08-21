"""Two-level Reactome hierarchy for pathway-composition figures.

The GMT we score against (``ReactomePathways.gmt``, filtered to 5-200 gene sets
with the broad disease/infection sets dropped) is flat. Reactome also publishes
the pathway tree, so we can lift every one of those leaf sets to one of the 29
top-level Homo sapiens root pathways ("Signal Transduction", "Metabolism",
"Immune System", ...). That gives the inner/outer rings of a sunburst.

Files (downloaded once into ``src/assayloop/data/gene_sets``):
  * ``ReactomePathways.txt``          stable_id, name, species
  * ``ReactomePathwaysRelation.txt``  parent_id, child_id

A pathway can have several parents; we keep the alphabetically-first root so the
mapping is deterministic. Result is cached to ``reactome_two_level.json``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

GENE_SETS = Path(__file__).resolve().parents[1] / "data" / "gene_sets"
CACHE = GENE_SETS / "reactome_two_level.json"


def _build() -> dict:
    id2name, name2id = {}, {}
    for ln in (GENE_SETS / "ReactomePathways.txt").read_text().splitlines():
        p = ln.split("\t")
        if len(p) == 3 and p[2] == "Homo sapiens":
            id2name[p[0]] = p[1]
            name2id.setdefault(p[1], p[0])

    parents = defaultdict(list)
    parents_inv = defaultdict(list)
    has_parent = set()
    for ln in (GENE_SETS / "ReactomePathwaysRelation.txt").read_text().splitlines():
        p = ln.split("\t")
        if len(p) == 2 and p[0].startswith("R-HSA-") and p[1].startswith("R-HSA-"):
            parents[p[1]].append(p[0])
            parents_inv[p[0]].append(p[1])
            has_parent.add(p[1])
    roots = sorted(i for i in id2name if i not in has_parent)

    root_set = set(roots)

    def _roots_of(pid, seen=None):
        """All root ancestors of pid (breadth-first over the parent DAG)."""
        if pid in root_set:
            return [pid]
        seen = seen or set()
        out, frontier = [], [pid]
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for par in parents.get(cur, []):
                if par in root_set:
                    out.append(par)
                else:
                    frontier.append(par)
        return out

    # level-2 nodes: the direct children of each root
    tier2 = {c for r in roots for c in parents_inv.get(r, [])}

    def _tier2_of(pid, root_id):
        """Level-2 ancestors of pid that sit under root_id."""
        if pid in tier2 and root_id in parents.get(pid, []):
            return [pid]
        out, frontier, seen = [], [pid], set()
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for par in parents.get(cur, []):
                if par in tier2 and root_id in parents.get(par, []):
                    out.append(par)
                elif par != root_id:
                    frontier.append(par)
        return out

    # pathway NAME -> (top-level category, level-2 subcategory).
    # Both deterministic: alphabetically first among the valid ancestors.
    cat, sub = {}, {}
    for pid, nm in id2name.items():
        rids = _roots_of(pid)
        if not rids:
            cat[nm] = sub[nm] = nm
            continue
        rid = sorted(rids, key=lambda r: id2name[r])[0]
        cat[nm] = id2name[rid]
        t2 = sorted({id2name[t] for t in _tier2_of(pid, rid)})
        sub[nm] = t2[0] if t2 else id2name[rid]
    return {"category_of": cat, "subcategory_of": sub,
            "roots": sorted(id2name[r] for r in roots)}


def load() -> dict:
    """{"category_of": {leaf: root}, "subcategory_of": {leaf: tier2}, "roots": [...]}"""
    if CACHE.is_file():
        return json.loads(CACHE.read_text())
    data = _build()
    CACHE.write_text(json.dumps(data))
    return data


if __name__ == "__main__":
    if CACHE.is_file():
        CACHE.unlink()
    d = load()
    n2 = len(set(d["subcategory_of"].values()))
    print(f"{len(d['category_of'])} pathways -> {n2} level-2 groups "
          f"-> {len(d['roots'])} top-level categories")
    per = defaultdict(set)
    for leaf, c in d["category_of"].items():
        per[c].add(d["subcategory_of"][leaf])
    for c, s in sorted(per.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {len(s):3d} level-2 groups   {c}")
    print("cached to", CACHE)
