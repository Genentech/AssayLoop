"""Per-round pathway/complex composition of handoff hits, with model-derived
gene->gene influence bubbles.

Two views per screen:
  * Stacked bars: for each acquisition round, the proportion of that round's
    HITS belonging to each top pathway (Reactome) / complex (CORUM). Shows how
    the biological composition of discoveries shifts over the timeline.
  * Influence arcs: curved annotations connecting a gene hit in an early round
    to a downstream gene hit, labelled with the AssayLoop model's score change
    Delta = score(Y | context + {X observed as hit}) - score(Y | context).
    This shows how finding one hit raises the model's belief in another, within
    the screen's signature biology.

Requires the AssayLoop checkpoint (gf-bpmf-train-hits-rl-fg-s19). Per-round data
is read from the timeline cache (run paper_handoff_timeline first, or it will
be built on demand). Influence is cached to handoff_influence_cache.json.

Usage::

    uv run python -m assayloop.scripts.paper_handoff_composition --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

from assayloop import config
from assayloop.scripts.paper_handoff_timeline import (
    DEFAULT_SCREENS, ANALYSIS_DIR, METHOD_COLORS,
    load_method_rounds, _load_pathway_membership, _load_complex_membership,
    _shorten_pathway,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_handoff_composition")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
INFLUENCE_CACHE = ANALYSIS_DIR / "handoff_influence_cache.json"
HANDOFF_METHOD = "Gemini → AssayLoop Handoff"
HANDOFF_N = 3  # LLM warm-start rounds before AssayLoop takes over

# Row order for the composition figures (top -> bottom). Influence arcs (derived
# from the AssayLoop model) are drawn on the AssayLoop row.
METHOD_ROWS = ["AssayLoop", HANDOFF_METHOD, "Gemini-3.1-Pro"]
METHOD_LABELS = {
    "AssayLoop": "AssayFormer",
    HANDOFF_METHOD: "AssayLoop\n(Gemini Handoff)",
    "Gemini-3.1-Pro": "Gemini-3.1-Pro",
}

# Validated CVD-safe categorical palette (dataviz skill, light surface), in the
# fixed CVD-safe slot order, extended with two distinct hues for slots 9-10.
# Applied identically across all panels so colors are consistent by slot.
PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
    "#17a2b8",  # teal (extension)
    "#8c564b",  # brown (extension)
]
# When the two screens together need >10 categories, extend with the muted-but-
# distinct tab20b/tab20c hues (accepting a mild CVD tradeoff past slot 10 — the
# user asked for more colors to read the compositional change).
import matplotlib.colors as _mcolors  # noqa: E402
EXT_PALETTE = PALETTE + [
    _mcolors.to_hex(c) for c in
    list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
]
OTHER_COLOR = "#d9d9d9"
TOP_K = 14

# Global group->color registry, shared across BOTH screens within a figure so a
# pathway/complex keeps one color everywhere (reset per membership figure).
_COLOR_REGISTRY = {}


def _color_for(group):
    if group == "Other":
        return OTHER_COLOR
    if group not in _COLOR_REGISTRY:
        _COLOR_REGISTRY[group] = EXT_PALETTE[len(_COLOR_REGISTRY) % len(EXT_PALETTE)]
    return _COLOR_REGISTRY[group]


def _pooled_order(all_rounds, screens, method_names, membership):
    """All groups ordered by pooled distinct hit genes across screens+methods."""
    grp_genes = defaultdict(set)
    for screen, _ in screens:
        for m in method_names:
            for genes, hits in all_rounds.get(screen, {}).get(m, []):
                for g, h in zip(genes, hits):
                    if h:
                        for grp in membership.get(g.upper(), ()):
                            grp_genes[grp].add((screen, g.upper()))
    return sorted(grp_genes, key=lambda gp: -len(grp_genes[gp]))

# Curated signature genes per screen (all verified hits in the handoff run).
# Probe->target influence is computed within each set.
SIGNATURE_GENES = {
    "U_1733_merged": [
        "TNFRSF1A", "TRADD", "TRAF2", "RIPK1", "MAP3K7", "TAB2",
        "IKBKG", "CHUK", "IKBKB", "NFKBIA", "TNFAIP3", "RNF31", "RBCK1",
    ],
    "2466": [
        # HUSH / heterochromatin silencing (de-repress the AAV transgene)
        "SETDB1", "MORC2", "MPHOSPH8", "ATRX",
        # Fanconi anemia core complex
        "FANCA", "FANCD2", "FANCI", "FANCM", "UBE2T",
        # SMC5/6 and MRN genome maintenance
        "SMC5", "SMC6", "NSMCE2", "MRE11", "NBN",
    ],
}

def _hit_round_map(rounds):
    """gene(upper) -> 1-based round of first hit."""
    hr = {}
    for i, (genes, hits) in enumerate(rounds):
        for g, h in zip(genes, hits):
            if h and g.upper() not in hr:
                hr[g.upper()] = i + 1
    return hr


def _shared_group(x, y, memberships):
    """Return a pathway/complex shared by genes x and y, or None.

    We give ~100 genes to the model each round, so a hit can't be causally tied
    to one future hit. For the paper we therefore only draw a probe→target arc
    when the two genes share a known pathway or complex (a defensible functional
    link), not merely a high model influence score."""
    for mem in memberships:
        shared = mem.get(x.upper(), set()) & mem.get(y.upper(), set())
        if shared:
            # prefer the smallest (most specific) shared group
            return min(shared, key=len)
    return None


def _select_forward_pairs(screen, rounds, influence, memberships, k=3):
    """Top-k positive probe→target influences where the target is discovered in
    a LATER round than the probe AND the two genes share a pathway/complex."""
    scr = {p: t for p, t in influence.get(screen, {}).items()
           if not p.startswith("_")}
    hr = _hit_round_map(rounds)
    cands = []
    for x, tgts in scr.items():
        rx = hr.get(x.upper())
        if rx is None:
            continue
        for y, d in tgts.items():
            ry = hr.get(y.upper())
            if x == y or ry is None or ry <= rx or d <= 0:
                continue
            grp = _shared_group(x, y, memberships)
            if grp is None:
                continue
            cands.append((d, x, y, rx, ry, grp))
    cands.sort(key=lambda t: -t[0])
    picks, used_x, used_y = [], set(), set()
    for d, x, y, rx, ry, grp in cands:
        if x in used_x or y in used_y:
            continue
        picks.append((x, y, d, rx, ry, grp))
        used_x.add(x)
        used_y.add(y)
        if len(picks) >= k:
            break
    return picks


def _select_signature_bubbles(screen, rounds, influence, k=3):
    """Top-k positive probe→target influences among the co-discovered signature
    genes (same round), shown as bubbles rather than self-loops."""
    scr = {p: t for p, t in influence.get(screen, {}).items()
           if not p.startswith("_")}
    hr = _hit_round_map(rounds)
    sig = set(g.upper() for g in SIGNATURE_GENES.get(screen, []))
    cands = []
    for x, tgts in scr.items():
        if x.upper() not in sig or hr.get(x.upper()) is None:
            continue
        for y, d in tgts.items():
            if x == y or y.upper() not in sig or hr.get(y.upper()) is None:
                continue
            if hr[x.upper()] != hr[y.upper()] or d <= 0:
                continue
            cands.append((d, x, y, hr[x.upper()]))
    cands.sort(reverse=True)
    picks, used = [], set()
    for d, x, y, r in cands:
        key = (x, y)
        if key in used or (y, x) in used:
            continue
        picks.append((x, y, d, r))
        used.add(key)
        if len(picks) >= k:
            break
    return picks


def _group_sizes(membership):
    """group name -> total number of genes (for specificity ranking)."""
    sz = Counter()
    for grps in membership.values():
        for grp in grps:
            sz[grp] += 1
    return sz


def _most_specific(gene, membership, top_set, sizes):
    """Assign a gene to its smallest (most specific) group within top_set."""
    best, best_size = None, 10 ** 9
    for grp in membership.get(gene.upper(), ()):
        if grp not in top_set:
            continue
        size = sizes.get(grp, 10 ** 9)
        if size < best_size:
            best, best_size = grp, size
    return best


def _screen_top_groups(all_rounds, screen, method_names, membership, k=TOP_K):
    """Top-k groups for ONE screen, pooled across its methods. Per-screen (not
    global) so each screen shows its own biology; colors are shared across that
    screen's method rows (the key AssayLoop-vs-Handoff-vs-Gemini comparison)."""
    grp_genes = defaultdict(set)
    for m in method_names:
        for genes, hits in all_rounds.get(screen, {}).get(m, []):
            for g, h in zip(genes, hits):
                if h:
                    for grp in membership.get(g.upper(), ()):
                        grp_genes[grp].add(g.upper())
    ranked = sorted(grp_genes, key=lambda gp: -len(grp_genes[gp]))
    return ranked[:k]


OOU_KEY = "__out_of_universe__"  # batch genes outside the filtered universe


def _composition_by_round(rounds, membership, top_groups, sizes):
    """counts[round][group] = # HIT genes assigned to group; plus 'Other'."""
    top_set = set(top_groups)
    counts = []
    for genes, hits in rounds:
        c = Counter()
        for g, h in zip(genes, hits):
            if not h:
                continue
            grp = _most_specific(g, membership, top_set, sizes)
            c[grp if grp else "Other"] += 1
        counts.append(c)
    return counts


def _batch_composition_by_round(rounds, membership, top_groups, sizes, universe):
    """counts[round][group] = # of ALL suggested genes assigned to group.

    In-universe genes are colored by category (or 'Other'); genes outside the
    filtered universe go to OOU_KEY (drawn as a distinct 'wasted pick' segment).
    Bar height = batch size, so out-of-universe suggestions add visible waste."""
    top_set = set(top_groups)
    counts = []
    for genes, hits in rounds:
        c = Counter()
        for g in genes:
            if g.upper() not in universe:
                c[OOU_KEY] += 1
                continue
            grp = _most_specific(g, membership, top_set, sizes)
            c[grp if grp else "Other"] += 1
        counts.append(c)
    return counts


UNIVERSE_CACHE = ANALYSIS_DIR / "universe_genes.json"


def _load_universe(refresh=False):
    """Genes appearing in >=2 public screens (the filtered universe). Cached."""
    if UNIVERSE_CACHE.exists() and not refresh:
        try:
            return set(json.loads(UNIVERSE_CACHE.read_text()))
        except Exception:
            pass
    from assayloop.tasks import load_screens
    freq = Counter()
    for s in load_screens(target_set="public"):
        for g in set(s.genes):
            freq[g.upper()] += 1
    uni = sorted(g for g in freq if freq[g] >= 2)
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_CACHE.write_text(json.dumps(uni))
    log.info("Universe: %d genes (>=2 screens)", len(uni))
    return set(uni)


# Reactome top-level categories, ordered by priority (informative first, giant
# catch-alls last). Each hit gene is assigned to the highest-priority category
# it belongs to -> broad, mostly-colored composition bars (vs the long tail of
# specific pathways, which leaves most bars grey).
BROAD_CATEGORIES = [
    "DNA Repair", "DNA Replication", "Cell Cycle", "Chromatin organization",
    "Programmed Cell Death", "Cellular responses to stress",
    "Metabolism of RNA", "Gene expression (Transcription)",
    "Metabolism of proteins", "Metabolism of lipids", "Metabolism",
    "Cytokine Signaling in Immune system", "Innate Immune System",
    "Adaptive Immune System", "Immune System",
    "Membrane Trafficking", "Vesicle-mediated transport",
    "Transport of small molecules", "Hemostasis",
    "Signaling by Rho GTPases", "Signal Transduction",
    "Developmental Biology", "Disease",
]


# Coarse CORUM complex families, matched by keyword on the complex name (first
# match wins, so order = priority). Reduces the specific-complex long tail.
COMPLEX_FAMILIES = [
    ("Ribosome / translation / rRNA",
     ["ribosom", "40s", "60s", "43s", "48s", "polysome", "eif", "translation",
      "ejc", "exon junction", "nop56", "nop58", "pre-rrna", "rrna", "nucleolar",
      "utp", "aminoacyl", "synthetase", "trna", "keops", "exosome",
      "rnase", "mrp", "rna processing"]),
    ("Spliceosome", ["spliceosom", "snrnp", "u1 ", "u2 ", "u4/u6", "u5 ",
                     "prp19", "sm core"]),
    ("Proteasome", ["proteasome", "26s", "20s", "19s", "pa28", "pa700",
                    "pa200"]),
    ("DNA repair / replication",
     ["repair", "rad51", "brca", "mrn", "9-1-1", "replicat", "mcm", "fanconi",
      "rpa ", "origin", "recombinat", "shieldin", "resect", "orc", "rfc",
      "ctf18", "replication factor", "gins", "cmg", "primase", "pol delta",
      "pol epsilon", "swsap", "zswim", "shu complex", "bloom", "blm "]),
    ("Transcription / Mediator",
     ["mediator", "rna polymerase", "transcription", "tfii", "elongat",
      "pol ii", "pol i ", "integrator", "nelf", "paf1", "tfiid", "tfiih"]),
    ("Chromatin / cohesin / histone",
     ["chromatin", "histone", "hat ", "hdac", "nucleosome", "swi/snf", "baf",
      "prc", "polycomb", "saga", "staga", "nurd", "ino80", "set ", "mll",
      "cohesin", "condensin", "nipbl", "rad21", "smc1", "smc3", "smc5", "smc6",
      "mau2", "stag", "wapl", "scc", "set1", "nsl", "kat", "moz", "hbo1",
      "dot1", "hira", "dnmt", "sin3", "atac"]),
    ("TNF / NF-κB signaling", ["tnf", "nf-kappa", "nf-κb", "tnfr", "ikk",
                               "i-kappa", "traf", "tradd", "rip1", "lubac"]),
    ("Nuclear pore / transport",
     ["nuclear pore", "nucleoporin", "importin", "exportin", "karyopherin",
      "tnpo", "transportin", "ran-", "ran ", "tho complex", "thoc",
      "trex", "mrna export", "nxf1"]),
    ("Ubiquitin ligase", ["ubiquitin", " e3", "scf ", "cullin", "crl", "apc/c",
                          "anaphase-promoting", "cop9", "signalosome"]),
    ("Mitochondrial / OXPHOS", ["mitochond", "respiratory", "atp synthase",
                                "oxphos", "cytochrome", "tim ", "tom "]),
    ("Chaperone / folding",
     ["cct", "tcp1", "chaperonin", "prefoldin", "chaperone", "r2tp", "pfd",
      "hsp", "tric"]),
    ("Actin / cytoskeleton",
     ["arp2/3", "arp3", "actin", "wave", "whamm", "wash", "formin", "spectrin",
      "myosin", "capz", "capping", "ccc complex", "ccc-", "retriever"]),
    ("Vesicle / trafficking",
     ["copi", "copii", "clathrin", "snare", "exocyst", "retromer", "hops",
      "vesicle", "adaptor ", "ist1", "escrt", "rab", "arf", "golgi"]),
    ("Kinase / phosphatase / cell cycle",
     ["kinase", "phosphatase", "cdk", "cyclin", "pp2a", "pp1 ", "mis12",
      "kinetochore", "mitotic", "stripak", "striatin", "chromosomal passenger",
      "aurora", "incenp", "cen complex", "cenp", "centromer", "centrosom",
      "cep", "centriol", "spindle", "ndc80"]),
]


def _match_complex_family(name):
    low = (name or "").lower()
    for fam, kws in COMPLEX_FAMILIES:
        if any(kw in low for kw in kws):
            return fam
    return None


def _load_complex_family_membership():
    """gene(upper) -> {coarse CORUM family names} (keyword-matched)."""
    import pandas as pd
    # Raises if ASSAYLOOP_GROUND_TRUTH is unset/absent rather than returning
    # an empty membership, which would read as "no complexes recovered".
    corum_dir = config.ground_truth_dir("CORUM-HUMAN")
    membership = defaultdict(set)
    subunits = pd.read_parquet(corum_dir / "dimension_Subunit.parquet")
    try:
        cx = pd.read_parquet(corum_dir / "fact_Complex.parquet")
        cid_to_name = dict(zip(cx.index, cx["complex_name"]))
    except Exception:
        cid_to_name = {}
    by_cid = defaultdict(set)
    for _, row in subunits.iterrows():
        g = row.get("gene_name")
        if isinstance(g, str) and g.strip():
            by_cid[row["complex_id"]].add(g.strip().upper())
    for cid, genes in by_cid.items():
        fam = _match_complex_family(cid_to_name.get(cid, ""))
        if fam:
            for g in genes:
                membership[g].add(fam)
    return membership


def _load_broad_pathway_membership():
    """gene(upper) -> {single Reactome top-level category} (highest priority)."""
    gmt = (Path(__file__).resolve().parents[1] / "data" / "gene_sets"
           / "ReactomePathways.gmt")
    prio = {c: i for i, c in enumerate(BROAD_CATEGORIES)}
    gene_best = {}
    if not gmt.exists():
        return defaultdict(set)
    with open(gmt) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[0] not in prio:
                continue
            cat = parts[0]
            for raw in parts[2:]:
                g = raw.strip().upper()
                if not g:
                    continue
                if g not in gene_best or prio[cat] < prio[gene_best[g]]:
                    gene_best[g] = cat
    membership = defaultdict(set)
    for g, cat in gene_best.items():
        membership[g].add(cat)
    return membership


# ----------------------------------------------------------------------------
# Influence (AssayLoop model)
# ----------------------------------------------------------------------------

INFLUENCE_VERSION = 3  # bump when the probe/target scheme changes


def _compute_influence(checkpoint, ckpt_file, screens, device, targets_by_screen):
    """Return {screen: {probe: {target: delta}}}.

    Probes are the signature genes; targets are ALL hit genes in the screen's
    handoff run (so we can find forward-in-time probe→later-hit relationships).
    delta = V[target] . (encode(desc, ctx={probe:hit}) - encode(desc, ctx={})).
    """
    import torch
    from assayloop.amortized.data import GeneVocab
    from assayloop.amortized.model import RankerConfig, RankerNet
    from assayloop.amortized.text_embed import (
        get_text_embedder, embed_texts, screen_description_text)
    from assayloop.tasks import load_screens

    dev = torch.device(device if device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = RANKERS_DIR / checkpoint
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    vocab = GeneVocab.load(ckpt_dir / "vocab.json")
    net = RankerNet(RankerConfig(**cfg["arch"]))
    state = torch.load(ckpt_dir / ckpt_file, map_location=dev, weights_only=False)
    net.load_state_dict(state)
    net.to(dev).eval()

    embedder = get_text_embedder(cfg.get("text_backend", "auto"))
    V = net.gene_emb.weight.detach()

    screen_recs = {s.dataset_name: s for s in load_screens(target_set="public")}

    def _vid(g):
        v = vocab.to_idx(g)
        return v if v != 0 else vocab.to_idx(g.upper())

    out = {}
    for screen, _ in screens:
        rec = screen_recs.get(screen)
        if rec is None:
            log.warning("Screen %s not in public set; skipping influence", screen)
            continue
        text = screen_description_text(rec)
        emb = embed_texts([text], embedder)[0]
        desc = torch.tensor(emb, dtype=torch.float32, device=dev).unsqueeze(0)

        probes = {g: _vid(g) for g in SIGNATURE_GENES.get(screen, [])}
        probes = {g: v for g, v in probes.items() if v}
        targets = {g: _vid(g) for g in targets_by_screen.get(screen, [])}
        targets = {g: v for g, v in targets.items() if v}
        if not probes or not targets:
            continue

        target_syms = list(targets.keys())
        target_ids = torch.tensor([targets[g] for g in target_syms], device=dev)
        V_tgt = V[target_ids]  # (T, d)

        empty_ids = torch.zeros(1, 0, dtype=torch.long, device=dev)
        empty_hit = torch.zeros(1, 0, dtype=torch.long, device=dev)
        empty_pad = torch.zeros(1, 0, device=dev)
        with torch.no_grad():
            base_repr = net.encode(desc, empty_ids, empty_hit, empty_pad)

        scr_out = {}
        for probe, pid in probes.items():
            ctx_ids = torch.tensor([[pid]], device=dev)
            ctx_hit = torch.tensor([[1]], device=dev)
            ctx_pad = torch.zeros(1, 1, device=dev)
            with torch.no_grad():
                probe_repr = net.encode(desc, ctx_ids, ctx_hit, ctx_pad)
                delta = (probe_repr - base_repr).squeeze(0)
                dscore = (V_tgt @ delta).cpu().numpy()
            scr_out[probe] = {t: float(v) for t, v in zip(target_syms, dscore)}
        out[screen] = scr_out
        log.info("Influence: %s (%d probes x %d targets)", screen,
                 len(probes), len(target_syms))

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"_version": INFLUENCE_VERSION, **out}


def _load_or_compute_influence(checkpoint, ckpt_file, screens, device, refresh,
                               targets_by_screen):
    if INFLUENCE_CACHE.exists() and not refresh:
        cached = json.loads(INFLUENCE_CACHE.read_text())
        if (cached.get("_version") == INFLUENCE_VERSION
                and all(s in cached for s, _ in screens)):
            log.info("Using cached influence -> %s", INFLUENCE_CACHE)
            return cached
    inf = _compute_influence(checkpoint, ckpt_file, screens, device,
                             targets_by_screen)
    INFLUENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INFLUENCE_CACHE.write_text(json.dumps(inf, indent=2))
    log.info("Wrote influence cache -> %s", INFLUENCE_CACHE)
    return inf


def _first_hit_round(rounds, gene):
    """1-based round in which `gene` is first acquired as a hit (or None)."""
    for i, (genes, hits) in enumerate(rounds):
        for g, h in zip(genes, hits):
            if h and g.upper() == gene.upper():
                return i + 1
    return None


CANDIDATES_PATH = ANALYSIS_DIR / "handoff_influence_candidates.txt"


def _batch_group_counts(genes, membership):
    """Counter of group -> # suggested genes in that group (multi-membership;
    a gene in several groups counts for each)."""
    c = Counter()
    for g in genes:
        for grp in membership.get(g.upper(), ()):
            c[grp] += 1
    return c


def _print_influence_candidates(screens, all_rounds, influence,
                                broad_mem, pw_mem, cx_mem, cxfam_mem,
                                min_count=4, top_n=4):
    """Write + print candidates for manual annotation, per screen and method:

      (a) STARTING STRATEGY — per round, how concentrated the SUGGESTED batch is
          in each pathway (broad Reactome category) and complex family. Reveals
          e.g. 'Gemini opens by proposing 30 DNA-repair genes'.
      (b) FORWARD hit->hit — probe hit at round rX raises the model's score for a
          gene discovered at a later round rY, where the two share a known
          pathway/complex (a defensible functional link).

    (a) is a property of the batch a method proposes; (b) uses model inference
    (Δscore = belief shift when the probe is observed). The user hand-picks from
    this list and adds callouts to the image later."""
    lines = []
    lines.append("=" * 78)
    lines.append("HANDOFF ANNOTATION CANDIDATES (s19) — pick for callouts")
    lines.append("(a) starting strategy: suggested-batch concentration per round")
    lines.append("      pathway = broad Reactome category; family = CORUM family")
    lines.append("      (only groups with >= %d suggested genes are listed)" % min_count)
    lines.append("(b) forward hit->hit: probe(rX) raises model score for a later")
    lines.append("      hit(rY); genes share a pathway/complex. d = belief shift.")
    lines.append("=" * 78)
    for screen, label in screens:
        lines.append("")
        lines.append("### %s  (%s)" % (label, screen))
        for method in METHOD_ROWS:
            rounds = all_rounds.get(screen, {}).get(method)
            if not rounds:
                continue
            lines.append("  -- %s --" % METHOD_LABELS[method].replace("\n", " "))

            lines.append("   (a) starting strategy (suggested-batch concentration):")
            for i, (genes, _hits) in enumerate(rounds):
                pw = _batch_group_counts(genes, broad_mem)
                fam = _batch_group_counts(genes, cxfam_mem)
                pw_top = [(g, n) for g, n in pw.most_common(top_n) if n >= min_count]
                fam_top = [(g, n) for g, n in fam.most_common(top_n)
                           if n >= min_count]
                if not pw_top and not fam_top:
                    continue
                pw_s = ", ".join("%s x%d" % (_shorten_pathway(g), n)
                                 for g, n in pw_top) or "-"
                fam_s = ", ".join("%s x%d" % (g, n) for g, n in fam_top) or "-"
                lines.append("       r%-2d pathway: %s" % (i + 1, pw_s))
                lines.append("           family:  %s" % fam_s)

            fwd = _select_forward_pairs(screen, rounds, influence,
                                        [pw_mem, cx_mem], k=12)
            if fwd:
                lines.append("   (b) forward hit->hit (shared pathway/complex):")
                for x, y, d, rx, ry, grp in fwd:
                    lines.append("       r%d->r%-2d  %-8s -> %-8s  d%+.2f   [%s]"
                                 % (rx, ry, x, y, d, _shorten_pathway(grp)))
    lines.append("=" * 78)
    text = "\n".join(lines) + "\n"
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(text)
    print(text)
    log.info("Wrote annotation candidates -> %s", CANDIDATES_PATH)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="gf-bpmf-train-hits-rl-fg-s19")
    ap.add_argument("--ckpt-file", default="model_last.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--screens", default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute per-round + influence caches")
    args = ap.parse_args()

    screens = ([(s.strip(), s.strip()) for s in args.screens.split(",")]
               if args.screens else DEFAULT_SCREENS)

    log.info("Loading membership + per-round data...")
    pw_mem = _load_pathway_membership()             # specific pathways (arcs)
    broad_mem = _load_broad_pathway_membership()    # top-level categories (bars)
    cx_mem = _load_complex_membership()             # specific CORUM complexes
    cxfam_mem = _load_complex_family_membership()   # coarse CORUM families
    all_rounds = load_method_rounds(screens, refresh=args.refresh)

    # Targets for influence = hit genes across ALL methods (union), so the arc
    # selection works whichever method's discovery rounds we anchor to.
    targets_by_screen = {}
    for screen, _ in screens:
        hit_genes = set()
        for method in METHOD_ROWS:
            for genes, hits in all_rounds.get(screen, {}).get(method, []):
                for g, h in zip(genes, hits):
                    if h:
                        hit_genes.add(g)
        targets_by_screen[screen] = sorted(hit_genes)

    influence = _load_or_compute_influence(
        args.checkpoint, args.ckpt_file, screens, args.device, args.refresh,
        targets_by_screen)

    out_dir = ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(args.refresh)

    _print_influence_candidates(screens, all_rounds, influence,
                                broad_mem, pw_mem, cx_mem, cxfam_mem)

    # (bar_membership, arc_membership, metric_name, tag).
    #  - pathway: broad top-level categories for bars, specific pathways for arcs
    #  - complex: specific CORUM complexes
    #  - complex_families: coarse keyword families (request: less grey)
    figures = [
        (broad_mem, pw_mem, "Reactome Pathway", "pathway"),
        (cx_mem, cx_mem, "CORUM Complex", "complex"),
        (cxfam_mem, cx_mem, "CORUM Complex family", "complex_families"),
    ]
    for mem, arc_mem, metric_name, tag in figures:
        _make_diverging_figure(mem, arc_mem, metric_name, tag, screens,
                               all_rounds, influence, universe, out_dir=out_dir)


def _make_diverging_figure(mem, arc_mem, metric_name, tag, screens, all_rounds,
                           influence, universe, out_dir):
    """Two stacked bars per round, vertically arranged: the SUGGESTED batch
    composition points up (height = batch size; out-of-universe picks form a
    distinct 'wasted' segment), the HITS composition points down. Each half is
    normalized to its own max so both compositions are readable; absolute counts
    are printed on the bars."""
    n = len(screens)
    nrows = len(METHOD_ROWS)
    sizes = _group_sizes(mem)
    _COLOR_REGISTRY.clear()

    # Per-screen top groups; assign colors in global-frequency order (consistent
    # across screens + ordered legend).
    screen_tops = {s: _screen_top_groups(all_rounds, s, METHOD_ROWS, mem, k=TOP_K)
                   for s, _ in screens}
    legend_groups = set().union(*screen_tops.values()) if screen_tops else set()
    for g in _pooled_order(all_rounds, screens, METHOD_ROWS, mem):
        if g in legend_groups:
            _color_for(g)

    # Per-screen normalization maxima
    batch_max, hits_max = {}, {}
    for screen, _ in screens:
        bm = hm = 1
        for method in METHOD_ROWS:
            for genes, hits in all_rounds.get(screen, {}).get(method, []):
                bm = max(bm, len(genes))
                hm = max(hm, sum(hits))
        batch_max[screen], hits_max[screen] = bm, hm

    fig, axes = plt.subplots(nrows, n, figsize=(8 * n, 13), squeeze=False)
    GAP = 0.07  # blank margin on each side of the zero baseline

    for col, (screen, screen_label) in enumerate(screens):
        top_groups = screen_tops[screen]
        bmax, hmax = batch_max[screen], hits_max[screen]

        for row, method in enumerate(METHOD_ROWS):
            ax = axes[row][col]
            rounds = all_rounds.get(screen, {}).get(method)
            if not rounds:
                ax.set_visible(False)
                continue

            batch = _batch_composition_by_round(rounds, mem, top_groups, sizes,
                                                universe)
            hitc = _composition_by_round(rounds, mem, top_groups, sizes)
            n_rounds = len(rounds)
            x = np.arange(1, n_rounds + 1)

            # Distinct background tint per half so the two independently-scaled
            # zones (batch up to ~100, hits up to ~15) don't read as one axis.
            ax.axhspan(0, 1 + GAP + 0.28, color="#4a3aa7", alpha=0.05, zorder=0)
            ax.axhspan(-(1 + GAP + 0.28), 0, color="#e34948", alpha=0.06, zorder=0)

            # UP: batch composition (normalized to batch_max), OOU on top
            up = np.full(n_rounds, GAP)
            for cat in top_groups + ["Other", OOU_KEY]:
                vals = np.array([batch[i].get(cat, 0) for i in range(n_rounds)],
                                dtype=float) / bmax
                color = "#777777" if cat == OOU_KEY else _color_for(cat)
                hatch = "///" if cat == OOU_KEY else None
                ax.bar(x, vals, bottom=up, width=0.8, color=color,
                       edgecolor="white", linewidth=0.4, hatch=hatch)
                up += vals
            batch_tot = [int(sum(batch[i].values())) for i in range(n_rounds)]
            for i in range(n_rounds):
                ax.text(x[i], up[i] + 0.02, "%d" % batch_tot[i], ha="center",
                        va="bottom", fontsize=5, color="#666666")

            # DOWN: hits composition (normalized to hits_max), drawn negative
            dn = np.full(n_rounds, GAP)
            for cat in top_groups + ["Other"]:
                vals = np.array([hitc[i].get(cat, 0) for i in range(n_rounds)],
                                dtype=float) / hmax
                ax.bar(x, -vals, bottom=-dn, width=0.8, color=_color_for(cat),
                       edgecolor="white", linewidth=0.4)
                dn += vals
            hit_tot = [int(sum(hitc[i].values())) for i in range(n_rounds)]
            for i in range(n_rounds):
                ax.text(x[i], -dn[i] - 0.03, "%d" % hit_tot[i], ha="center",
                        va="top", fontsize=5, color="#666666")

            # Per-panel cumulative total hits (bottom-right, in method color).
            mc = METHOD_COLORS.get(method, "#333333")
            ax.text(0.985, 0.02, "total hits: %d" % sum(hit_tot),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, fontweight="bold", color=mc,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=mc, lw=0.8, alpha=0.9))

            ax.axhline(0, color="#333333", lw=0.8)

            # Handoff marker: LLM warm-starts n rounds, then AssayLoop takes over.
            if method == HANDOFF_METHOD and n_rounds > HANDOFF_N:
                xh = HANDOFF_N + 0.5
                ax.axvline(xh, color=mc, lw=1.0, ls="--", alpha=0.7, zorder=1)
                ax.text(xh, -(1 + GAP + 0.24), "Handoff (n = %d)" % HANDOFF_N,
                        ha="center", va="bottom", fontsize=8, color=mc)

            is_top = (row == 0)
            ax.set_ylim(-(1 + GAP + 0.28), 1 + GAP + 0.28)
            ax.set_xlim(0.4, n_rounds + 0.6)
            ax.set_xticks(x)
            ax.set_yticks([-(1 + GAP), 0, 1 + GAP])
            ax.set_yticklabels(["%d hits" % hmax, "0", "%d batch" % bmax],
                               fontsize=8.5)
            if row == nrows - 1:
                ax.set_xlabel("Acquisition round", fontsize=9)
            if col == 0:
                mc = METHOD_COLORS.get(method, "#333333")
                ax.set_ylabel("%s\n\n↓ hits    ↑ batch" % METHOD_LABELS[method],
                              fontsize=9.5, color=mc, fontweight="bold",
                              labelpad=8)
            if is_top:
                ax.set_title(screen_label, fontsize=14, fontweight="bold", pad=6)

    # Combined legend (registry colors + OOU + Other)
    ordered = [g for g in _pooled_order(all_rounds, screens, METHOD_ROWS, mem)
               if g in legend_groups]
    handles = [mpatches.Patch(color=_color_for(g), label=_shorten_pathway(g))
               for g in ordered]
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other (in %s)"
                                  % metric_name.split()[-1].lower()))
    handles.append(mpatches.Patch(facecolor="#777777", hatch="///",
                                  edgecolor="white",
                                  label="Out-of-universe (wasted suggestion)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0.01, 0.06, 1, 0.99])

    fname = out_dir / ("paper_handoff_composition_%s.png" % tag)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / ("paper_handoff_composition_%s.pdf" % tag),
                dpi=300, bbox_inches="tight")
    log.info("Wrote %s", fname)
    plt.close(fig)


ARC_COLOR = "#00897B"
ARC_FILL = "#E0F2F1"
ARC_INK = "#004D40"


def _draw_influence_arcs(ax, forward_pairs, bubbles, n_rounds, ymax):
    """Annotate model-derived influence above the bars.

    Band positions are expressed as multiples of ymax (the bar y-scale) so the
    annotations sit in the reserved headroom regardless of hit-count scale.

    forward_pairs: (probe, target, delta, rx, ry, shared_group) with ry > rx ->
        curved forward arrow from the probe's round to the target's round,
        labelled with the pathway/complex the two genes share.
    bubbles: (probe, target, delta, round) co-discovered signature pairs ->
        drawn as a compact strip near the top of the headroom.
    """
    mid_x = (n_rounds + 1) / 2
    if bubbles:
        r0 = bubbles[0][3]
        strip = "   ·   ".join("%s→%s Δ+%.1f" % (p, t, d)
                               for p, t, d, _ in bubbles)
        ax.text(
            mid_x, ymax * 1.88,
            "Round-%d signature co-boosts:  %s" % (r0, strip),
            ha="center", va="center", fontsize=6.6, fontweight="bold",
            color=ARC_INK, zorder=12,
            bbox=dict(boxstyle="round,pad=0.35", fc=ARC_FILL,
                      ec=ARC_COLOR, lw=0.8, alpha=0.97))

    # Forward-in-time arcs, staggered bands within the headroom
    bands = [ymax * 1.20, ymax * 1.42, ymax * 1.64]
    for k, (probe, target, delta, rx, ry, grp) in enumerate(forward_pairs):
        arc_y = bands[k % len(bands)]
        arrow = FancyArrowPatch(
            (rx, arc_y), (ry, arc_y),
            connectionstyle="arc3,rad=-0.35",
            arrowstyle="-|>", mutation_scale=12,
            color=ARC_COLOR, lw=1.8, alpha=0.9, zorder=10)
        ax.add_patch(arrow)
        ax.plot([rx, ry], [arc_y, arc_y], "o", ms=4, color=ARC_COLOR, zorder=11)
        ax.annotate(
            "%s (r%d) → %s (r%d)   Δ+%.1f\n%s" % (
                probe, rx, target, ry, delta, _shorten_pathway(grp)),
            xy=((rx + ry) / 2, arc_y + ymax * 0.03), ha="center", va="bottom",
            fontsize=6.4, fontweight="bold", color=ARC_INK, zorder=12,
            bbox=dict(boxstyle="round,pad=0.3", fc=ARC_FILL,
                      ec=ARC_COLOR, lw=0.7, alpha=0.97))


if __name__ == "__main__":
    main()
