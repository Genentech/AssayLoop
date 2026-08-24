"""Effective number of Reactome pathways covered by a set of gene picks.

``exp(Shannon entropy)`` over the pathway distribution of a bag of genes,
exponentiated to give an effective count. A method that concentrates on a few
programs scores low; one that spreads across distinct biology scores high.

The vocabulary counted over is Reactome's **level-2 groups** (186 nodes here),
not the GMT's 2012 leaf sets -- see :func:`gmt_membership` for why the leaf tier
is too fine to carry an interpretable count.

**Each gene is assigned to exactly one of its pathways, drawn uniformly, and
the estimate is averaged over draws.** The obvious alternative -- give every
gene weight 1 split evenly across its ``k`` pathways -- is what the sunburst
wedges show, and it is what these counts equal *in expectation*, but it must
not be used for the entropy: it gives a bag of ``m`` genes a support ceiling of
``sum(k)`` rather than ``m``, so a method picking well-studied hub genes scores
high for annotation depth alone. Measured, that inverts the batch-scope
ranking: kNN's picks sit in 5.03 level-2 groups each against a uniform draw's
2.52, which is enough to score kNN *above* random on fractional weights at
batch scope (43.7 vs 41.1) while one pathway per gene puts it correctly below
(19.0 vs 21.8) and it sits far below at EP-D. The ratio is scale-free, so a
larger reference count widens the gap instead of closing it, and coarsening the
vocabulary does not remove it either -- the fix has to be the
one-pathway-per-gene assignment, which pins every method's ceiling to ``m``
exactly.

Three scopes, from the same pick stream:

==========  ======================================  ==========================
Column      Unit                                    Aggregation
==========  ======================================  ==========================
``EP-B``    one acquisition batch (100 genes)       mean over batches in a
                                                    screen, then over screens
``EP-S``    all picks in one screen (1000 genes)    mean over screens
``EP-D``    all picks pooled over the test set      none -- one number
==========  ======================================  ==========================

**Rarefaction.** ``exp(H)`` computed from observed frequencies is sample-size
sensitive in two ways: ``exp(H) <= S`` and the observed support ``S`` grows
with the number of picks, and the plug-in entropy is biased low by roughly
``(S-1)/(2N)`` nats because unobserved categories contribute 0 instead of
their true positive share. Methods do not supply equal numbers of *annotated*
genes -- only 47% of the f2 universe is in the filtered GMT at all, and methods
sit at 59%-82% -- so their raw estimates carry different bias and are not
comparable row to row.

Each scope is therefore subsampled to a fixed annotated-gene count
(``M_BATCH`` / ``M_SCREEN`` / ``M_DATASET``) and averaged over ``R_*`` draws.
The reference counts are *fixed constants*, deliberately not derived from the
minimum across whatever methods happen to be in the table: a min-over-methods
target would mean adding one new baseline row silently changed every other
row's score. Units with fewer than the reference count are dropped and counted
in ``ep_n_dropped``; if a scope loses more than ``1 - RETENTION`` of its units
the whole cell is reported as ``None``, because the units that survive are the
better-annotated ones and their mean would flatter the method.

Two things are deliberately kept apart at batch scope. A batch the method never
filled (LLM harnesses routinely emit short batches, and Kimi-K2.6's median batch
is 32 picks) is excluded from EP-B as a run artifact -- that is what the
Shortfall column measures. A *full* batch whose genes simply are not in Reactome
is real behaviour and is not excluded: BioBO, Haystacks and RF + UCB emit
complete 100-gene batches whose 5th percentile carries 4, 9 and 10 annotated
genes respectively, against ~47 for a uniform draw, so they dash out of EP-B
rather than being scored on their annotated minority.

The un-rarefied *fractional-weight* plug-in values are returned alongside as
``*_raw``. They are not the reported statistic -- they carry both the sample-
size bias and the annotation-depth confound above -- but they are what
:mod:`assayloop.scripts.plot_pathway_sunburst` printed historically, so they
are kept as the module's regression check against that implementation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np

# Rarefaction reference counts, in *annotated* genes per unit. See module
# docstring for why these are fixed rather than data-derived, and RETENTION
# below for what happens to a method that cannot supply them.
#
# Calibrated against the measured per-unit annotated-gene distribution over all
# 58 methods in the full-genome table, after truncated batches are excluded:
#   M_BATCH   30 -- 5th-percentile *full* batch supplies 47-62 annotated genes
#                   for every method except BioBO (4), Haystacks (9),
#                   RF + UCB (10) and Kimi-K2.6 (24), which dash out.
#   M_SCREEN 200 -- smallest per-screen count is 103 (Claude Haiku-4.5, which
#                   dashes); every other method clears 200.
#   M_DATASET 6000 -- smallest pooled count is 6,362 (Claude Haiku-4.5), so
#                   every method renders a value.
M_BATCH = 30
M_SCREEN = 200
M_DATASET = 6_000

# Fraction of a scope's units that must survive the >= M_* filter for its mean
# to be reported. The survivors are the better-annotated units, so below this
# the mean flatters the method and a dash is the honest cell.
RETENTION = 0.95

# Subsample draws averaged per unit. Set by measuring the seed-to-seed spread
# on real pick streams (seeds 0-2, the six sunburst methods) and raising R until
# it sits well under the printed precision. Already at R=100 the spread is
# <= 0.02 (EP-B), and EP-B is the tightest scope: it is bounded above by M_BATCH
# so its between-method range is ~8 (13.8 for AssayLLM to 21.8 for a uniform
# draw). That is a 400x signal-to-noise ratio; all three scopes are printed to
# one decimal.
R_BATCH = 400
R_SCREEN = 400
R_DATASET = 300

SEED = 0

_SCOPES = ("batch", "screen", "dataset")


def scope_rng(scope: str, seed: int = SEED):
    """The RNG stream :func:`effective_pathways` draws that scope's subsamples from.

    Exposed so a caller computing one scope standalone -- the sunburst's Random
    panel, the heatmap's EP-D row -- reproduces the table's number exactly
    instead of landing ~0.1 away on a different draw.
    """
    return np.random.default_rng([seed, _SCOPES.index(scope)])


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

OTHER_GROUP = "Other (unmapped Reactome leaf)"


@lru_cache(maxsize=1)
def leaf_membership() -> dict[str, tuple[str, ...]]:
    """gene (upper) -> Reactome *leaf* sets, the 5-200 gene disease-filtered GMT.

    The raw tier, before the level-2 lift :func:`gmt_membership` applies. Used
    by the sunburst and the LLM pathway heatmap, which do their own roll-up
    through :mod:`assayloop.scripts.pathway_hierarchy` and need leaf names to
    map from; nothing scored in the tables uses it.
    """
    from assayloop.scripts.paper_handoff_timeline import _load_pathway_membership
    return {g: tuple(sorted(ps)) for g, ps in _load_pathway_membership().items()}


@lru_cache(maxsize=1)
def gmt_membership() -> dict[str, tuple[str, ...]]:
    """gene (upper) -> Reactome *level-2* groups it belongs to.

    The GMT itself (5-200 gene sets, disease-filtered) is the same source the
    pathway sunburst and the handoff timeline use, so the three stay consistent
    by construction -- but its 2012 leaf sets are far too fine to count with.
    With 2012 categories over 10,480 annotated genes, 30 genes drawn one
    pathway each essentially never collide, so EP-B is pinned just under
    ``M_BATCH`` for every method (24.5-27.6 out of 30) and the statistic
    measures "how close to M did you get" rather than how much biology the
    batch touched.

    Each leaf is therefore lifted to its Reactome level-2 group -- the direct
    children of the 29 top-level roots ("Innate Immune System", "Cell Cycle,
    Mitotic", "Signaling by Receptor Tyrosine Kinases", ...) -- via
    :mod:`assayloop.scripts.pathway_hierarchy`, the same tier the sunburst's
    outer ring draws. 185 are populated by this GMT; the 36 leaf sets with no
    hierarchy entry share a single ``OTHER_GROUP`` bucket rather than each
    becoming its own node, which would pad the vocabulary with singletons, for
    186 nodes in all. A uniform draw from the f2 acquisition universe then
    scores 21.8 / 56.4 / 82.1 at the three scopes, so the counts read as
    absolute numbers of biological programs against a fixed, nameable
    denominator.

    Coarsening also makes the estimate far less sample-size dependent, because
    the vocabulary saturates well before the scope does: a uniform draw gains
    34.6 groups going from M=30 to M=200 but only 25.7 more going from M=200 to
    M=6000, against 79.5 and 714.6 at leaf tier. Only batch scope still leans
    hard on rarefaction. Lifting one tier further, to the 29 roots,
    is more readable still but flattens EP-D to 15.3-20.0 against a random
    ceiling of 19.2, collapsing the across-screen turnover gap from ~30 points
    to ~5. Level-2 is the coarsest vocabulary that keeps it.
    """
    from assayloop.scripts.pathway_hierarchy import load as load_hierarchy

    sub_of = load_hierarchy()["subcategory_of"]
    return {g: tuple(sorted({sub_of.get(p, OTHER_GROUP) for p in ps}))
            for g, ps in leaf_membership().items()}


def effective_n(weights) -> float:
    """exp(Shannon entropy) of a weight distribution, i.e. its effective size."""
    w = np.asarray(list(weights), float)
    w = w[w > 0]
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(math.exp(-(w * np.log(w)).sum()))


def pathway_weights(genes, membership=None) -> tuple[dict[str, float], int]:
    """Fractional leaf-pathway weights for a bag of genes.

    Returns ``(pathway -> weight, n_annotated)``. Unannotated genes are dropped,
    matching the sunburst's attribution.
    """
    membership = gmt_membership() if membership is None else membership
    by_path: dict[str, float] = defaultdict(float)
    n_ann = 0
    for g in genes:
        ps = membership.get(g.upper())
        if not ps:
            continue
        n_ann += 1
        w = 1.0 / len(ps)
        for p in ps:
            by_path[p] += w
    return dict(by_path), n_ann


# ---------------------------------------------------------------------------
# Rarefaction
# ---------------------------------------------------------------------------

class _Unit:
    """A bag of annotated genes in CSR form, cheap to subsample repeatedly.

    ``ptr[i]:ptr[i+1]`` slices ``pid``/``pw`` to gene ``i``'s pathway ids and
    their 1/len(pathways) weights. Drawing one pathway per gene is then an
    offset into each gene's own slice -- a handful of vectorised operations per
    draw, independent of how many genes are selected. ``pw`` is used only by
    :meth:`raw`, the legacy fractional-weight check.
    """

    __slots__ = ("ptr", "pid", "pw", "n")

    def __init__(self, genes, membership, pid_of):
        ptr = [0]
        pid: list[int] = []
        pw: list[float] = []
        for g in genes:
            ps = membership.get(g.upper())
            if not ps:
                continue
            w = 1.0 / len(ps)
            for p in ps:
                pid.append(pid_of[p])
                pw.append(w)
            ptr.append(len(pid))
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.pid = np.asarray(pid, dtype=np.int64)
        self.pw = np.asarray(pw, dtype=np.float64)
        self.n = len(ptr) - 1          # annotated genes

    def _entropy_of(self, pos: np.ndarray) -> float:
        pid = self.pid[pos]
        pw = self.pw[pos]
        order = np.argsort(pid, kind="stable")
        pid_s, pw_s = pid[order], pw[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_s)) + 1))
        sums = np.add.reduceat(pw_s, starts)
        sums = sums[sums > 0]
        if sums.size == 0:
            return float("nan")
        p = sums / sums.sum()
        return float(math.exp(-(p * np.log(p)).sum()))

    def raw(self) -> float:
        """Un-rarefied fractional-weight plug-in value. Regression check only.

        See the module docstring: this is the historical sunburst statistic, not
        the reported one, because its support ceiling scales with annotation
        depth rather than with the number of genes.
        """
        if self.n == 0:
            return float("nan")
        return self._entropy_of(np.arange(self.pid.size))

    def _one_pathway_each(self, sel: np.ndarray, lens: np.ndarray, rng) -> float:
        """exp(H) with one pathway drawn uniformly per selected gene."""
        pos = self.ptr[sel] + (rng.random(sel.size) * lens[sel]).astype(np.int64)
        counts = np.bincount(self.pid[pos])
        p = counts[counts > 0] / sel.size
        return float(math.exp(-(p * np.log(p)).sum()))

    def rarefied(self, m: int | None, r: int, rng) -> float | None:
        """Mean exp(H) over ``r`` subsamples of ``m`` annotated genes.

        Each draw subsamples ``m`` genes *and* assigns each of them one of its
        pathways at random, so the support ceiling is exactly ``m`` for every
        method. ``m=None`` uses all the unit's genes and averages over the
        pathway assignment only.

        Returns None if the unit cannot supply ``m`` genes -- the caller counts
        those as dropped rather than silently comparing unequal samples.
        """
        if m is None:
            m = self.n
        if self.n < m or self.n == 0:
            return None
        lens = np.diff(self.ptr)
        every = np.arange(self.n) if m == self.n else None
        acc = 0.0
        for _ in range(r):
            sel = every if every is not None else rng.choice(
                self.n, size=m, replace=False)
            acc += self._one_pathway_each(sel, lens, rng)
        return acc / r


def effective_pathways(screen_batches, *, rarefy: bool = True,
                       seed: int = SEED, membership=None) -> dict:
    """EP at batch / screen / dataset scope for one method.

    ``screen_batches`` is ``list[screen][batch] -> list[gene]``, the shape every
    caller already has (``result.json`` ``steps[].acquired_batch``, or the JSONL
    batch lists).

    Returns rarefied ``ep_batch`` / ``ep_screen`` / ``ep_dataset`` plus the
    plug-in ``*_raw`` counterparts, the annotated fraction, and how many units
    were too small to rarefy.
    """
    membership = gmt_membership() if membership is None else membership
    pid_of: dict[str, int] = {}
    for ps in membership.values():
        for p in ps:
            if p not in pid_of:
                pid_of[p] = len(pid_of)

    # A batch the method never filled to its typical size is a run artifact --
    # the Shortfall column already reports that -- not a low-diversity batch, and
    # its handful of genes must not drag the EP-B reference count down. Measure
    # each method against *its own* median batch, then let rarefaction equalise
    # what survives: that is what makes a 32-pick Kimi batch and a 100-pick RF
    # batch comparable at all.
    all_picks = [len(b) for batches in screen_batches for b in batches if b]
    min_full = 0.9 * float(np.median(all_picks)) if all_picks else 0.0

    # One independent stream per scope rather than one shared generator. With a
    # shared generator the dataset draw inherits whatever state the batch and
    # screen draws left behind, so EP-D silently shifts if R_BATCH changes or a
    # method has a different number of batches -- and a caller that wants EP-D
    # alone (the sunburst hole, the heatmap row) cannot reproduce it, landing
    # ~0.1 off. Independent streams make each scope reproducible on its own.
    rng_b, rng_s, rng_d = (scope_rng(s, seed) for s in _SCOPES)
    n_picks = 0
    dropped = 0
    batch_elig = batch_kept = 0     # full batches offered / rarefied
    screen_elig = screen_kept = 0

    batch_r: list[float] = []      # per-screen means of rarefied batch EP
    batch_raw: list[float] = []
    screen_r: list[float] = []
    screen_raw: list[float] = []
    pooled: list[str] = []
    batch_ann: list[int] = []      # annotated genes per unit, for calibration
    batch_npicks: list[int] = []   # picks per batch, to spot degenerate batches
    screen_ann: list[int] = []

    for batches in screen_batches:
        per_screen_r: list[float] = []
        per_screen_raw: list[float] = []
        screen_genes: list[str] = []
        for b in batches:
            if not b:
                continue
            n_picks += len(b)
            screen_genes.extend(b)
            u = _Unit(b, membership, pid_of)
            batch_ann.append(u.n)
            batch_npicks.append(len(b))
            # Truncated batches still contribute their picks to EP-S and EP-D --
            # the genes are real -- but they are not a batch-scope unit.
            if len(b) < min_full or u.n < 2:
                continue
            batch_elig += 1
            per_screen_raw.append(u.raw())
            if rarefy:
                v = u.rarefied(M_BATCH, R_BATCH, rng_b)
                if v is None:
                    dropped += 1
                else:
                    batch_kept += 1
                    per_screen_r.append(v)
        if per_screen_raw:
            batch_raw.append(float(np.mean(per_screen_raw)))
        if per_screen_r:
            batch_r.append(float(np.mean(per_screen_r)))
        if screen_genes:
            pooled.extend(screen_genes)
            u = _Unit(screen_genes, membership, pid_of)
            screen_ann.append(u.n)
            if u.n >= 2:
                screen_elig += 1
                screen_raw.append(u.raw())
                if rarefy:
                    v = u.rarefied(M_SCREEN, R_SCREEN, rng_s)
                    if v is None:
                        dropped += 1
                    else:
                        screen_kept += 1
                        screen_r.append(v)

    ep_d_raw = ep_d = None
    n_ann = 0
    if pooled:
        u = _Unit(pooled, membership, pid_of)
        n_ann = u.n
        if u.n >= 2:
            ep_d_raw = u.raw()
            if rarefy:
                ep_d = u.rarefied(M_DATASET, R_DATASET, rng_d)
                if ep_d is None:
                    dropped += 1

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    # Annotated-count distribution per unit, so callers can verify the fixed
    # reference counts are actually suppliable before trusting the numbers.
    ba = np.asarray(batch_ann) if batch_ann else np.zeros(0)
    bp = np.asarray(batch_npicks) if batch_npicks else np.zeros(0)
    sa = np.asarray(screen_ann) if screen_ann else np.zeros(0)
    # Restrict to *full* batches: a batch the method never finished filling is a
    # run artifact, not a low-diversity batch, and should not set M_BATCH.
    full = ba[bp >= min_full] if bp.size else ba

    # Dropping the units that cannot supply M_* keeps every retained unit
    # comparable, but the survivors are the better-annotated ones -- so once the
    # drop rate is material the mean over survivors flatters the method. Past
    # that point report nothing rather than a biased number.
    ep_b = _mean(batch_r) if rarefy else _mean(batch_raw)
    ep_s = _mean(screen_r) if rarefy else _mean(screen_raw)
    b_ret = (batch_kept / batch_elig) if batch_elig else None
    s_ret = (screen_kept / screen_elig) if screen_elig else None
    if rarefy and b_ret is not None and b_ret < RETENTION:
        ep_b = None
    if rarefy and s_ret is not None and s_ret < RETENTION:
        ep_s = None

    return {
        "ep_batch_retention": b_ret,
        "ep_screen_retention": s_ret,
        "ep_batch_ann_min": int(ba.min()) if ba.size else None,
        "ep_batch_ann_p5": float(np.percentile(ba, 5)) if ba.size else None,
        "ep_batch_pick_med": float(np.median(bp)) if bp.size else None,
        "ep_batch_pick_p5": float(np.percentile(bp, 5)) if bp.size else None,
        "ep_batch_n_full": int(full.size),
        "ep_batch_full_min": int(full.min()) if full.size else None,
        "ep_batch_full_p5": float(np.percentile(full, 5)) if full.size else None,
        "ep_screen_ann_min": int(sa.min()) if sa.size else None,
        "ep_n_batches": int(ba.size),
        "ep_batch": ep_b,
        "ep_screen": ep_s,
        "ep_dataset": ep_d if rarefy else ep_d_raw,
        "ep_batch_raw": _mean(batch_raw),
        "ep_screen_raw": _mean(screen_raw),
        "ep_dataset_raw": ep_d_raw,
        "ep_annotated_frac": (n_ann / n_picks) if n_picks else None,
        "ep_n_annotated": n_ann,
        "ep_n_picks": n_picks,
        "ep_n_dropped": dropped,
    }
