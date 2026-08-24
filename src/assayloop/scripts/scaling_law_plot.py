"""Plot the v2 scaling law: NVR@10 vs data, and NVR@10 vs model size.

Reads ``scaling2_eval.json`` from each ``scl2-*`` run (written by
``scaling_law_eval``) and produces two clean figures:

  1. ``scaling_data_nvr_at_10.png`` - NVR@10 vs #training screens (log x),
     SFT and RL series, mean +/- SEM over seeds (data sweep, tier L).
  2. ``scaling_model_nvr_at_10.png`` - NVR@10 vs transformer parameters (log x),
     SFT and RL series (model sweep, all data).

Also emits budget→recall and NVR@k line/heatmap figures. Every figure is saved
as both PNG (preview) and PDF (paper). For each metric, the data and model
figures share a y-axis so they're directly comparable.

Usage::

    uv run python -m assayloop.scripts.scaling_law_plot
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from assayloop.scripts._figure_io import save_figure
from assayloop import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scaling_plot")

RANKERS_DIR = config.OUTPUT_PATH / "rankers"
EVAL_FILE = "scaling2_eval.json"
TIER_ORDER = ["XS", "S", "M", "L", "XL"]

SFT_STYLE = dict(marker="o", ms=5, lw=1.8, color="#1f77b4", label="SFT")
RL_STYLE = dict(marker="s", ms=5, lw=1.8, ls="--", color="#d62728", label="+ GRPO (RL)")


def _save(fig, fpath):
    """Save a figure as both PNG (preview) and PDF (paper), then close it."""
    save_figure(fig, fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _phase_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    return [
        Line2D([0], [0], color="gray", lw=2, ls="-", label="SFT"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="+ GRPO (RL)"),
        Patch(facecolor="gray", alpha=0.15, label="RL gain over SFT"),
    ]


def _count_params(arch: dict) -> int:
    """Transformer parameter count for a run's arch (excludes any BERT encoder;
    with --no-description there is none)."""
    from assayloop.amortized.model import RankerConfig, RankerNet
    net = RankerNet(RankerConfig(**arch))
    total = 0
    for name, p in net.named_parameters():
        if name.startswith("bert") or ".bert." in name:
            continue
        total += p.numel()
    return int(total)


def _load_records(rankers: Path, prefix: str) -> list[dict]:
    recs = []
    for d in sorted(rankers.glob(f"{prefix}*")):
        ev = d / EVAL_FILE
        if not ev.exists():
            continue
        try:
            m = json.loads(ev.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "nvr_at_10" not in m:
            continue
        recs.append(m)
    return recs


def _agg(records, key_fn, metric):
    """metric per key -> list of per-seed values."""
    agg = defaultdict(list)
    for r in records:
        if metric in r and r[metric] is not None:
            agg[key_fn(r)].append(r[metric])
    return agg


def _series(agg, xs):
    """Return (x, mean, sem) for the given ordered x keys present in agg."""
    X, Y, E = [], [], []
    for x in xs:
        vals = agg.get(x)
        if not vals:
            continue
        X.append(x)
        Y.append(float(np.mean(vals)))
        E.append(float(np.std(vals) / max(np.sqrt(len(vals)), 1)))
    return X, Y, E


def _plot_pair(sft_agg, rl_agg, x_order, xlabel, ylabel, title, fpath,
               xscale="log", xbase=10, invert_y=False, xticklabels=None,
               ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for agg, style in [(sft_agg, SFT_STYLE), (rl_agg, RL_STYLE)]:
        X, Y, E = _series(agg, x_order)
        if X:
            ax.errorbar(X, Y, yerr=E, capsize=3, **style)
    if xscale == "log":
        ax.set_xscale("log", base=xbase)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(x_order)
        ax.set_xticklabels(xticklabels)
    if ylim is not None:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()      # flips the (already-set) ylim so lower = better up top
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, fpath)


def _mean_by(records, series_of, value_key):
    """(series, x_key) -> mean over seeds, for a per-record {x_key: value} dict."""
    cells = defaultdict(list)
    for r in records:
        s = series_of(r)
        if s is None:
            continue
        for k, v in r.get(value_key, {}).items():
            cells[(s, int(k))].append(v)
    return {key: float(np.mean(v)) for key, v in cells.items()}


def _metric_heatmap(records, x_of, x_order, x_labels, value_key, row_label,
                    clabel, title, fpath, cmap="viridis"):
    """Heatmap: rows = the metric's x-keys (k rounds / recall %), cols = scale
    axis (x_order), cell = mean over seeds. ``records`` pre-filtered to one
    (sweep, phase)."""
    rows = sorted({int(k) for r in records for k in r.get(value_key, {})})
    if not rows or not x_order:
        return
    means = _mean_by(records, x_of, value_key)
    M = np.full((len(rows), len(x_order)), np.nan)
    for ri, rk in enumerate(rows):
        for xi, x in enumerate(x_order):
            if (x, rk) in means:
                M[ri, xi] = means[(x, rk)]
    if np.all(np.isnan(M)):
        return
    lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
    thr = lo + 0.55 * (hi - lo)
    fig, ax = plt.subplots(figsize=(1.15 * len(x_order) + 3, 0.42 * len(rows) + 2.5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(k) for k in rows], fontsize=8)
    ax.set_ylabel(row_label)
    ax.set_title(title)
    for ri in range(len(rows)):
        for xi in range(len(x_order)):
            if not np.isnan(M[ri, xi]):
                ax.text(xi, ri, f"{M[ri, xi]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if M[ri, xi] < thr else "black")
    fig.colorbar(im, ax=ax, label=clabel)
    fig.tight_layout()
    _save(fig, fpath)


def _draw_metric_lines(ax, sft_recs, rl_recs, series_of, series_order,
                       series_labels, value_key, xlabel, ylabel, xlog=True,
                       ylim=None, xlim=None):
    """Draw the SFT/RL/gain series into ``ax`` (no legend, no save). Returns
    (scale_handles, has_rl, drew)."""
    from matplotlib.lines import Line2D
    sft = _mean_by(sft_recs, series_of, value_key)
    rl = _mean_by(rl_recs, series_of, value_key)
    xkeys = sorted({k for (_, k) in list(sft) + list(rl)})
    if not xkeys or not series_order:
        return [], False, False
    cmap = plt.get_cmap("viridis")
    n = len(series_order)
    drew = has_rl = False
    for si, s in enumerate(series_order):
        color = cmap(si / max(n - 1, 1))
        xs = [k for k in xkeys if (s, k) in sft]
        if xs:
            ax.plot(xs, [sft[(s, k)] for k in xs], marker="o", ms=3, lw=1.4,
                    ls="-", color=color)
            drew = True
        xr = [k for k in xkeys if (s, k) in rl]
        if xr:
            ax.plot(xr, [rl[(s, k)] for k in xr], marker="s", ms=3, lw=1.4,
                    ls="--", color=color)
            drew = has_rl = True
        common = [k for k in xkeys if (s, k) in sft and (s, k) in rl]
        if common:
            ax.fill_between(common, [sft[(s, k)] for k in common],
                            [rl[(s, k)] for k in common], color=color,
                            alpha=0.15, linewidth=0)
    if not drew:
        return [], False, False
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.25)
    handles = [Line2D([0], [0], color=cmap(si / max(n - 1, 1)), lw=2,
                      label=series_labels[si]) for si in range(n)]
    return handles, has_rl, True


def _metric_lines(sft_recs, rl_recs, series_of, series_order, series_labels,
                  value_key, xlabel, ylabel, title, fpath, xlog=True,
                  ylim=None, xlim=None):
    """Standalone single-panel line figure (SFT solid, RL dashed, shaded gain)."""
    fig, ax = plt.subplots(figsize=(7.8, 5))
    handles, has_rl, drew = _draw_metric_lines(
        ax, sft_recs, rl_recs, series_of, series_order, series_labels, value_key,
        xlabel, ylabel, xlog=xlog, ylim=ylim, xlim=xlim)
    if not drew:
        plt.close(fig)
        return
    ax.set_title(title)
    leg1 = ax.legend(handles=handles, fontsize=7, loc="best", title="scale")
    ax.add_artist(leg1)
    if has_rl:
        ax.legend(handles=_phase_handles(), fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, fpath)


def _combined_2x2(rows, yl_nvrk, yl_brec, fpath):
    """2x2 line figure: columns = [NVR@k, budget->recall], rows = the given
    (row_label, sft, rl, series_of, series_order, series_labels) tuples (data on
    top, model on bottom). One shared phase legend + a per-row scale legend."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    row_handles = [None, None]
    has_rl = False
    for ri, (rlabel, s, r, sof, order, lbls) in enumerate(rows):
        h, hr, _ = _draw_metric_lines(
            axes[ri][0], s, r, sof, order, lbls, "nvr_at_k",
            "Acquisition Steps k  (budget = 100·k genes)",
            "%s\n\nEF @ k" % rlabel, xlog=True, ylim=yl_nvrk)
        _draw_metric_lines(
            axes[ri][1], s, r, sof, order, lbls, "budget_to_recall",
            "recall threshold (%)", "budget (frac. of f2 universe, lower = better)",
            xlog=False, ylim=yl_brec)
        row_handles[ri] = h
        has_rl = has_rl or hr
    axes[0][0].set_title("EF@k", fontsize=12, fontweight="bold")
    axes[0][1].set_title("budget → recall", fontsize=12, fontweight="bold")
    # per-row scale legend (right of the right panel); shared phase legend below
    for ri in range(2):
        if row_handles[ri]:
            axes[ri][1].legend(handles=row_handles[ri], fontsize=9,
                               loc="center left", bbox_to_anchor=(1.02, 0.5),
                               title="scale", frameon=False)
    if has_rl:
        fig.legend(handles=_phase_handles(), fontsize=11, loc="lower center",
                   ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, fpath)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rankers-dir", default=str(RANKERS_DIR))
    ap.add_argument("--prefix", default="scl2-")
    ap.add_argument("--out-dir",
                    default=str(config.OUTPUT_PATH / "analysis" / "scaling_law_v2"))
    args = ap.parse_args()

    rankers = Path(args.rankers_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(rankers, args.prefix)
    if not records:
        log.warning("No %s records found under %s", EVAL_FILE, rankers)
        return
    data_recs = [r for r in records if r.get("sweep") == "data"]
    model_recs = [r for r in records if r.get("sweep") == "model"]
    log.info("Loaded %d eval records (%d data, %d model)",
             len(records), len(data_recs), len(model_recs))

    data_sft = [r for r in data_recs if not r["is_rl"]]
    data_rl = [r for r in data_recs if r["is_rl"]]
    model_sft = [r for r in model_recs if not r["is_rl"]]
    model_rl = [r for r in model_recs if r["is_rl"]]

    # param count per model tier (from each run's config.json arch)
    tier_params: dict[str, int] = {}
    for r in model_recs:
        if r["tier"] in tier_params:
            continue
        cfg_path = rankers / r["run_name"] / "config.json"
        if cfg_path.exists():
            arch = json.loads(cfg_path.read_text()).get("arch")
            if arch:
                try:
                    tier_params[r["tier"]] = _count_params(arch)
                except Exception as e:  # noqa: BLE001
                    log.warning("param count failed for %s: %s", r["tier"], e)

    size_x = lambda r: r["train_size"]
    param_x = lambda r: tier_params.get(r["tier"])
    tier_x = lambda r: r["tier"]

    # Shared y-limits per metric, so the data and model figures for the SAME
    # metric use the same y-axis (comparable side by side in the paper).
    def _pair_ylim(metric, floor0=False):
        vals = []
        for recs, kf in [(data_sft, size_x), (data_rl, size_x),
                         (model_sft, param_x), (model_rl, param_x)]:
            for lst in _agg(recs, kf, metric).values():
                if lst:
                    m = float(np.mean(lst)); s = float(np.std(lst) / max(np.sqrt(len(lst)), 1))
                    vals += [m - s, m + s]
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        if floor0:
            lo = min(lo, 0.0)
        pad = (hi - lo) * 0.06 or 0.1
        return (lo - pad, hi + pad)

    def _line_ylim(vk, floor0=False):
        vals = []
        for recs, kf in [(data_sft, size_x), (data_rl, size_x),
                         (model_sft, tier_x), (model_rl, tier_x)]:
            vals += list(_mean_by(recs, kf, vk).values())
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        if floor0:
            lo = min(lo, 0.0)
        pad = (hi - lo) * 0.06 or 0.1
        return (lo - pad, hi + pad)

    yl_nvr10 = _pair_ylim("nvr_at_10")
    yl_b50 = _pair_ylim("budget_to_50_recall", floor0=True)
    yl_nvrk = _line_ylim("nvr_at_k")
    yl_brec = _line_ylim("budget_to_recall", floor0=True)

    # ---- Figure 1: data scaling (x = #screens) ----
    if data_recs:
        sizes = sorted({r["train_size"] for r in data_recs})
        size_lbls = [str(s) for s in sizes]
        for metric, ylabel, fname, inv, yl in [
            ("nvr_at_10", "NVR @ 1000-gene budget", "scaling_data_nvr_at_10.png", False, yl_nvr10),
            ("budget_to_50_recall", "Fraction of f2 universe to 50% recall",
             "scaling_data_budget_to_50.png", True, yl_b50),
        ]:
            sft = _agg(data_sft, size_x, metric)
            rl = _agg(data_rl, size_x, metric)
            if not sft and not rl:
                continue
            _plot_pair(sft, rl, sizes, "Training screens", ylabel,
                       "Data scaling (production model, f2 universe)",
                       out_dir / fname, xscale="log", xbase=10,
                       invert_y=inv, xticklabels=size_lbls, ylim=yl)

        # ---- NVR@k and budget→recall vs scale (heatmaps + combined lines) ----
        for phase, recs in [("sft", data_sft), ("rl", data_rl)]:
            _metric_heatmap(recs, size_x, sizes, size_lbls, "nvr_at_k",
                            "AL rounds k  (budget=100·k)", "NVR @ k",
                            f"Data: NVR@k ({phase.upper()}, f2 universe)",
                            out_dir / f"scaling_data_nvr_heatmap_{phase}.png")
            _metric_heatmap(recs, size_x, sizes, size_lbls, "budget_to_recall",
                            "recall threshold (%)", "budget (frac. of f2 universe)",
                            f"Data: budget→recall ({phase.upper()}, f2 universe)",
                            out_dir / f"scaling_data_budget_heatmap_{phase}.png",
                            cmap="viridis_r")
        _metric_lines(data_sft, data_rl, size_x, sizes, [f"{s} scr" for s in sizes],
                      "nvr_at_k", "AL rounds k  (budget = 100·k genes)", "NVR @ k",
                      "Data scaling: NVR@k (SFT vs +GRPO, f2 universe)",
                      out_dir / "scaling_data_nvr_vs_k.png", ylim=yl_nvrk)
        _metric_lines(data_sft, data_rl, size_x, sizes, [f"{s} scr" for s in sizes],
                      "budget_to_recall", "recall threshold (%)",
                      "budget (fraction of f2 universe, lower = better)",
                      "Data scaling: budget→recall (SFT vs +GRPO, f2 universe)",
                      out_dir / "scaling_data_budget_vs_recall.png", xlog=False,
                      ylim=yl_brec)

    # ---- Figure 2: model-size scaling (x = transformer params) ----
    if model_recs and tier_params:
        present_tiers = [t for t in TIER_ORDER if t in tier_params]
        xs = [tier_params[t] for t in present_tiers]
        hm_labels = [f"{t}\n{tier_params[t]/1e6:.2f}M" for t in present_tiers]
        line_labels = [f"{t} ({tier_params[t]/1e6:.2f}M)" for t in present_tiers]
        for metric, ylabel, fname, inv, yl in [
            ("nvr_at_10", "NVR @ 1000-gene budget", "scaling_model_nvr_at_10.png", False, yl_nvr10),
            ("budget_to_50_recall", "Fraction of f2 universe to 50% recall",
             "scaling_model_budget_to_50.png", True, yl_b50),
        ]:
            sft = _agg(model_sft, param_x, metric)
            rl = _agg(model_rl, param_x, metric)
            if not sft and not rl:
                continue
            _plot_pair(sft, rl, xs, "Transformer parameters", ylabel,
                       "Model-size scaling (all data, scaled K, f2 universe)",
                       out_dir / fname, xscale="log", xbase=10, invert_y=inv,
                       xticklabels=hm_labels, ylim=yl)

        for phase, recs in [("sft", model_sft), ("rl", model_rl)]:
            _metric_heatmap(recs, param_x, xs, hm_labels, "nvr_at_k",
                            "AL rounds k  (budget=100·k)", "NVR @ k",
                            f"Model: NVR@k ({phase.upper()}, f2 universe)",
                            out_dir / f"scaling_model_nvr_heatmap_{phase}.png")
            _metric_heatmap(recs, param_x, xs, hm_labels, "budget_to_recall",
                            "recall threshold (%)", "budget (frac. of f2 universe)",
                            f"Model: budget→recall ({phase.upper()}, f2 universe)",
                            out_dir / f"scaling_model_budget_heatmap_{phase}.png",
                            cmap="viridis_r")
        _metric_lines(model_sft, model_rl, tier_x, present_tiers, line_labels,
                      "nvr_at_k", "AL rounds k  (budget = 100·k genes)", "NVR @ k",
                      "Model-size scaling: NVR@k (SFT vs +GRPO, f2 universe)",
                      out_dir / "scaling_model_nvr_vs_k.png", ylim=yl_nvrk)
        _metric_lines(model_sft, model_rl, tier_x, present_tiers, line_labels,
                      "budget_to_recall", "recall threshold (%)",
                      "budget (fraction of f2 universe, lower = better)",
                      "Model-size scaling: budget→recall (SFT vs +GRPO, f2 universe)",
                      out_dir / "scaling_model_budget_vs_recall.png", xlog=False,
                      ylim=yl_brec)

    # ---- combined 2x2: [NVR@k | budget→recall] x [data | model] ----
    if data_recs and model_recs and tier_params:
        sizes = sorted({r["train_size"] for r in data_recs})
        present_tiers = [t for t in TIER_ORDER if t in tier_params]
        rows = [
            ("Data scaling", data_sft, data_rl, size_x, sizes,
             [str(s) for s in sizes]),
            ("Model scaling", model_sft, model_rl, tier_x, present_tiers,
             [f"{t} ({tier_params[t] / 1e6:.2f}M)" for t in present_tiers]),
        ]
        _combined_2x2(rows, yl_nvrk, yl_brec,
                      out_dir / "scaling_combined_lines.png")

    # ---- dump the aggregated table for the paper ----
    summary = {"data": {}, "model": {}}
    for r in data_recs:
        k = f"d{r['train_size']}-{'rl' if r['is_rl'] else 'sft'}"
        summary["data"].setdefault(k, []).append(r["nvr_at_10"])
    for r in model_recs:
        k = f"{r['tier']}-{'rl' if r['is_rl'] else 'sft'}"
        summary["model"].setdefault(k, []).append(r["nvr_at_10"])
    agg_summary = {sweep: {k: {"nvr_at_10_mean": float(np.mean(v)),
                              "nvr_at_10_sem": float(np.std(v) / max(np.sqrt(len(v)), 1)),
                              "n_seeds": len(v)}
                          for k, v in d.items()}
                   for sweep, d in summary.items()}
    (out_dir / "scaling_summary.json").write_text(json.dumps(agg_summary, indent=2))
    log.info("Wrote %s", out_dir / "scaling_summary.json")


if __name__ == "__main__":
    main()
