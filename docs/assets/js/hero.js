// The landing page's headline recovery chart.
//
// A fixed cut of recovery.html: the same assets/data/recovery_mean.json, seven
// curves, no controls. It is the first thing on the page, so it has to be
// readable without being operated -- there is nothing to select, nothing to
// switch, and no modebar. The explorer one click below is where the other
// fifty-one methods, the per-screen curves and the other axes live.

(function () {
  const S = window.AssayLoop;

  //: The curves, keyed by the same string recovery.html and results.html join
  //: on. Order here does not matter -- the legend is sorted by where each curve
  //: ends, below. Kept in step with HEADLINE in recovery.js, so the
  //: explorer opens on the chart the reader just clicked through from.
  //:
  //: `label` is deliberately not the name the results table shows. The table
  //: has room to say which base model and which initialisation a row is
  //: ("Qwen3.6-27B (base) + SFT + GRPO (= AssayLLM)"); a legend on a teaser has
  //: room for the name of the thing ("AssayLLM"). `alias` is a name that the
  //: label replaced and that the reader still needs, shown on hover -- it is
  //: set on exactly one curve: the handoff, whose two halves are the whole
  //: point of the row. The recipes ("+ BPMF + GRPO", "+ SFT + GRPO") are not
  //: aliases in that sense; the table and the method page are where a recipe
  //: belongs.
  const SERIES = [
    { key: "Gemini-3.1-Pro - AssayLoop Handoff", label: "AssayLoop",
      alias: "Gemini-3.1-Pro → AssayFormer" },
    { key: "Transformer + GRPO (= AssayLoop)", label: "AssayFormer" },
    { key: "Qwen3.6-27B (base) + SFT + GRPO (= AssayLLM)", label: "AssayLLM" },
    { key: "Gemini-3.1-pro", label: "Gemini-3.1-pro" },
    { key: "ICBR-EF", label: "ICBR-EF" },
    { key: "BPMF", label: "BPMF" },
    // Draws the same grey as BPMF: both are "adaptive experimental design", and
    // giving this one a colour of its own would put it in a family it is not
    // in. The palette is the one thing every chart on the site shares.
    { key: "Haystacks", label: "Probability-of-Hit" },
  ];

  //: The chance line, same as the explorer's -- not a family colour, because it
  //: is not a method. build_data.py computes it exactly (a uniform draw of the
  //: same number of genes from the same 21,147-gene pool); see
  //: build_random_reference there for why that is also the EF = 1 line.
  const CHANCE_COLOR = "rgba(27,31,36,0.55)";

  function acquiredFraction(values, universeSize) {
    return values.map((n) => n / universeSize);
  }

  function chanceTrace(mean, universeSize) {
    const ref = mean.random_reference;
    if (!ref || !ref.frac_hits) return null;
    return {
      x: [0].concat(acquiredFraction(ref.n_acquired, universeSize)),
      y: [0].concat(ref.frac_hits),
      name: "Picking at random",
      type: "scatter",
      mode: "lines",
      line: { color: CHANCE_COLOR, width: 1.5, dash: "dash" },
      hovertemplate: "<b>Picking at random</b><br>%{x:.1%} of genes acquired" +
                     "<br>%{y:.1%} of hits recovered<extra></extra>",
    };
  }

  function traceFor(spec, meta, universeSize) {
    const color = S.familyColor(meta.family);
    const title = spec.alias ? `${spec.label} (${spec.alias})` : spec.label;
    return {
      // Both series start at the origin: before anything is assayed, no hits
      // have been found. The export starts at the end of round one, and a line
      // that begins a hundred genes in leaves the corner of the plot empty.
      x: [0].concat(acquiredFraction(meta.n_acquired, universeSize)),
      y: [0].concat(meta.frac_hits),
      name: spec.label,
      type: "scatter",
      mode: "lines",
      line: { color, width: 2.6 },
      hovertemplate: `<b>${title}</b><br>%{x:.1%} of genes acquired` +
                     "<br>%{y:.1%} of hits recovered<extra></extra>",
    };
  }

  async function init() {
    const node = document.getElementById("hero-plot");
    if (!node) return;
    const status = document.getElementById("hero-status");

    let mean, summary;
    try {
      [mean, summary] = await Promise.all([
        S.fetchJSON("assets/data/recovery_mean.json"),
        S.fetchJSON("assets/data/summary.json"),
      ]);
    } catch (err) {
      S.showError(node, err);
      return;
    }

    const universeSize = summary.universe_size;
    if (!Number.isFinite(universeSize) || universeSize <= 0) {
      S.showError(node, new Error("assets/data/summary.json has no valid universe_size"));
      return;
    }

    const chance = chanceTrace(mean, universeSize);
    const missing = SERIES.filter((s) => !mean.methods[s.key]).map((s) => s.key);
    if (!chance) missing.push("the chance line (random_reference)");
    // Legend order = the order the curves finish in, best first, so reading
    // down the legend and reading down the right-hand edge of the chart give
    // the same ranking. Same reasoning as the recovery page's hover card.
    const traces = SERIES
      .filter((s) => mean.methods[s.key])
      .map((s) => ({ spec: s, meta: mean.methods[s.key] }))
      .sort((a, b) => b.meta.frac_hits[b.meta.frac_hits.length - 1]
                    - a.meta.frac_hits[a.meta.frac_hits.length - 1])
      .map(({ spec, meta }) => traceFor(spec, meta, universeSize));
    // Appended after the sort, so it draws on top of the curves and reads last
    // in the legend: it is the floor they are measured against, not a rival.
    if (chance) traces.push(chance);

    // Same rule as the rest of the site: a curve that cannot be drawn is named,
    // not dropped from a chart that then looks complete. This one is the first
    // thing on the landing page, which makes a quiet gap worse, not better.
    if (missing.length && status) {
      status.hidden = false;
      status.textContent =
        `Not plotted — assets/data/recovery_mean.json has no curve for ` +
        `${missing.join(", ")}. Re-run docs/build_data.py.`;
    }
    if (!traces.length) return;

    const layout = S.mergeLayout({
      xaxis: { title: "Genes acquired (cumulative)", tickformat: ".0%",
               dtick: 0.01, rangemode: "tozero" },
      yaxis: { title: "Fraction of hits recovered", tickformat: ".0%",
               rangemode: "tozero" },
      legend: { orientation: "h", y: -0.24, x: 0 },
      hovermode: "closest",
      // r leaves room for the last x tick: the curves run to about 5% and a
      // narrower right margin cuts the label in half.
      margin: { l: 66, r: 40, t: 12, b: 96 },
      height: 400,
    });
    // No modebar: there are no options on this chart, and a zoom/pan toolbar
    // appearing on hover is an option.
    Plotly.newPlot(node, traces, layout,
                   Object.assign({}, S.PLOTLY_CONFIG, { displayModeBar: false }));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
