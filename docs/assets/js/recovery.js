// The recovery-curve explorer.
//
// Two data files: recovery_mean.json (59 methods x 10 steps, with SEM) and
// recovery_by_screen.json (the same, per test screen). Both come from
// export_recovery_curves.py by way of docs/build_data.py. Methods are keyed by
// the same string the results table uses, so the two pages agree on names.

(function () {
  const S = window.AssayLoop;

  // Y-axis options. `mean` is the key in recovery_mean.json; `screen` is the
  // key in recovery_by_screen.json, which carries fewer columns -- a metric
  // with no per-screen key simply disables the per-screen view rather than
  // plotting the mean and calling it a screen.
  // `chance` is the one-line reading of the dashed reference line on that
  // axis. The line itself is computed in build_data.py, not here -- see
  // build_random_reference, whose docstring is where the arithmetic and the
  // EF-1 equivalence are argued.
  const METRICS = {
    frac_hits: {
      mean: "frac_hits", screen: "frac_hits", sem: "sem_frac_hits",
      label: "Fraction of hits recovered", tickformat: ".0%", digits: 3,
      hover: (v) => (v * 100).toFixed(1) + "%",
      chance: "a uniform draw of the same number of genes from the same pool, " +
              "which scores EF 1 by definition. Anything above it will have an EF &gt; 1.",
    },
    cum_hits: {
      mean: "cum_hits", screen: "cum_hits",
      label: "Cumulative hits found", tickformat: ",d", digits: 1,
      hover: (v) => v.toFixed(1),
      chance: "the hits a uniform draw of the same size would be expected to " +
              "turn up. It rises with the screen's hit count, which is why it " +
              "sits so much lower on this axis for some screens than others.",
    },
    // No out_of_universe axis. It is zero for every method whose picks were
    // parsed against the gene universe, which is all but the three raw-decode
    // Qwen rows -- an axis that draws a flat line at zero 55 times out of 58
    // teaches the reader nothing. The quantity is still in the data and still
    // in the Shortfall column on the results table.
    in_universe_not_lib: {
      mean: "in_universe_not_lib", screen: null,
      label: "Cumulative picks outside the screen's library", tickformat: ",d", digits: 1,
      hover: (v) => v.toFixed(1),
      chance: "where a uniform draw from the shared pool would land, since " +
              "roughly a tenth of that pool is outside any one screen's " +
              "library. On this axis lower is better, so a curve above the " +
              "line is wasting more budget than picking at random would.",
    },
  };

  // The same seven curves the landing page opens with -- see assets/js/hero.js,
  // which this list is kept in step with so that arriving here from the teaser
  // shows the chart the reader just clicked on, with the controls added rather
  // than the methods swapped. kNN is not among them: it is dominated by BPMF at
  // every round, and one adaptive-design reference point is enough on a preset
  // whose job is the headline comparison.
  const HEADLINE = [
    "Gemini-3.1-Pro - AssayLoop Handoff",
    "Transformer + GRPO (= AssayLoop)",
    "Qwen3.6-27B (base) + SFT + GRPO (= AssayLLM)",
    "Gemini-3.1-pro",
    "ICBR-EF",
    "BPMF",
    "Haystacks",
  ];

  const state = { metric: "frac_hits", view: "mean", screen: null,
                  band: "none", family: "", selected: [] };
  let MEAN = null, BY_SCREEN = null;

  //: Methods that proposed substantially more genes than their budget allowed,
  //: keyed by method, with the ratio build_data.py computed. They are plotted
  //: like everything else -- see the x-axis note below -- and listed on the
  //: page so the long tail on their curve is legible as what it is.
  let OVER_BUDGET = {};

  function plottable() {
    return Object.entries(MEAN.methods);
  }

  function methodEntries() {
    return plottable()
      .filter(([, m]) => !state.family || m.family === state.family);
  }

  /** The x series for a method: genes it actually assayed, cumulative.
   *
   * Not the requested budget, and the two come apart in both directions. A
   * method can answer a 100-gene round with fewer usable genes than that and
   * end the tenth round well short of 1,000 -- the finetuned Qwen rows all do,
   * finishing between 760 and 830 -- and the paper's own accounting agrees that
   * this is where they belong: EF divides by the genes acquired and charges the
   * unfilled remainder, and nAUC integrates against the fraction of the library
   * *effectively* queried. Every curve is at what it spent, so a vertical slice
   * is the same amount of assay for every method on it.
   */
  function xOf(key, meta) {
    if (state.view === "screen") {
      const perScreen = (BY_SCREEN.methods[key] || {})[state.screen];
      return perScreen && perScreen.n_acquired;
    }
    return meta.n_acquired;
  }

  function rgba(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
  }

  /** Prepend the origin to a series.
   *
   * Every curve on this page passes through (0, 0) -- before anything has been
   * assayed no hits have been found, and nothing has been picked outside the
   * library either. The export starts at the end of round one, so the drawn
   * line used to begin a hundred genes in and leave the corner of the plot
   * empty, which reads as a chart that has been cropped. `rangemode: "tozero"`
   * on the axes is not enough: it pads the range and leaves the line short.
   *
   * Note that this shifts every point one index to the right, which the hover
   * card compensates for -- see showTooltip. */
  function fromOrigin(series) {
    return [0].concat(series);
  }

  //: Line styles, cycled within a family. Colour is the family, which is what
  //: the reader needs first and what every other chart on the site agrees on --
  //: but it means two methods from the same family draw the same line, and the
  //: headline preset selects two "adaptive experimental design" rows. The dash
  //: separates them without claiming they are in different families.
  const DASHES = ["solid", "dot", "dash", "longdash", "dashdot"];

  //: Deliberately not a family colour. The chance line is not a method and
  //: should not read as one -- it is the axis the others are measured against.
  const CHANCE_COLOR = "rgba(27,31,36,0.55)";
  const CHANCE_LABEL = "Picking at random";

  /** The chance line for the current view, or null if this build has none.
   *
   * build_data.py computes it exactly (see build_random_reference): a uniform
   * draw of n genes from the shared candidate pool. The per-screen view uses
   * that screen's own reference, which differs from the mean's wherever its
   * library or hit count does. */
  function chanceSeries() {
    const metric = METRICS[state.metric];
    if (state.view === "screen") {
      const ref = (BY_SCREEN.random_reference || {})[state.screen];
      return (ref && metric.screen && ref[metric.screen]) ? ref : null;
    }
    const ref = MEAN.random_reference;
    return (ref && ref[metric.mean]) ? ref : null;
  }

  function chanceTrace() {
    const metric = METRICS[state.metric];
    const ref = chanceSeries();
    if (!ref) return null;
    const key = state.view === "screen" ? metric.screen : metric.mean;
    return {
      x: fromOrigin(ref.n_acquired), y: fromOrigin(ref[key]),
      name: CHANCE_LABEL, type: "scatter", mode: "lines",
      line: { color: CHANCE_COLOR, width: 1.5, dash: "dash" },
      hoverinfo: "none",
    };
  }

  function buildTraces() {
    const metric = METRICS[state.metric];
    const traces = [];
    const nthInFamily = {};

    for (const key of state.selected) {
      const meta = MEAN.methods[key];
      if (!meta) continue;
      const color = S.familyColor(meta.family);
      const n = nthInFamily[meta.family] = (nthInFamily[meta.family] || 0) + 1;
      const dash = DASHES[(n - 1) % DASHES.length];
      // Same minus-sign fix as the table: "- hit labels" is an operator.
      const label = S.methodLabel(meta.name);
      const x = xOf(key, meta);
      if (!x) continue;
      const x0 = fromOrigin(x);

      if (state.view === "screen") {
        const perScreen = (BY_SCREEN.methods[key] || {})[state.screen];
        if (!perScreen || !metric.screen) continue;
        traces.push({
          x: x0, y: fromOrigin(perScreen[metric.screen]),
          name: label, type: "scatter",
          mode: "lines+markers", line: { color, width: 2.5, dash },
          marker: { color, size: 6 },
          // See buildTooltip: hover is drawn by hand so the entries can be
          // ordered by value. "none" still fires plotly_hover; "skip" does not.
          hoverinfo: "none",
        });
        continue;
      }

      const y = meta[metric.mean];
      if (state.band === "sem" && metric.sem && meta[metric.sem]) {
        const sem = meta[metric.sem];
        // The band pinches shut at the origin, which is right: the spread
        // across screens at zero genes assayed is zero.
        const upper = fromOrigin(y.map((v, i) => v + sem[i]));
        const lower = fromOrigin(y.map((v, i) => v - sem[i]));
        traces.push({
          x: x0.concat(x0.slice().reverse()),
          y: upper.concat(lower.reverse()),
          fill: "toself", fillcolor: rgba(color, 0.13),
          line: { color: "transparent" }, hoverinfo: "skip",
          showlegend: false, type: "scatter",
        });
      }
      traces.push({
        x: x0, y: fromOrigin(y), name: label, type: "scatter",
        mode: "lines+markers",
        line: { color, width: 2.5, dash }, marker: { color, size: 6 },
        hoverinfo: "none",
      });
    }
    // Last, so it draws over the bands and sits at the end of the legend: it
    // is the reference the curves are read against, not one of them.
    const chance = chanceTrace();
    if (traces.length && chance) traces.push(chance);
    return traces;
  }

  // ------------------------------------------------------------- hover label
  //
  // Plotly's "x unified" hover lists the traces in trace order, which is the
  // order the methods were selected in -- so at any x the labels are in a
  // different order from the curves they point at, and reading the chart means
  // matching colours by eye. This draws the box instead, sorted by the value at
  // the hovered x, so the top entry is the top curve.

  let tip = null;

  function tooltipNode(plot) {
    // isConnected, not just null: Plotly.purge empties the graph div, which
    // takes the tooltip with it and leaves this holding a detached node.
    if (!tip || !tip.isConnected) {
      tip = document.createElement("div");
      tip.className = "hover-card";
      tip.hidden = true;
      plot.appendChild(tip);
    }
    return tip;
  }

  /** The plotted series, as {label, color, value, x} at the hovered index.
   *
   * The index is the acquisition round, which every method shares; the x is
   * that method's own genes-assayed total at it, which they do not. Both go
   * into the card, because "round 10" and "1,000 genes" stopped being the same
   * statement the moment the axis became what was actually spent. */
  function seriesAt(index) {
    const metric = METRICS[state.metric];
    const out = [];
    for (const key of state.selected) {
      const meta = MEAN.methods[key];
      if (!meta) continue;
      const y = state.view === "screen"
        ? ((BY_SCREEN.methods[key] || {})[state.screen] || {})[metric.screen]
        : meta[metric.mean];
      if (!y || y[index] === undefined || y[index] === null) continue;
      const x = xOf(key, meta);
      out.push({ label: S.methodLabel(meta.name),
                 color: S.familyColor(meta.family), value: y[index],
                 x: x && x[index] });
    }
    // Sorted in with the methods rather than pinned to the bottom of the card,
    // so the reader can see at a glance which curves are above chance here --
    // which is not the same set at round 1 as at round 10.
    const chance = chanceSeries();
    if (chance) {
      const key = state.view === "screen"
        ? METRICS[state.metric].screen : METRICS[state.metric].mean;
      out.push({ label: CHANCE_LABEL, color: CHANCE_COLOR,
                 value: chance[key][index], x: chance.n_acquired[index] });
    }
    return out.sort((a, b) => b.value - a.value);
  }

  function showTooltip(plot, data) {
    const point = data.points && data.points[0];
    if (!point || !data.event) return;
    const metric = METRICS[state.metric];
    // Index 0 is the origin fromOrigin() prepended, not a round. There is
    // nothing to say about it -- no method has assayed anything yet -- so the
    // card stays hidden there rather than reporting a round zero.
    const index = point.pointIndex - 1;
    if (index < 0) return;
    const rows = seriesAt(index);
    if (!rows.length) return;

    const node = tooltipNode(plot);
    const round = index + 1;
    node.innerHTML =
      `<div class="hover-x">Round ${round} of ${MEAN.steps.length}</div>` +
      rows.map((r) =>
        `<div class="hover-row">` +
        `<span class="hover-swatch" style="background:${r.color}"></span>` +
        `<span class="hover-name">${r.label}</span>` +
        `<span class="hover-sub">${r.x === undefined ? "" :
           Math.round(r.x).toLocaleString() + " assayed"}</span>` +
        `<span class="hover-value">${metric.hover(r.value)}</span></div>`
      ).join("");
    node.hidden = false;

    // Placed relative to the plot box, and flipped to the left of the cursor
    // when it would otherwise run off the right edge.
    const box = plot.getBoundingClientRect();
    const x = data.event.clientX - box.left;
    const y = data.event.clientY - box.top;
    const flip = x + node.offsetWidth + 24 > box.width;
    node.style.left = `${flip ? x - node.offsetWidth - 16 : x + 16}px`;
    node.style.top =
      `${Math.max(8, Math.min(y - node.offsetHeight / 2,
                              box.height - node.offsetHeight - 8))}px`;
  }

  function hideTooltip() {
    if (tip) tip.hidden = true;
  }

  function render() {
    const metric = METRICS[state.metric];
    const node = document.getElementById("plot");
    const caption = document.getElementById("caption");

    if (state.view === "screen" && !metric.screen) {
      Plotly.purge(node);
      node.innerHTML = "";
      caption.innerHTML =
        `<span class="label">Not available per screen.</span> ` +
        `"${metric.label}" is only exported as a mean over screens. ` +
        `Switch the view back to "Mean over all screens", or pick another Y axis.`;
      return;
    }

    const traces = buildTraces();
    if (!traces.length) {
      Plotly.purge(node);
      node.innerHTML = "";
      caption.innerHTML = `<span class="label">Nothing selected.</span> ` +
        `Pick one or more methods above, or use a preset.`;
      return;
    }

    const layout = S.mergeLayout({
      // Automatic ticks, not one per round: the curves no longer share an x,
      // so a tick array taken from the first selected method would label the
      // grid with one method's spending.
      xaxis: { title: "Genes assayed (cumulative)", rangemode: "tozero" },
      yaxis: { title: metric.label, tickformat: metric.tickformat,
               rangemode: "tozero" },
      legend: { orientation: "h", y: -0.18, x: 0 },
      // Not "x unified": that box lists methods in trace order. See the hover
      // label section -- these events drive a tooltip sorted by value instead.
      hovermode: "x",
      margin: { l: 72, r: 24, t: 20, b: 110 },
    });
    Plotly.react(node, traces, layout, S.PLOTLY_CONFIG);
    hideTooltip();
    node.removeAllListeners && node.removeAllListeners("plotly_hover");
    node.removeAllListeners && node.removeAllListeners("plotly_unhover");
    node.on("plotly_hover", (data) => showTooltip(node, data));
    node.on("plotly_unhover", hideTooltip);

    if (state.view === "screen") {
      const t = BY_SCREEN.screen_totals[state.screen];
      caption.innerHTML =
        `<span class="label">Screen ${state.screen}.</span> ` +
        `${t.total_hits} hits in a library of ${t.library_size.toLocaleString()} genes, ` +
        `against a shared candidate universe of ${t.universe_size.toLocaleString()}. ` +
        `A single screen, not an average &mdash; the curves are correspondingly jagged. ` +
        chanceNote();
    } else {
      const n = MEAN.methods[state.selected[0]].n_screens;
      caption.innerHTML =
        `<span class="label">Mean over ${n} held-out screens.</span> ` +
        (state.band === "sem"
          ? "Shaded bands are &plusmn;1 standard error of the mean across screens. "
          : "") +
        `Every method picks from the same full-genome candidate pool. ` +
        chanceNote();
    }
  }

  /** One sentence about the dashed line -- or about its absence.
   *
   * Same rule as everywhere else on the site: a reference the chart cannot
   * draw is named, not silently left off a plot that then looks complete. */
  function chanceNote() {
    if (chanceSeries()) {
      return `<span class="label">The dashed line is chance:</span> ` +
             METRICS[state.metric].chance;
    }
    return `<em>No chance line: this build of ` +
           `${state.view === "screen" ? "recovery_by_screen.json"
                                      : "recovery_mean.json"} ` +
           `has no random_reference. Re-run docs/build_data.py.</em>`;
  }

  function syncMethodList() {
    const sel = document.getElementById("methods");
    sel.innerHTML = "";
    const entries = methodEntries().sort((a, b) => {
      const fa = S.FAMILIES.findIndex((f) => f.key === a[1].family);
      const fb = S.FAMILIES.findIndex((f) => f.key === b[1].family);
      return fa - fb || a[1].name.localeCompare(b[1].name);
    });
    let group = null, optgroup = null;
    for (const [key, meta] of entries) {
      if (meta.family !== group) {
        group = meta.family;
        optgroup = document.createElement("optgroup");
        optgroup.label = S.familyLabel(group);
        sel.appendChild(optgroup);
      }
      const opt = document.createElement("option");
      opt.value = key;
      const label = S.methodLabel(meta.name);
      opt.textContent = meta.in_table ? label : `${label} (not in table)`;
      opt.selected = state.selected.includes(key);
      optgroup.appendChild(opt);
    }
  }

  function setSelected(keys) {
    state.selected = keys.filter((k) => MEAN.methods[k]);
    syncMethodList();
    render();
  }

  function bestOfEachFamily() {
    const best = {};
    for (const [key, meta] of plottable()) {
      const final = meta.frac_hits[meta.frac_hits.length - 1];
      if (!best[meta.family] || final > best[meta.family].final) {
        best[meta.family] = { key, final };
      }
    }
    return S.FAMILIES.map((f) => best[f.key]).filter(Boolean).map((b) => b.key);
  }

  /** Populate OVER_BUDGET and say on the page which methods overran. */
  function noteOverBudget() {
    const tol = MEAN.overshoot_tolerance;
    if (tol === undefined) {
      // An older recovery_mean.json has no overshoot field, so the note below
      // cannot be written. Say so rather than leaving the page silent about a
      // property of the curves it is drawing.
      console.error("recovery_mean.json predates the budget-overshoot check; " +
                    "re-run docs/build_data.py.");
      return;
    }
    for (const [key, meta] of Object.entries(MEAN.methods)) {
      if (meta.budget_overshoot > tol) OVER_BUDGET[key] = meta.budget_overshoot;
    }
    const el = document.getElementById("overbudget-note");
    const n = Object.keys(OVER_BUDGET).length;
    if (!el || !n) return;
    el.hidden = false;
    el.innerHTML =
      "<strong>The x axis is genes assayed, not genes budgeted.</strong> " +
      `${n === 1 ? "One method returns" : `${n} methods return`} more genes ` +
      "per round than the round asked for &mdash; " +
      Object.entries(OVER_BUDGET).sort((a, b) => b[1] - a[1]).map(([k, r]) =>
        `<em>${S.methodLabel(MEAN.methods[k].name)}</em> at ` +
        `${r.toFixed(2)}&times;`).join(", ") +
      " by the tenth round &mdash; so their curves run further to the right " +
      "than the rest and end past the 1,000-gene mark. That is the honest " +
      "placement, and it is the same accounting the results table uses: EF " +
      "divides by every gene acquired, including the ones outside the screen's " +
      "library. Read a vertical slice, not a round number. " +
      "<a href=\"results.html\">See the table &rarr;</a>";
  }

  async function init() {
    const caption = document.getElementById("caption");
    try {
      [MEAN, BY_SCREEN] = await Promise.all([
        S.fetchJSON("assets/data/recovery_mean.json"),
        S.fetchJSON("assets/data/recovery_by_screen.json"),
      ]);
    } catch (err) {
      S.showError(caption, err);
      return;
    }

    noteOverBudget();

    const fam = document.getElementById("family");
    for (const f of S.FAMILIES) {
      if (!plottable().some(([, m]) => m.family === f.key)) continue;
      const opt = document.createElement("option");
      opt.value = f.key; opt.textContent = f.label;
      fam.appendChild(opt);
    }

    const screenSel = document.getElementById("screen");
    for (const s of BY_SCREEN.screens) {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s;
      screenSel.appendChild(opt);
    }
    state.screen = BY_SCREEN.screens[0];

    document.getElementById("metric").addEventListener("change", (e) => {
      state.metric = e.target.value; render();
    });
    document.getElementById("view").addEventListener("change", (e) => {
      state.view = e.target.value;
      document.getElementById("screen-control").hidden = state.view !== "screen";
      render();
    });
    screenSel.addEventListener("change", (e) => {
      state.screen = e.target.value; render();
    });
    document.getElementById("band").addEventListener("change", (e) => {
      state.band = e.target.value; render();
    });
    fam.addEventListener("change", (e) => {
      state.family = e.target.value;
      syncMethodList();
    });
    document.getElementById("methods").addEventListener("change", (e) => {
      state.selected = Array.from(e.target.selectedOptions).map((o) => o.value);
      render();
    });
    document.getElementById("preset-headline").addEventListener("click",
      () => setSelected(HEADLINE));
    document.getElementById("preset-families").addEventListener("click",
      () => setSelected(bestOfEachFamily()));
    document.getElementById("preset-clear").addEventListener("click",
      () => setSelected([]));

    // Any headline method absent from the data is a build problem, not a
    // display problem -- say so instead of quietly plotting a shorter list.
    const absent = HEADLINE.filter((k) => !MEAN.methods[k]);
    if (absent.length) {
      console.warn("Headline preset references unknown methods:", absent);
    }
    setSelected(HEADLINE);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
