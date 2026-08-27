// The pathway sunburst, live rather than as a picture of one.
//
// Figure 6 prints six sunbursts side by side at about two inches across, which
// is enough to see that the methods differ and not enough to see how. This is
// the same six panels in the same order with the same hues -- so a colour means
// the same Reactome category in every panel, which is the whole point of the
// figure -- but hovering names a wedge and clicking a category drills into its
// level-2 groups. "One method" swaps the grid for a single large panel.
//
// Reads assets/data/sunburst.json, which docs/build_data.py aggregates using
// the cutoffs read out of plot_pathway_sunburst.py itself, so the rings here
// hold exactly what the paper's rings hold.

(function () {
  const S = window.AssayLoop;
  const state = { method: "" };   // "" is the six-panel grid
  let DATA = null;
  // Panel index -> the first of the two annotation slots holding its centre
  // text, so the drill-down can rewrite the one panel that was clicked.
  let holeAt = [];
  let panels = [];

  const GRID = { rows: 2, cols: 3 };

  /** One sunburst trace, placed in a rectangle of the paper coordinate space. */
  function trace(name, domain) {
    const m = DATA.methods[name];
    // Values are integer millionths of the panel -- see build_data.py for why
    // they are not floats. Plotly only ever ratios them; the hover wants a
    // number a reader can hold, so convert to gene-equivalents here.
    const counts = m.values.map((v) => (v / DATA.scale) * m.n_annotated);
    return {
      type: "sunburst",
      ids: m.ids,
      labels: m.labels,
      parents: m.parents,
      values: m.values,
      customdata: counts,
      branchvalues: "total",
      // The synthetic root plus the two rings the figure draws.
      maxdepth: 3,
      marker: { colors: m.colors, line: { color: "#ffffff", width: 1.2 } },
      // The wedges arrive already ordered largest-first within each category;
      // letting Plotly re-sort would break the alternating tints.
      sort: false,
      rotation: 90,
      // No wedge labels: at this size they do not fit, and the hover and the
      // colour key carry the names. The centre number is a layout annotation.
      textinfo: "none",
      hovertemplate:
        "<b>%{label}</b><br>%{percentRoot:.1%} of annotated picks<br>" +
        "%{customdata:.0f} gene-equivalents<extra></extra>",
      domain: domain,
    };
  }

  //: Point sizes for the two views. The grid draws six panels across a wide
  //: container, so its type has to hold up at a third of the column -- the
  //: first cut used the chart-furniture sizes (12.5pt titles, 16pt centres) and
  //: they read as captions rather than as labels.
  const TYPE = {
    grid: { title: 16, hole: 26 },
    one: { hole: 36 },
  };

  /** The centre of a panel: EP-D at the top level, the node's name once
   *  drilled in, so the hole never keeps showing a number for a ring that is
   *  no longer on screen. Returns a *pair* of annotations, always in that
   *  order, so the caller can relayout both by index.
   *
   *  Two annotations rather than one two-line one. Plotly lays a `<b>80</b><br>
   *  <span size=11>EP-D</span>` block out from the larger font's line box and
   *  then centres that box, which puts the number above the middle and drops
   *  the sub-label onto the inner ring -- visibly wrong in the six-panel grid.
   *  Anchoring each line at the same point and nudging it by a fixed pixel
   *  offset is stable, because the hole's radius is set by the panel's pixel
   *  height (900 / 620 below) and does not move when the container resizes. */
  function hole(main, sub, x, y, size) {
    function line(text, font, yshift) {
      return {
        text: text, showarrow: false, align: "center",
        font: font, yshift: yshift,
        x: x, y: y, xref: "paper", yref: "paper",
        xanchor: "center", yanchor: "middle",
      };
    }
    return [
      line(main ? `<b>${main}</b>` : "",
           { family: "Inter, sans-serif", size: size, color: "#0b0b0b" },
           Math.round(size * 0.28)),
      // Drilled in there is no number above it, so the label is the whole of
      // the centre and sits on the point rather than below it.
      line(sub,
           { family: "Inter, sans-serif", size: Math.round(size * 0.44),
             color: "#52514e" },
           main ? -Math.round(size * 0.47) : 0),
    ];
  }

  function epHole(name, x, y, size) {
    return hole(DATA.methods[name].eff_pathways.toFixed(0), "EP-D", x, y, size);
  }

  /** The shared category key. Plotly draws no legend for a sunburst, and the
   *  panels are only comparable if the reader knows the hues are shared. */
  function renderLegend() {
    const node = document.getElementById("sunburst-legend");
    if (!node) return;
    node.innerHTML = DATA.categories.map((c) =>
      `<span class="swatch-item"><span class="swatch" style="background:` +
      `${DATA.colors[c]}"></span>${c}</span>`).join("");
  }

  /** Re-label one panel's hole after a click. `id` is the node Plotly is about
   *  to zoom to; the synthetic root means we are back at the top. */
  function onDrill(node, panel, id) {
    const a = holeAt[panel];
    if (a === undefined) return;
    const name = panels[panel];
    const size = panels.length > 1 ? TYPE.grid.hole : TYPE.one.hole;
    const anns = node.layout.annotations.slice();
    const cx = anns[a].x;
    const cy = anns[a].y;
    let next;
    if (!id || id === DATA.root) {
      next = epHole(name, cx, cy, size);
    } else {
      const m = DATA.methods[name];
      const label = m.labels[m.ids.indexOf(id)] || id;
      // A level-2 group name runs long; the hole is small. Two words in and
      // the reader knows which wedge they are in, and the hover has the rest.
      const short = label.length > 22 ? label.slice(0, 21) + "…" : label;
      next = hole("", short, cx, cy, size);
    }
    anns[a] = next[0];
    anns[a + 1] = next[1];
    // The whole array, not `annotations[i]`. Handing Plotly an object at an
    // existing array index *inserts* rather than replaces, which left the
    // drilled label sitting on top of the EP-D number it was meant to stand in
    // for -- two centres in one hole.
    Plotly.relayout(node, { annotations: anns });
  }

  function wire(node) {
    node.on("plotly_sunburstclick", (ev) => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      onDrill(node, pt.curveNumber, ev.nextLevel);
    });
  }

  function renderGrid(node) {
    panels = DATA.order.slice();
    holeAt = [];
    const traces = [];
    const annotations = [];
    // Plotly's own grid leaves no room for a per-panel title, so the cells are
    // laid out by hand: a band at the top of each cell carries the name.
    const cellW = 1 / GRID.cols;
    const cellH = 1 / GRID.rows;
    const band = 0.10;
    panels.forEach((name, i) => {
      const col = i % GRID.cols;
      const row = Math.floor(i / GRID.cols);
      const x0 = col * cellW;
      // Rows read top-down; paper coordinates run bottom-up.
      const yTop = 1 - row * cellH;
      const yLo = yTop - cellH + 0.02;
      const yHi = yTop - band;
      traces.push(trace(name, {
        x: [x0 + 0.012, x0 + cellW - 0.012], y: [yLo, yHi],
      }));
      annotations.push({
        text: S.methodLabel(name), showarrow: false,
        font: { family: "Inter, sans-serif", size: TYPE.grid.title,
                color: "#0b0b0b" },
        x: x0 + cellW / 2, y: yHi + 0.014, xref: "paper", yref: "paper",
        xanchor: "center", yanchor: "bottom",
      });
      holeAt.push(annotations.length);
      epHole(name, x0 + cellW / 2, (yLo + yHi) / 2, TYPE.grid.hole)
        .forEach((a) => annotations.push(a));
    });
    Plotly.react(node, traces, S.mergeLayout({
      margin: { l: 4, r: 4, t: 8, b: 4 },
      height: 900,
      annotations: annotations,
    }), S.PLOTLY_CONFIG);
  }

  function renderOne(node, name) {
    panels = [name];
    holeAt = [0];
    Plotly.react(node, [trace(name, { x: [0, 1], y: [0, 1] })], S.mergeLayout({
      margin: { l: 0, r: 0, t: 10, b: 0 },
      height: 620,
      annotations: epHole(name, 0.5, 0.5, TYPE.one.hole),
    }), S.PLOTLY_CONFIG);
  }

  function render() {
    const node = document.getElementById("sunburst");
    const caption = document.getElementById("sunburst-caption");
    const name = state.method;

    if (name && !DATA.methods[name]) {
      Plotly.purge(node);
      caption.textContent = "";
      return;
    }
    if (name) renderOne(node, name); else renderGrid(node);
    wire(node);

    const instructions =
      `The number at the centre is EP-D, the effective pathway count at ` +
      `dataset scope. Hover a wedge to name it; click a category to zoom in, ` +
      `click the centre to zoom back out.`;

    if (!name) {
      caption.innerHTML = instructions;
      return;
    }
    const m = DATA.methods[name];
    caption.innerHTML =
      `<span class="label">${S.methodLabel(name)}.</span> ` +
      `${m.n_annotated.toLocaleString()} of ${m.n_picks.toLocaleString()} ` +
      `picks are annotated in Reactome. ` + instructions;
  }

  async function init() {
    const node = document.getElementById("sunburst");
    if (!node) return;
    try {
      DATA = await S.fetchJSON("assets/data/sunburst.json");
    } catch (err) {
      S.showError(node, err);
      return;
    }
    renderLegend();

    const select = document.getElementById("sunburst-method");
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All six methods, as in the paper";
    select.appendChild(all);
    for (const n of DATA.order) {
      const opt = document.createElement("option");
      opt.value = n;
      opt.textContent = S.methodLabel(n);
      select.appendChild(opt);
    }
    select.value = state.method;
    select.addEventListener("change", () => {
      state.method = select.value;
      // The trace count changes between the two views, so a stale trace would
      // otherwise linger in a cell nothing draws into any more.
      Plotly.purge(node);
      render();
    });
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
