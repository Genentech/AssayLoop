// The two analysis-page charts that have data behind them.
//
//   * which Reactome roots each model over- and under-picks, against a
//     uniform draw from the candidate universe;
//   * what observing one gene as a hit does to AssayFormer's belief about
//     every other gene.
//
// Both are heatmaps in the paper. On the page they stay heatmaps, but the
// numbers are readable on hover, which is the whole reason to do this live:
// the paper prints a 11x29 and a 12x20 grid at column width.

(function () {
  const S = window.AssayLoop;

  // A diverging ramp centred on "same as chance". Blue below, orange above --
  // the site's accent and its complement, so it does not read as a third
  // palette bolted on.
  const DIVERGING = [
    [0.0, "#2f6db1"], [0.25, "#8ab4e8"], [0.5, "#f4f4f1"],
    [0.75, "#f0a868"], [1.0, "#c2410c"],
  ];

  // Where the pathway scale saturates, in log2 units: half chance to twice
  // chance. Taken from the paper's own figure (plot_llm_pathway_heatmap.py,
  // CAP = 1.0). Scaling to the data instead puts the range at -9.1..+2.1,
  // because a few near-total avoidances (kNN barely touching extracellular
  // matrix organization, AssayFormer barely touching reproduction) drag the
  // blue end out until every ordinary over- and under-pick renders as the same
  // near-white cell. Values past the cap keep their true number in the hover.
  const PATHWAY_CAP = 1.0;

  /** Symmetric limit for a diverging scale, so 0 lands in the middle. */
  function symmetricLimit(z) {
    let m = 0;
    for (const row of z) {
      for (const v of row) {
        if (v !== null && Math.abs(v) > m) m = Math.abs(v);
      }
    }
    return m || 1;
  }

  async function renderPathways() {
    const node = document.getElementById("pathway-heatmap");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/pathway_heatmap.json?v=26ae8bdc");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    // Plotly draws the first row at the bottom; the JSON's order is the one
    // the paper argues in, so flip it to keep that reading top-down.
    const methods = data.methods.slice().reverse();
    const z = data.z.slice().reverse();
    const hover = data.hover.slice().reverse();
    const customdata = hover.map((row, i) => {
      const method = methods[i];
      const summary =
        `EP-D: ${data.eff_pathways[method].toFixed(1)}` +
        `<br>unique genes: ${data.unique_genes[method].toLocaleString()}` +
        `<br>Reactome-annotated picks: ` +
        `${(100 * data.annotated_fraction[method]).toFixed(0)}%`;
      return row.map((cell) => `${cell}<br>${summary}`);
    });

    const trace = {
      type: "heatmap",
      x: data.categories,
      // Method name alone. The row label used to carry each model's EP-D in
      // brackets, which put a second, unrelated number on an axis that is
      // already about a different quantity; the diversity page is where EP
      // belongs, and the hover below still names the row.
      y: methods,
      z: z,
      customdata: customdata,
      colorscale: DIVERGING,
      // z is left uncapped and the *scale* is clamped instead, so Plotly
      // saturates the colour while %{z} in the hover still reports what the
      // cell actually is.
      zmin: -PATHWAY_CAP, zmax: PATHWAY_CAP,
      xgap: 1, ygap: 1,
      hovertemplate: "<b>%{y}</b><br>%{customdata}<br>" +
                     "log<sub>2</sub> ratio to chance: %{z:+.2f}<extra></extra>",
      colorbar: {
        title: { text: "log₂ vs chance", side: "right" },
        thickness: 12, len: 0.8, outlinewidth: 0,
        // Say that the ends are stops, not maxima.
        tickvals: [-1, -0.5, 0, 0.5, 1],
        ticktext: ["≤ −1", "−0.5", "0", "+0.5", "≥ +1"],
      },
    };

    const layout = S.mergeLayout({
      height: 70 + 30 * methods.length,
      margin: { l: 220, r: 30, t: 10, b: 190 },
      xaxis: { tickangle: -40, automargin: true, showgrid: false },
      yaxis: { automargin: true, showgrid: false },
    });
    Plotly.react(node, [trace], layout, S.PLOTLY_CONFIG);

    const caption = document.getElementById("pathway-heatmap-caption");
    if (caption) {
      caption.innerHTML =
        `<span class="label">Orange is over-picked, blue under-picked.</span> ` +
        `The scale saturates at &plusmn;1 in log<sub>2</sub> space, from half ` +
        `the random share to twice the random share. Hover for the uncapped ` +
        `ratio, raw pathway share, EP-D, unique-gene count, and Reactome ` +
        `annotation coverage. A blank cell means the model never selected a ` +
        `gene from that category. ` +
        `<a href="diversity.html">How effective pathways are calculated ` +
        `&rarr;</a>`;
    }
  }

  async function renderInfluence() {
    const node = document.getElementById("influence-matrix");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/influence.json");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    const probes = data.probes.slice().reverse();
    const z = probes.map((p) => data.z[data.probes.indexOf(p)]);
    const lim = symmetricLimit(z);

    // The paper draws fifteen of these 240 cells and names the mechanism it
    // reads into each. Carry those notes into the hover rather than dropping
    // them: they are the interpretation, and the matrix alone does not have it.
    const notes = probes.map((p) =>
      data.targets.map((t) => {
        const n = data.featured[`${p}\t${t}`];
        // Labelled, not bare. On its own a two-word mechanism ("Nucleolar
        // stress") reads as another data field of the cell rather than as the
        // paper's reading of it.
        return n ? `<br><i>Explanation: ${n.note}</i>` : "";
      }));

    const trace = {
      type: "heatmap",
      x: data.targets, y: probes, z: z,
      customdata: notes,
      colorscale: DIVERGING,
      zmin: -lim, zmax: lim,
      xgap: 1, ygap: 1,
      hovertemplate:
        "observing <b>%{y}</b> as a hit moves <b>%{x}</b> by " +
        "%{z:+.3f}%{customdata}<extra></extra>",
      colorbar: {
        title: { text: "Δ hit probability", side: "right" },
        thickness: 12, len: 0.8, outlinewidth: 0,
      },
    };

    // Ring the cells the paper singles out, so the figure's claim is locatable
    // in the matrix rather than only in the hover.
    const shapes = [];
    for (const key of Object.keys(data.featured)) {
      const [probe, target] = key.split("\t");
      const yi = probes.indexOf(probe);
      const xi = data.targets.indexOf(target);
      if (yi < 0 || xi < 0) continue;
      shapes.push({
        type: "rect",
        x0: xi - 0.5, x1: xi + 0.5, y0: yi - 0.5, y1: yi + 0.5,
        line: { color: "#1b1f24", width: 1.4 }, fillcolor: "rgba(0,0,0,0)",
      });
    }

    const layout = S.mergeLayout({
      height: 90 + 34 * probes.length,
      margin: { l: 80, r: 30, t: 10, b: 110 },
      xaxis: { tickangle: -45, automargin: true, showgrid: false,
               title: "target gene" },
      yaxis: { automargin: true, showgrid: false, title: "probe observed as a hit" },
      shapes: shapes,
    });
    Plotly.react(node, [trace], layout, S.PLOTLY_CONFIG);

    const caption = document.getElementById("influence-matrix-caption");
    if (caption) {
      caption.innerHTML =
        `<span class="label">Outlined cells are the pairs the paper discusses;` +
        `</span> hover one for the mechanism it reads into it. Orange means ` +
        `observing the probe as a hit raises the model's belief about the ` +
        `target, and blue means it lowers it. Influence is a directional, ` +
        `context-dependent update rather than a symmetric gene association.`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderPathways();
    renderInfluence();
  });
})();
