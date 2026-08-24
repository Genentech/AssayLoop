// Leave-one-phenotype-out, and the label ablation, as live charts.
//
// Both live on results.html and both are small enough that a picture of them
// was never the right medium: five folds by three methods, and seven dumbbells.
// Reads assets/data/lopo.json and assets/data/label_ablation.json.

(function () {
  const S = window.AssayLoop;

  // The three LOPO series, in the order they are argued: the baseline, what
  // supervised training buys over it, and what RL buys on top.
  const SERIES_COLOR = {
    knn: "#8a8880",
    sup: "#1FA187",
    rl_best: "#c2410c",
  };

  async function renderLopo() {
    const node = document.getElementById("lopo-chart");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/lopo.json");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    // Fold labels carry the test-set size, which is the first thing anyone
    // asks when a fold looks like an outlier.
    const ticks = data.folds.map((f, i) => `${f}<br>` +
      `<span style="font-size:11px;opacity:0.65">n = ` +
      `${data.n_test[i].toLocaleString()}</span>`);

    const traces = data.series.map((s) => ({
      type: "bar",
      name: s.label,
      x: ticks,
      y: s.values,
      marker: { color: SERIES_COLOR[s.key] || "#8a8880" },
      hovertemplate: `<b>${s.label}</b><br>EF %{y:.2f}&times;<extra></extra>`,
    }));

    const layout = S.mergeLayout({
      barmode: "group",
      height: 460,
      yaxis: { title: "Enrichment factor (EF)", rangemode: "tozero" },
      xaxis: { title: "" },
      legend: { orientation: "h", y: 1.12, x: 0 },
      // 1x is chance. Without the line the bars float, and the reader has to
      // find the number on the axis to know which of them cleared it.
      shapes: [{
        type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1,
        line: { color: "#9ca3af", width: 1, dash: "dot" },
      }],
      annotations: [{
        xref: "paper", x: 1, y: 1, yanchor: "bottom", xanchor: "right",
        text: "chance", showarrow: false,
        font: { size: 11, color: "#8a8880" },
      }],
    });
    Plotly.react(node, traces, layout, S.PLOTLY_CONFIG);

    const caption = document.getElementById("lopo-caption");
    if (caption) {
      caption.innerHTML =
        `<span class="label">Five folds.</span> Each bar is one of AssayBench's ` +
        `broad phenotype families held out of training entirely and then ` +
        `evaluated on. The ordering holds in every fold.`;
    }
  }

  async function renderAblation() {
    const node = document.getElementById("ablation-chart");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/label_ablation.json");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    // The figure's own order: LLMs by EF, then the two rows blinded a
    // different way. Plotly's category axis draws bottom-up, so reverse.
    const rows = data.rows.slice().reverse();
    const names = rows.map((r) => r.name);

    // One connector per row, drawn as a single trace with nulls between the
    // segments -- seven two-point traces would put seven entries in the legend.
    const linkX = [], linkY = [];
    for (const r of rows) {
      linkX.push(r.without_feedback, r.with_feedback, null);
      linkY.push(r.name, r.name, null);
    }

    const traces = [
      {
        type: "scatter", mode: "lines", x: linkX, y: linkY,
        line: { color: "#8ab4e8", width: 3 },
        hoverinfo: "skip", showlegend: false,
      },
      {
        type: "scatter", mode: "markers", name: "Without hit feedback",
        x: rows.map((r) => r.without_feedback), y: names,
        marker: { size: 13, color: "#8ab4e8" },
        hovertemplate: "<b>%{y}</b><br>blind: EF %{x:.2f}&times;<extra></extra>",
      },
      {
        type: "scatter", mode: "markers", name: "With hit feedback",
        x: rows.map((r) => r.with_feedback), y: names,
        marker: { size: 13, color: "#1d4f91" },
        customdata: rows.map((r) => [r.gap, r.gap_stderr, r.group]),
        hovertemplate:
          "<b>%{y}</b><br>with feedback: EF %{x:.2f}&times;" +
          "<br>gain: +%{customdata[0]:.2f} &plusmn; %{customdata[1]:.2f}" +
          "<br><i>blinded by: %{customdata[2]}</i><extra></extra>",
      },
    ];

    // The rule between the two blinding mechanisms, placed between the last
    // "prompt-stripped" row and the first "no readout" row.
    const boundary = rows.findIndex((r) => r.group !== rows[0].group);
    const shapes = boundary > 0 ? [{
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: boundary - 0.5, y1: boundary - 0.5,
      line: { color: "#d1d5db", width: 1 },
    }] : [];

    const layout = S.mergeLayout({
      height: 60 + 52 * rows.length,
      // `automargin` on the x axis as well as a bigger bottom margin: 56px is
      // the tick row plus the title only if the title sits tight against it,
      // and Plotly clipped the descenders off "Enrichment factor (EF)".
      margin: { l: 150, r: 70, t: 10, b: 78 },
      xaxis: { title: "Enrichment factor (EF)", automargin: true },
      yaxis: { type: "category", automargin: true },
      legend: { orientation: "h", y: 1.08, x: 0 },
      shapes: shapes,
    });
    Plotly.react(node, traces, layout, S.PLOTLY_CONFIG);

    const caption = document.getElementById("ablation-caption");
    if (caption) {
      caption.innerHTML =
        `<span class="label">Every method gains.</span> The two rows below the ` +
        `rule are blinded differently: the LLMs run the same loop with the hit ` +
        `labels stripped from the prompt, while AssayFormer and AssayLoop see ` +
        `no readout at all. Hover for the paired standard error of each gap, ` +
        `taken over the 20 shared test screens.`;
    }
  }

  function init() {
    renderLopo();
    renderAblation();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
