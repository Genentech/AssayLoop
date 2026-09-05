// Full-test-set LOPO comparison and label-feedback ablation for results.html.
// Reads assets/data/lopo.json and assets/data/label_ablation.json.

(function () {
  const S = window.AssayLoop;

  async function renderLopo() {
    const node = document.getElementById("lopo-chart");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/lopo.json?v=bcc98e9f");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    const rows = data.rows.slice().reverse();
    const trace = {
      type: "bar", orientation: "h",
      x: rows.map((r) => r.ef),
      y: rows.map((r) => r.label),
      text: rows.map((r) => r.ef.toFixed(2)),
      textposition: "outside",
      cliponaxis: false,
      marker: { color: rows.map((r) => r.color) },
      hovertemplate: "<b>%{y}</b><br>EF: %{x:.2f}&times;<extra></extra>",
    };
    const layout = S.mergeLayout({
      height: 350,
      title: { text: `Enrichment factor on the ${data.scope.toLowerCase()}`,
               x: 0.5, xanchor: "center" },
      margin: { l: 180, r: 70, t: 58, b: 72 },
      xaxis: { title: data.metric, range: [0, 5.35], automargin: true },
      yaxis: { type: "category", automargin: true },
      showlegend: false,
    });
    Plotly.react(node, [trace], layout, S.PLOTLY_CONFIG);
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
        customdata: rows.map((r) => [r.gap, r.group]),
        hovertemplate:
          "<b>%{y}</b><br>with feedback: EF %{x:.2f}&times;" +
          "<br>gain: +%{customdata[0]:.2f}" +
          "<br><i>blinded by: %{customdata[1]}</i><extra></extra>",
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

  }

  function init() {
    renderLopo();
    renderAblation();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
