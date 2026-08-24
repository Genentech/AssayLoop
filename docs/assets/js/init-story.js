// Textbook-biology recovery against downstream task performance.
//
// One point per embedding initialisation at each of two training stages,
// connected vertically: x is a property of the raw embedding and does not move
// when you train on it. The claim is the negative slope -- the embedding that
// best recovers curated interaction networks is the worst one to start from.
// Reads assets/data/init_story.json.

(function () {
  const S = window.AssayLoop;

  // Only two inits carry the argument; the rest are the cloud that makes the
  // trend a trend rather than an anecdote.
  const CALLED_OUT = { BPMF: "#2f6db1", GenePT: "#c2410c" };
  const GRAY = "#a9a79f";

  /** Least-squares fit over `pts`, returned as endpoints across `[x0, x1]`. */
  function fit(pts, x0, x1) {
    const n = pts.length;
    const mx = pts.reduce((a, p) => a + p[0], 0) / n;
    const my = pts.reduce((a, p) => a + p[1], 0) / n;
    let num = 0, den = 0;
    for (const [x, y] of pts) { num += (x - mx) * (y - my); den += (x - mx) ** 2; }
    const slope = den ? num / den : 0;
    return [[x0, my + slope * (x0 - mx)], [x1, my + slope * (x1 - mx)]];
  }

  async function init() {
    const node = document.getElementById("init-story");
    if (!node) return;
    let data;
    try {
      data = await S.fetchJSON("assets/data/init_story.json");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    const pts = data.points;
    const xs = pts.map((p) => p.auroc);
    const pad = (Math.max(...xs) - Math.min(...xs)) * 0.12;
    const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;

    const traces = [];

    // The two fits go in first so the markers draw over them.
    for (const [key, label, dash] of [["supervised", "supervised", "dot"],
                                      ["rl", "+ GRPO", "dot"]]) {
      const line = fit(pts.map((p) => [p.auroc, p[key]]), x0, x1);
      traces.push({
        type: "scatter", mode: "lines",
        x: line.map((q) => q[0]), y: line.map((q) => q[1]),
        line: { color: "#c9c8c0", width: 1.2, dash: dash },
        name: `${label} trend`, hoverinfo: "skip", showlegend: false,
      });
    }

    // A connector per init, drawn with nulls between so one trace carries
    // several. Plotly cannot colour segments of a line trace individually, so
    // the two called-out inits get their own and the rest share the grey one.
    for (const group of [null, "BPMF", "GenePT"]) {
      const sel = group ? pts.filter((p) => p.init === group)
                        : pts.filter((p) => !CALLED_OUT[p.init]);
      if (!sel.length) continue;
      const gx = [], gy = [];
      for (const p of sel) { gx.push(p.auroc, p.auroc, null); gy.push(p.supervised, p.rl, null); }
      traces.push({
        type: "scatter", mode: "lines", x: gx, y: gy,
        line: { color: group ? CALLED_OUT[group] : GRAY, width: 2 },
        hoverinfo: "skip", showlegend: false,
      });
    }

    const dbList = data.databases.join(" / ");
    const hover = (stage) => (p) =>
      `<b>${p.init}</b><br>${stage}: EF ${p[stage === "+ GRPO" ? "rl" : "supervised"].toFixed(2)}×` +
      `<br>mean recovery AUROC ${p.auroc.toFixed(3)}<br>` +
      data.databases.map((db) => `${db} ${p.per_db[db].toFixed(3)}`).join(" · ");

    traces.push({
      type: "scatter", mode: "markers", name: "Supervised",
      x: xs, y: pts.map((p) => p.supervised),
      marker: {
        size: 13, color: pts.map((p) => CALLED_OUT[p.init] || GRAY),
        line: { width: 0 },
      },
      text: pts.map(hover("supervised")),
      hovertemplate: "%{text}<extra></extra>",
    });
    traces.push({
      type: "scatter", mode: "markers+text", name: "+ GRPO (RL)",
      x: xs, y: pts.map((p) => p.rl),
      marker: {
        size: 14, color: "rgba(0,0,0,0)",
        line: { width: 2.2, color: pts.map((p) => CALLED_OUT[p.init] || GRAY) },
      },
      text: pts.map((p) => p.init),
      textposition: "top center",
      textfont: {
        size: 12,
        color: pts.map((p) => CALLED_OUT[p.init] ? "#1b1f24" : "#52514e"),
      },
      customdata: pts.map(hover("+ GRPO")),
      hovertemplate: "%{customdata}<extra></extra>",
    });

    const layout = S.mergeLayout({
      height: 520,
      margin: { l: 70, r: 30, t: 16, b: 62 },
      xaxis: { title: `Average recovery AUROC (${dbList})`, range: [x0, x1] },
      yaxis: { title: "Downstream task performance (AssayFormer EF)" },
      legend: { orientation: "h", y: 1.1, x: 0 },
      hovermode: "closest",
    });
    // No caption. The chart's own legend and axis titles say what the marks
    // are, and the paragraph above it makes the argument; a caption restating
    // both was three sentences of duplication under a small chart.
    Plotly.react(node, traces, layout, S.PLOTLY_CONFIG);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
