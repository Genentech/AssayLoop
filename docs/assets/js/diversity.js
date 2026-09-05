// The pathway-diversity explorer.
//
// Reads assets/data/diversity.json, which docs/build_data.py slices out of the
// same `full_genome_table.py --json` run that produces the paper's table. The
// EP columns are nullable by design: a scope below the retention floor has no
// value, and every view here has to keep saying so rather than drawing a zero.

(function () {
  const S = window.AssayLoop;

  const SCOPES = {
    // One decimal at every scope, matching the LaTeX table: the vocabulary is
    // 186 Reactome level-2 groups, and the between-method range at batch scope
    // is only ~8 groups wide, so the tenths digit carries signal.
    ep_b: { label: "EP-B (per batch)", short: "EP-B", digits: 1,
            axis: "Effective number of pathways per 100-gene batch" },
    ep_s: { label: "EP-S (per screen)", short: "EP-S", digits: 1,
            axis: "Effective number of pathways per screen" },
    ep_d: { label: "EP-D (whole test set)", short: "EP-D", digits: 1,
            axis: "Effective number of pathways across the test set" },
    vendi: { label: "Vendi score", short: "Vendi", digits: 1,
             axis: "Vendi score (embedding diversity)" },
  };

  const state = { scope: "ep_b", scatterScope: "ep_s", family: "",
                  ablations: "hide", order: "value" };
  let DATA = null;

  // The manuscript now uses the descriptive method name Probability-of-Hit.
  // Keep the underlying result key unchanged so this page still joins cleanly
  // to the shared benchmark JSON.
  function methodLabel(text) {
    return S.methodLabel(text === "Haystacks" ? "Probability-of-Hit" : text);
  }

  function isAblationRow(row) {
    return row.family === "ablation" || row.display === "- hit labels";
  }

  function visibleRows() {
    return DATA.rows.filter((r) => {
      if (state.ablations === "hide" && isAblationRow(r)) return false;
      if (state.family && r.family !== state.family) return false;
      return true;
    });
  }

  function scopeNote(key) {
    const counts = DATA.reference_counts;
    if (key === "ep_b") {
      return `Each method is compared using ${counts.batch} Reactome-annotated ` +
             `genes per full batch.`;
    }
    if (key === "ep_s") {
      return `Each method is compared using ${counts.screen} Reactome-annotated ` +
             `genes per screen.`;
    }
    if (key === "ep_d") {
      return `Each method is compared using ${counts.dataset.toLocaleString()} ` +
             `Reactome-annotated genes across the test set.`;
    }
    return "Embedding diversity within acquisition batches, scaled relative to random picking.";
  }

  // ------------------------------------------------------------------ bars

  function renderBars() {
    const scope = SCOPES[state.scope];
    const all = visibleRows();
    // Nulls are not zero-height bars. They are pulled out and named in the
    // caption instead, so the chart never implies a method scored badly when
    // in fact it was not scored at all.
    const scored = all.filter((r) => r[state.scope] !== null);
    const unscored = all.filter((r) => r[state.scope] === null);

    const rows = state.order === "value"
      ? scored.slice().sort((a, b) => a[state.scope] - b[state.scope])
      : scored.slice().reverse();  // table order reads top-down on a bar chart

    const node = document.getElementById("bars");
    const caption = document.getElementById("bars-caption");

    if (!rows.length) {
      Plotly.purge(node);
      node.innerHTML = "";
      caption.innerHTML = `<span class="label">Nothing to plot.</span> ` +
        `No method in this filter has a ${scope.short} value.`;
      return;
    }

    const trace = {
      type: "bar", orientation: "h",
      x: rows.map((r) => r[state.scope]),
      y: rows.map((r) => methodLabel(r.name)),
      marker: { color: rows.map((r) => S.familyColor(r.family)) },
      hovertemplate: `<b>%{y}</b><br>${scope.short}: %{x}<extra></extra>`,
    };
    const layout = S.mergeLayout({
      // Same clipping fix as the ablation chart: 56px of bottom margin fits
      // the tick row and the axis title only if they sit on top of each other,
      // and Plotly cropped the descenders off "...per 100-gene batch".
      xaxis: { title: scope.axis, rangemode: "tozero", automargin: true },
      yaxis: { automargin: true, tickfont: { size: 11 } },
      margin: { l: 260, r: 24, t: 20, b: 72 },
      height: Math.max(360, 22 * rows.length + 90),
      showlegend: false,
    });
    Plotly.react(node, [trace], layout, S.PLOTLY_CONFIG);

    // Two different reasons a method has no bar, and they are not
    // interchangeable: the floor is a property of the metric, "not evaluated"
    // is a property of this build. Name them separately.
    const belowFloor = unscored.filter((r) => r.available !== false);
    const notRun = unscored.filter((r) => r.available === false);
    const why = [];
    if (belowFloor.length) {
      const retention = Math.round(DATA.retention_floor * 100);
      why.push(`Not shown: ${belowFloor.map((r) => methodLabel(r.name)).join(", ")} ` +
               `&mdash; fewer than ${retention}% of their units contain enough ` +
               `Reactome-annotated genes to be scored. <a href="#floor">Why.</a>`);
    }
    if (notRun.length) {
      why.push(`Also missing: ${notRun.map((r) => methodLabel(r.name)).join(", ")} ` +
               `&mdash; not evaluated on this build, so nothing was measured ` +
               `at any scope.`);
    }
    const note = scopeNote(state.scope);
    caption.innerHTML =
      `<span class="label">${scope.label}.</span> ${note} ` +
      (why.length ? why.join(" ")
                  : "Every method in this filter has a value at this scope.");
  }

  // --------------------------------------------------------------- scatter

  function renderScatter() {
    const scope = SCOPES[state.scatterScope];
    const rows = visibleRows().filter(
      (r) => r[state.scatterScope] !== null && r.ef !== null);
    const node = document.getElementById("scatter");
    const caption = document.getElementById("scatter-caption");

    const traces = S.FAMILIES.map((f) => {
      const sub = rows.filter((r) => r.family === f.key);
      if (!sub.length) return null;
      return {
        type: "scatter", mode: "markers", name: f.label,
        x: sub.map((r) => r[state.scatterScope]),
        y: sub.map((r) => r.ef),
        text: sub.map((r) => methodLabel(r.name)),
        marker: { color: f.color, size: 10, line: { color: "#ffffff", width: 1 } },
        hovertemplate: `<b>%{text}</b><br>${scope.short}: %{x}<br>` +
                       `EF: %{y:.2f}<extra></extra>`,
      };
    }).filter(Boolean);

    const layout = S.mergeLayout({
      xaxis: { title: scope.axis },
      yaxis: { title: "Enrichment factor (EF)" },
      legend: { orientation: "h", y: -0.22, x: 0 },
      margin: { l: 68, r: 24, t: 20, b: 96 },
      height: 520,
    });
    Plotly.react(node, traces, layout, S.PLOTLY_CONFIG);

    const dropped = visibleRows().length - rows.length;
    caption.innerHTML =
      `<span class="label">EF against ${scope.short}.</span> ` +
      `Each point is one method. ` +
      (dropped
        ? `${dropped} method${dropped === 1 ? " is" : "s are"} omitted for having no ` +
          `${scope.short} value.`
        : "");
  }

  // ------------------------------------------------- metric reference counts

  /** Fill the rarefaction counts quoted in "How the metric works".
   *
   * They are properties of the metric, not of this run, but they are read out
   * of the data rather than typed into the page so the prose cannot drift from
   * the numbers the table was actually scored with. */
  function renderMetricCounts() {
    const counts = DATA.reference_counts;
    document.getElementById("m-batch").textContent = counts.batch;
    document.getElementById("m-screen").textContent = counts.screen;
    document.getElementById("m-dataset").textContent = counts.dataset;
    const retention = `${Math.round(DATA.retention_floor * 100)}%`;
    document.querySelectorAll(".retention-floor").forEach((node) => {
      node.textContent = retention;
    });
  }

  // ------------------------------------------------------------------ init

  function renderAll() { renderBars(); renderScatter(); }

  async function init() {
    const caption = document.getElementById("bars-caption");
    try {
      DATA = await S.fetchJSON("assets/data/diversity.json");
    } catch (err) {
      S.showError(caption, err);
      return;
    }

    const fam = document.getElementById("family");
    for (const f of S.FAMILIES) {
      if (!DATA.rows.some((r) => r.family === f.key)) continue;
      const opt = document.createElement("option");
      opt.value = f.key; opt.textContent = f.label;
      fam.appendChild(opt);
    }

    document.getElementById("scope").addEventListener("change", (e) => {
      state.scope = e.target.value; renderBars();
    });
    document.getElementById("scatter-scope").addEventListener("change", (e) => {
      state.scatterScope = e.target.value; renderScatter();
    });
    fam.addEventListener("change", (e) => { state.family = e.target.value; renderAll(); });
    document.getElementById("ablations").addEventListener("change", (e) => {
      state.ablations = e.target.value; renderAll();
    });
    document.getElementById("order").addEventListener("change", (e) => {
      state.order = e.target.value; renderBars();
    });

    renderMetricCounts();
    renderAll();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
