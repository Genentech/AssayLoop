// The BPMF gene-embedding explorer.
//
// Reads assets/data/gene_umap.json, written by docs/build_umap.py. Every gene
// in that file is drawn. Points whose annotation is blank are drawn grey and
// labelled "unassigned" rather than filtered out -- for CORUM and Reactome
// that is most of the space, and hiding them would make the annotation look
// far more complete than it is, which is the opposite of the figure's point.

(function () {
  const S = window.AssayLoop;

  // Categorical colourings share one palette; continuous ones have their own
  // scale. `kind` decides which branch of buildTraces runs.
  const COLOURINGS = {
    cluster_label: { kind: "category", label: "HDBSCAN cluster" },
    complex: { kind: "category", label: "CORUM complex family" },
    pathway: { kind: "category", label: "Reactome pathway" },
    hit_rate: { kind: "continuous", label: "Hit rate" },
    essential: { kind: "boolean", label: "DepMap common-essential" },
  };

  // Okabe-Ito plus the site's accents: distinguishable without relying on
  // hue alone being read correctly, and stable across colourings.
  const PALETTE = [
    "#4C90D9", "#1FA187", "#d97706", "#7C5CBF", "#c2410c",
    "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#E69F00",
    "#8a8880", "#a3562f",
  ];
  const UNASSIGNED = "#d1d5db";
  const HIGHLIGHT = "#111827";

  // Categorical legends get unreadable past this many entries. The rest are
  // pooled into "other", and the status line says how many were pooled --
  // an unlabelled "other" bucket is how a chart quietly loses a category.
  const MAX_CATEGORIES = 12;

  const state = { colour: "cluster_label", minScreens: 0, search: "" };
  let DATA = null;

  function shortLabel(name) {
    const trimmed = String(name).replace(/^REACTOME_/, "").replace(/_/g, " ");
    return trimmed.length <= 42 ? trimmed : trimmed.slice(0, 39) + "…";
  }

  /** Genes the "highlight" box names, uppercased and de-duplicated. */
  function searchSet() {
    const names = state.search.split(/[\s,;]+/).map((s) => s.trim().toUpperCase());
    return new Set(names.filter(Boolean));
  }

  function visibleGenes() {
    return DATA.genes.filter((g) => g.screens >= state.minScreens);
  }

  function hoverText(g) {
    const parts = [
      `<b>${g.gene}</b>`,
      `${g.cluster_label}`,
      `hit rate ${(g.hit_rate * 100).toFixed(2)}% (${g.hits} / ${g.screens} screens)`,
    ];
    if (g.essential) parts.push("DepMap common-essential");
    if (g.complex) parts.push(`CORUM: ${g.complex}`);
    if (g.pathway) parts.push(`Reactome: ${shortLabel(g.pathway)}`);
    return parts.join("<br>");
  }

  const BASE_MARKER = { size: 3.5, opacity: 0.72 };

  function categoryTraces(genes, field) {
    const counts = new Map();
    for (const g of genes) {
      const v = g[field] || "";
      counts.set(v, (counts.get(v) || 0) + 1);
    }
    // "" is the unassigned bucket and is never ranked into the top-N; it gets
    // its own trace with its own colour so it can't be mistaken for a group.
    const ranked = [...counts.entries()]
      .filter(([v]) => v !== "")
      .sort((a, b) => b[1] - a[1]);
    const top = ranked.slice(0, MAX_CATEGORIES).map(([v]) => v);
    const topSet = new Set(top);
    const pooled = ranked.slice(MAX_CATEGORIES);

    const buckets = new Map(top.map((v) => [v, []]));
    const other = [];
    const unassigned = [];
    for (const g of genes) {
      const v = g[field] || "";
      if (v === "") unassigned.push(g);
      else if (topSet.has(v)) buckets.get(v).push(g);
      else other.push(g);
    }

    const traces = top.map((v, i) => trace(buckets.get(v), shortLabel(v), PALETTE[i % PALETTE.length]));
    if (other.length) {
      traces.push(trace(other, `other (${pooled.length} groups)`, "#8a8880"));
    }
    if (unassigned.length) {
      traces.push(trace(unassigned, `unassigned (${unassigned.length})`, UNASSIGNED));
    }
    return { traces, pooled: pooled.length, unassigned: unassigned.length };
  }

  function trace(genes, name, color) {
    return {
      type: "scattergl",
      mode: "markers",
      name,
      x: genes.map((g) => g.x),
      y: genes.map((g) => g.y),
      text: genes.map(hoverText),
      hoverinfo: "text",
      marker: Object.assign({ color }, BASE_MARKER),
    };
  }

  function continuousTrace(genes) {
    return [{
      type: "scattergl",
      mode: "markers",
      name: "hit rate",
      x: genes.map((g) => g.x),
      y: genes.map((g) => g.y),
      text: genes.map(hoverText),
      hoverinfo: "text",
      marker: Object.assign({
        color: genes.map((g) => g.hit_rate),
        colorscale: "Viridis",
        // Hit rates are long-tailed; without a cap the top few genes
        // compress everything else into the bottom of the scale.
        cmin: 0,
        cmax: 0.05,
        colorbar: { title: { text: "hit rate", side: "right" }, thickness: 12 },
        showscale: true,
      }, BASE_MARKER),
    }];
  }

  function booleanTraces(genes) {
    const yes = genes.filter((g) => g.essential);
    const no = genes.filter((g) => !g.essential);
    return [
      trace(no, `not essential (${no.length})`, UNASSIGNED),
      trace(yes, `common-essential (${yes.length})`, "#c2410c"),
    ];
  }

  function highlightTrace(genes) {
    const wanted = searchSet();
    if (!wanted.size) return { trace: null, found: [], missing: [] };
    const hits = genes.filter((g) => wanted.has(g.gene.toUpperCase()));
    const foundNames = new Set(hits.map((g) => g.gene.toUpperCase()));
    const missing = [...wanted].filter((n) => !foundNames.has(n));
    if (!hits.length) return { trace: null, found: [], missing };
    return {
      trace: {
        type: "scattergl",
        mode: "markers+text",
        name: "highlighted",
        x: hits.map((g) => g.x),
        y: hits.map((g) => g.y),
        text: hits.map((g) => g.gene),
        textposition: "top center",
        textfont: { size: 11, color: HIGHLIGHT },
        hovertext: hits.map(hoverText),
        hoverinfo: "text",
        marker: { size: 11, color: "rgba(0,0,0,0)", line: { color: HIGHLIGHT, width: 2 } },
      },
      found: hits,
      missing,
    };
  }

  function render() {
    const genes = visibleGenes();
    const spec = COLOURINGS[state.colour];
    let traces;
    let note = "";

    if (spec.kind === "continuous") {
      traces = continuousTrace(genes);
    } else if (spec.kind === "boolean") {
      traces = booleanTraces(genes);
    } else {
      const built = categoryTraces(genes, state.colour);
      traces = built.traces;
      const bits = [];
      if (built.pooled) bits.push(`${built.pooled} smaller groups pooled into "other"`);
      if (built.unassigned) {
        bits.push(`${built.unassigned} genes (${(100 * built.unassigned / genes.length).toFixed(0)}%) have no ${spec.label.toLowerCase()}`);
      }
      note = bits.join("; ");
    }

    const hl = highlightTrace(genes);
    if (hl.trace) traces.push(hl.trace);

    Plotly.react("scatter", traces, S.mergeLayout({
      xaxis: { title: { text: "UMAP 1" }, zeroline: false, showticklabels: false },
      yaxis: { title: { text: "UMAP 2" }, zeroline: false, showticklabels: false,
               scaleanchor: "x", scaleratio: 1 },
      legend: { itemsizing: "constant", font: { size: 11 } },
      hovermode: "closest",
    }), S.PLOTLY_CONFIG);

    const hidden = DATA.genes.length - genes.length;
    document.getElementById("status").textContent =
      `${genes.length.toLocaleString()} of ${DATA.genes.length.toLocaleString()} genes` +
      (hidden ? ` (${hidden.toLocaleString()} below the screen threshold)` : "") +
      (note ? `. ${note}.` : ".");

    const found = document.getElementById("found");
    if (!state.search.trim()) {
      found.textContent = "";
    } else {
      const parts = [`${hl.found.length} highlighted`];
      if (hl.missing.length) {
        // Named, not silently dropped: "TP53 isn't here" is a fact about the
        // embedding (or a typo), and either way the reader should see it.
        parts.push(`not in the embedding: ${hl.missing.join(", ")}`);
      }
      found.textContent = parts.join(" · ");
    }
  }

  async function init() {
    const status = document.getElementById("status");
    try {
      DATA = await S.fetchJSON("assets/data/gene_umap.json");
    } catch (err) {
      S.showError(status, err);
      return;
    }

    document.getElementById("meta").textContent =
      `${DATA.n_genes.toLocaleString()} genes · K = ${DATA.k} · ${DATA.projection} · seed ${DATA.seed}`;

    document.getElementById("colour").addEventListener("change", (e) => {
      state.colour = e.target.value;
      render();
    });
    document.getElementById("min-screens").addEventListener("input",
      S.debounce((e) => {
        state.minScreens = Number(e.target.value) || 0;
        render();
      }, 200));
    document.getElementById("gene-search").addEventListener("input",
      S.debounce((e) => {
        state.search = e.target.value;
        render();
      }, 200));

    S.initCopyButtons(document);
    render();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
