// Shared chrome + utilities for the AssayLoop Pages site.
//
// Ported from the AssayBench site so the two project pages behave the same
// way. Every page injects the header/footer from here and pulls its data
// through `fetchJSON`, so a missing or malformed data file surfaces as a
// visible error rather than a silently empty chart.

(function () {
  const NAV_LINKS = [
    { href: "index.html", label: "Overview", match: ["", "index.html"] },
    { href: "method.html", label: "Method", match: ["method.html"] },
    { href: "results.html", label: "Results", match: ["results.html"] },
    { href: "recovery.html", label: "Recovery curves", match: ["recovery.html"] },
    { href: "diversity.html", label: "Diversity", match: ["diversity.html"] },
    { href: "umap.html", label: "Gene embeddings", match: ["umap.html"] },
    { href: "analysis.html", label: "Analysis", match: ["analysis.html"] },
    { href: "cite.html", label: "Citation", match: ["cite.html"] },
  ];

  // Method families. Keep in sync with the --fam-* custom properties in
  // site.css and with the section grouping in full_genome_table.py.
  const FAMILIES = [
    { key: "llm", label: "Base LLMs", color: "#4C90D9" },
    { key: "assayllm", label: "AssayLLM (ours)", color: "#7C5CBF" },
    { key: "classical", label: "Adaptive experimental design", color: "#8a8880" },
    { key: "agent", label: "Agent harnesses", color: "#d97706" },
    { key: "assayformer", label: "AssayFormer (ours)", color: "#1FA187" },
    { key: "assayloop", label: "AssayLoop (ours)", color: "#c2410c" },
    { key: "ablation", label: "Ablations", color: "#9ca3af" },
  ];
  const FAMILY_BY_KEY = {};
  for (const f of FAMILIES) FAMILY_BY_KEY[f.key] = f;

  function currentSegment() {
    return window.location.pathname.split("/").pop() || "";
  }

  function injectHeader() {
    const placeholder = document.getElementById("site-header");
    if (!placeholder) return;
    const current = currentSegment();
    placeholder.innerHTML = `
      <header class="site-header">
        <div class="site-header-inner">
          <a href="index.html" class="site-brand">
            AssayLoop
            <span class="site-brand-sub">Amortized adaptive hit discovery in CRISPR screens</span>
          </a>
          <nav class="site-nav">
            ${NAV_LINKS.map((link) => {
              const active = link.match.includes(current) ? " class=\"active\"" : "";
              return `<a href=\"${link.href}\"${active}>${link.label}</a>`;
            }).join("")}
          </nav>
        </div>
      </header>
    `;
  }

  function injectFooter() {
    const placeholder = document.getElementById("site-footer");
    if (!placeholder) return;
    placeholder.innerHTML = `
      <footer class="site-footer">
        <div class="site-footer-inner">
          <div>AssayLoop &nbsp;&middot;&nbsp; Genentech, 2026
            &nbsp;&middot;&nbsp; <a href="https://github.com/Genentech/AssayLoop">GitHub</a>
            &nbsp;&middot;&nbsp; <a href="https://pypi.org/project/assaybench/">PyPI</a>
            &nbsp;&middot;&nbsp; <a href="https://genentech.github.io/AssayBench">AssayBench</a>
          </div>
          <div>Figures and numbers from
            <a href="cite.html">Biology-in-the-loop: Amortized Adaptive Hit Discovery in
            CRISPR Screens</a>.</div>
        </div>
      </footer>
    `;
  }

  // ---------------------------------------------------------------- helpers

  // Every page tags its <script src> with the build stamp that
  // `docs/stamp_assets.py` wrote, and carries a second stamp -- `data-build`,
  // a hash over assets/data/ -- in an attribute. Read the data one back off
  // our own tag and put it on the JSON fetches: without it a browser that has
  // cached last week's results.json keeps serving it, and the page looks stale
  // in a way that is indistinguishable from a build that never ran. It has to
  // be its own stamp rather than this file's, because a data rebuild does not
  // touch site.js and so would not move site.js's content hash.
  const BUILD = (function () {
    const tag = document.currentScript;
    if (tag && tag.dataset.build) return tag.dataset.build;
    const m = tag && tag.src && tag.src.match(/[?&]v=([^&]+)/);
    return m ? m[1] : "";
  })();

  function versioned(path) {
    if (!BUILD || /^https?:/.test(path)) return path;
    return path + (path.includes("?") ? "&" : "?") + "v=" + BUILD;
  }

  async function fetchJSON(path) {
    const resp = await fetch(versioned(path));
    if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
    return resp.json();
  }

  /** Render an error into `node` instead of leaving a blank panel behind. */
  function showError(node, err) {
    if (!node) throw err;
    node.className = "error-state";
    node.textContent =
      `${err.message}\n\n` +
      "This page is built from JSON in docs/assets/data/. If you are running " +
      "the site locally, run docs/build_data.py first.";
    // Still surface it in the console for anyone debugging.
    console.error(err);
  }

  const PLOTLY_THEME = {
    font: { family: "Inter, Segoe UI, sans-serif", color: "#1b1f24", size: 12 },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    margin: { l: 62, r: 24, t: 36, b: 64 },
    hoverlabel: { bgcolor: "#ffffff", bordercolor: "#d1d5db", font: { family: "Inter, sans-serif", size: 12, color: "#1b1f24" } },
    xaxis: { gridcolor: "rgba(128,128,128,0.18)", linecolor: "#d1d5db", zerolinecolor: "rgba(128,128,128,0.35)" },
    yaxis: { gridcolor: "rgba(128,128,128,0.18)", linecolor: "#d1d5db", zerolinecolor: "rgba(128,128,128,0.35)" },
    legend: { bgcolor: "rgba(255,255,255,0.7)", bordercolor: "#e5e7eb", borderwidth: 1 },
  };

  const PLOTLY_CONFIG = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  };

  function mergeLayout(layout) {
    layout = layout || {};
    return Object.assign({}, PLOTLY_THEME, layout, {
      xaxis: Object.assign({}, PLOTLY_THEME.xaxis, layout.xaxis || {}),
      yaxis: Object.assign({}, PLOTLY_THEME.yaxis, layout.yaxis || {}),
      margin: Object.assign({}, PLOTLY_THEME.margin, layout.margin || {}),
      hoverlabel: Object.assign({}, PLOTLY_THEME.hoverlabel, layout.hoverlabel || {}),
      legend: Object.assign({}, PLOTLY_THEME.legend, layout.legend || {}),
    });
  }

  function indexBy(records, key) {
    const out = {};
    for (const record of records) out[record[key]] = record;
    return out;
  }

  /** Format a number, rendering null/undefined/NaN as an em dash.
   *
   * The em dash is load-bearing: the Effective Pathways metric returns null
   * for a scope whose annotated-gene retention falls below its floor, and the
   * paper prints those cells as `--`. Never substitute a number there. */
  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toFixed(digits === undefined ? 3 : digits);
  }

  /** Render a method name for the page.
   *
   * The table's continuation rows are ablations written as operators on the
   * row above -- "+ SFT", "+ GRPO", "- hit labels". The JSON keeps the ASCII
   * hyphen the LaTeX table uses, so the two stay diffable, but a hyphen next
   * to a "+" reads as punctuation rather than as its opposite. Swap it for a
   * real minus sign (U+2212), which is the same width as the plus and sits at
   * the same height. Only a hyphen that stands alone as a word is touched;
   * "Qwen3.6-27B" and "Gemini-3.1-Pro" keep theirs. */
  function methodLabel(text) {
    return String(text).replace(/(^|\s)-(\s)/g, "$1−$2");
  }

  function familyColor(key) {
    return (FAMILY_BY_KEY[key] || {}).color || "#8a8880";
  }

  function familyLabel(key) {
    return (FAMILY_BY_KEY[key] || {}).label || key;
  }

  function setActiveTag(container, value) {
    Array.from(container.querySelectorAll(".tag")).forEach((node) => {
      node.classList.toggle("active", node.dataset.value === value);
    });
  }

  function debounce(fn, ms) {
    let handle = null;
    return function (...args) {
      clearTimeout(handle);
      handle = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  /** Wire up every `.copy-btn` whose `data-copy` holds the text to copy. */
  function initCopyButtons(root) {
    (root || document).querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(btn.dataset.copy || "");
          const original = btn.textContent;
          btn.textContent = "copied";
          setTimeout(() => { btn.textContent = original; }, 1200);
        } catch (err) {
          console.error(err);
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    injectHeader();
    injectFooter();
  });

  window.AssayLoop = {
    BUILD,
    FAMILIES,
    fetchJSON,
    versioned,
    showError,
    mergeLayout,
    PLOTLY_CONFIG,
    indexBy,
    fmt,
    methodLabel,
    familyColor,
    familyLabel,
    setActiveTag,
    debounce,
    initCopyButtons,
  };
})();
