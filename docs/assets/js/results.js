// The main results table.
//
// Reads assets/data/results.json, which docs/build_data.py copies straight out
// of `full_genome_table.py --json` -- the same run that emits the paper's
// LaTeX. Nothing here recomputes a metric; it sorts, filters, and formats.

(function () {
  const S = window.AssayLoop;

  // Column definitions. `fmt` is applied to the raw JSON value, so the percent
  // columns scale here rather than in the data -- the JSON stores fractions.
  const COLUMNS = [
    { key: "display", label: "Method", type: "text", align: "left",
      title: "Method name as it appears in the paper's table." },
    { key: "ef", label: "EF", digits: 2, arrow: "↑",
      title: "Hits found divided by the number expected from random selection, using the effective budget of in-library and invalid acquisitions." },
    { key: "nauc", label: "nAUC (%)", digits: 1, scale: 100, arrow: "↑",
      title: "Area under the cumulative-hits curve, normalized by a perfect hits-first oracle over the same effective budget." },
    { key: "frac", label: "FH (%)", digits: 1, scale: 100, arrow: "↑",
      title: "Fraction of all hits in the screen recovered within the acquisition budget." },
    { key: "shortfall", label: "SF (%)", digits: 1, scale: 100, arrow: "↓",
      title: "Fraction of acquired genes outside the screen's measured gene library." },
    { key: "pct_ess", label: "%ess", digits: 1, scale: 100,
      title: "Percentage of acquired hits that are DepMap common-essential genes." },
    { key: "ep_b", label: "EP-B", digits: 1,
      title: "Effective number of Reactome level-2 pathway groups in one acquisition batch, subsampled to 30 annotated genes." },
    { key: "ep_s", label: "EP-S", digits: 1,
      title: "Effective number of Reactome level-2 pathway groups across all picks in one screen, subsampled to 200 annotated genes." },
    { key: "ep_d", label: "EP-D", digits: 1,
      title: "Effective number of Reactome level-2 pathway groups across all picks in the test set, subsampled to 6,000 annotated genes." },
  ];

  // The rows the page argues from, by their full name in the JSON. Every
  // "(Ours)" row used to be highlighted, which meant eleven of fifty-six rows
  // were shaded and the shading stopped meaning anything. These six are the
  // comparison the paper actually makes: the two reference points anyone will
  // look for first (BPMF, Gemini-3.1-pro), the two components (AssayFormer,
  // AssayLLM), and the two systems built on them.
  const FEATURED = new Set([
    "Joint AssayLLM-AssayFormer GRPO",
    "Gemini-3.1-Pro → AssayFormer",
    "AssayFormer + BPMF + GRPO (= AssayFormer)",
    "Qwen3.6-27B (base) + SFT + GRPO (= AssayLLM)",
    "BPMF",
    "Gemini-3.1-pro",
  ]);

  const state = { rows: [], sortKey: null, sortDesc: true,
                  search: "", family: "", ablations: "hide",
                  unavailableNote: "" };

  function cellValue(row, col) {
    const v = row[col.key];
    if (col.type === "text") return v;
    if (v === null || v === undefined) return null;
    return col.scale ? v * col.scale : v;
  }

  function isAblationRow(row) {
    return row.family === "ablation" || row.display === "- hit labels";
  }

  function renderHead() {
    const head = document.getElementById("head");
    head.innerHTML = "";
    for (const col of COLUMNS) {
      const th = document.createElement("th");
      th.title = col.title;
      if (col.align === "left") th.className = "left";
      th.textContent = col.arrow ? `${col.label} ${col.arrow}` : col.label;
      const ind = document.createElement("span");
      ind.className = "sort-indicator";
      ind.textContent = state.sortKey === col.key ? (state.sortDesc ? " ▼" : " ▲") : "";
      th.appendChild(ind);
      th.addEventListener("click", () => {
        if (state.sortKey === col.key) {
          state.sortDesc = !state.sortDesc;
        } else {
          state.sortKey = col.key;
          // Text sorts A-Z first; numbers sort best-first, and for shortfall
          // "best" is small.
          state.sortDesc = col.type !== "text" && col.arrow !== "↓";
        }
        render();
      });
      head.appendChild(th);
    }
  }

  function visibleRows() {
    const q = state.search.trim().toLowerCase();
    return state.rows.filter((r) => {
      if (state.ablations === "hide" && isAblationRow(r)) return false;
      if (state.family && r.family !== state.family) return false;
      // Matched against the raw name, not the rendered one: the page shows a
      // real minus sign but nobody types U+2212 into a search box.
      if (q && !r.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }

  function sortRows(rows) {
    if (!state.sortKey) return rows;  // unsorted = the paper's own order
    const col = COLUMNS.find((c) => c.key === state.sortKey);
    const dir = state.sortDesc ? -1 : 1;
    return rows.slice().sort((a, b) => {
      const av = cellValue(a, col), bv = cellValue(b, col);
      // Nulls are "no value", not "worst value" -- park them at the bottom
      // whichever way the column is sorted, so flipping the arrow doesn't put
      // a row of dashes on top.
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      if (col.type === "text") return dir * String(av).localeCompare(String(bv));
      return dir * (av - bv);
    });
  }

  function render() {
    renderHead();
    const rows = sortRows(visibleRows());
    const body = document.getElementById("body");
    body.innerHTML = "";

    let section = null;
    for (const row of rows) {
      // Section headers only make sense in the paper's own order. Once the
      // reader sorts by a metric, the grouping is gone and printing headers
      // would imply a structure the rows no longer have.
      if (!state.sortKey && row.section !== section) {
        section = row.section;
        const tr = document.createElement("tr");
        tr.className = "family-row";
        const td = document.createElement("td");
        td.colSpan = COLUMNS.length;
        td.textContent = section;
        tr.appendChild(td);
        body.appendChild(tr);
      }

      const tr = document.createElement("tr");
      const unevaluated = row.available === false;
      if (unevaluated) tr.className = "unevaluated";
      else if (FEATURED.has(row.name)) tr.className = "highlight";
      for (const col of COLUMNS) {
        const td = document.createElement("td");
        if (col.type === "text") {
          // A continuation row reads as an operator on the row above it
          // ("+ GRPO", "− hit labels"). Giving it its own colour dot puts two
          // markers in front of the operator, which is one too many: the
          // parent directly above already shows the family colour, and the
          // indent already says whose child this is. Sorting breaks that
          // adjacency, so the dot comes back with the full name.
          const child = row.indent && !state.sortKey;
          td.className = child ? "left indent" : "left";
          const pill = document.createElement("span");
          pill.className = "model-pill";
          if (!child) {
            const sw = document.createElement("span");
            sw.className = "swatch";
            sw.style.background = S.familyColor(row.family);
            pill.appendChild(sw);
          }
          pill.appendChild(document.createTextNode(
            S.methodLabel(state.sortKey ? row.name : row.display)));
          pill.title = `${S.methodLabel(row.name)} — ${S.familyLabel(row.family)}`;
          td.appendChild(pill);
          // The paper's citation keys stay in results.json -- they are how a
          // row maps back to the LaTeX -- but they are not rendered. A bare
          // "Salakhutdinov2008-ad" next to a method name is a citation only to
          // someone holding the bibliography.
          if (unevaluated) {
            const tag = document.createElement("span");
            tag.className = "na-tag";
            tag.textContent = "not evaluated";
            tag.title = state.unavailableNote || "No result for this method in this build.";
            td.appendChild(tag);
          }
        } else {
          td.textContent = S.fmt(cellValue(row, col), col.digits);
          // Three different reasons produce the same em dash, and conflating
          // them is exactly the misreading this page is trying to avoid.
          if (unevaluated) {
            td.title = state.unavailableNote || "Not evaluated in this build.";
          } else if (row[col.key] === null && col.key.startsWith("ep_")) {
            td.title = "Below the rarefaction retention floor — not a zero.";
          } else if (row[col.key] === null) {
            td.title = "Not reported for this method.";
          }
        }
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }

    const total = state.rows.length;
    const shown = rows.length;
    document.getElementById("status").textContent =
      shown === total ? `${total} methods.`
                      : `Showing ${shown} of ${total} methods.`;
  }

  async function init() {
    const body = document.getElementById("body");
    let data;
    try {
      data = await S.fetchJSON("assets/data/results.json");
    } catch (err) {
      S.showError(document.getElementById("status"), err);
      body.innerHTML = "";
      return;
    }
    state.rows = data.rows;

    document.getElementById("n-methods").textContent = data.rows.length;
    document.getElementById("n-screens").textContent = data.n_screens;

    // build_data.py refuses to emit a row with no result unless whoever ran it
    // named the method and wrote this note, so if `names` is non-empty the note
    // is there too. Say it at the top of the page rather than only in tooltips.
    const gap = data.unavailable;
    if (gap && gap.names && gap.names.length) {
      state.unavailableNote = gap.note || "";
      const el = document.getElementById("unavailable-note");
      // Built as nodes, not innerHTML: the note and the method names come from
      // a JSON file, and this is the one place on the page that renders
      // free text somebody typed on a command line.
      const lead = document.createElement("strong");
      lead.textContent =
        `${gap.names.length} of ${data.rows.length} methods have no numbers ` +
        `on this build.`;
      el.appendChild(lead);
      el.appendChild(document.createTextNode(
        ` ${gap.note || ""} The rows are still listed, greyed and tagged ` +
        `"not evaluated": ${gap.names.join(", ")}.`));
      el.hidden = false;
    }

    const fam = document.getElementById("family");
    for (const f of S.FAMILIES) {
      if (!data.rows.some((r) => r.family === f.key)) continue;
      const opt = document.createElement("option");
      opt.value = f.key;
      opt.textContent = f.label;
      fam.appendChild(opt);
    }

    const notes = document.getElementById("notes");
    for (const col of COLUMNS) {
      if (col.type === "text") continue;
      const li = document.createElement("li");
      li.innerHTML = `<strong>${col.label}:</strong> ${col.title}`;
      notes.appendChild(li);
    }

    document.getElementById("search").addEventListener("input",
      S.debounce((e) => { state.search = e.target.value; render(); }, 120));
    fam.addEventListener("change", (e) => { state.family = e.target.value; render(); });
    document.getElementById("ablations").addEventListener("change",
      (e) => { state.ablations = e.target.value; render(); });
    document.getElementById("reset").addEventListener("click", () => {
      Object.assign(state, { sortKey: null, sortDesc: true, search: "",
                             family: "", ablations: "hide" });
      document.getElementById("search").value = "";
      fam.value = "";
      document.getElementById("ablations").value = "hide";
      render();
    });

    render();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
