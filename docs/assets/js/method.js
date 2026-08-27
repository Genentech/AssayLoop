// The pipeline figure, panel by panel.
//
// assets/data/panels.json is written by docs/build_paper_panels.py, which cuts
// the paper's composed figure along its own panel boxes. Panel captions live
// there too, next to the crop they describe.

(function () {
  const S = window.AssayLoop;

  // Each panel is its own headed subsection, stacked one per row. The heading
  // is what makes the order readable: eight cards two to a row leave the reader
  // to guess whether the sequence runs across or down.
  function panel(p) {
    return `<section class="panel" id="panel-${p.letter.toLowerCase()}">
      <h3 class="panel-head">
        <span class="panel-letter">${p.letter}</span>
        <span>${p.title}</span>
      </h3>
      <figure>
        <img src="${p.image}" alt="Pipeline panel ${p.letter}: ${p.title}" loading="lazy"/>
        <figcaption>${p.caption}</figcaption>
      </figure>
    </section>`;
  }

  async function init() {
    const node = document.getElementById("panels");
    const status = document.getElementById("panels-status");
    if (!node) return;

    let data;
    try {
      data = await S.fetchJSON("assets/data/panels.json?v=8310dc47");
    } catch (err) {
      S.showError(node, err);
      return;
    }

    const panels = (data.panels || []).filter((p) =>
      p.key.startsWith("pipeline_panel_"));
    if (!panels.length) {
      // Not an empty section with no explanation: build_paper_panels.py fails
      // loudly when the figure's layout changes, and this is what that looks
      // like from the page's side.
      S.showError(node, new Error(
        "panels.json has no pipeline panels. Run " +
        "`python docs/build_paper_panels.py --only pipeline`."));
      return;
    }
    // A panel with no heading would render as an untitled subsection, which is
    // the thing the headings exist to prevent. Say what to re-run instead.
    const untitled = panels.filter((p) => !p.title).map((p) => p.letter);
    if (untitled.length) {
      S.showError(node, new Error(
        `panels.json has no title for panel(s) ${untitled.join(", ")}. ` +
        "Run `python docs/build_paper_panels.py --only pipeline`."));
      return;
    }

    node.innerHTML = panels.map(panel).join("");
    if (status) {
      status.textContent =
        `${panels.length} panels, cropped from the paper's Figure 1.`;
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
