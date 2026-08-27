// Paper figures, dropped into whichever section is about them.
//
// A page marks a spot with `<div data-figures="lopo,scaling"></div>` and this
// fills it from assets/data/figures.json, in the order the attribute lists.
// One renderer for every page, so the markup of a figure card exists once.
//
// A slot may add `data-figure-width="520"` to cap the card. Figures are not one
// shape: `label_ablation` is a seven-row dumbbell that wants the full column,
// `embedding_drift` is a 1744x3345 portrait that at full column width is two
// screens tall. Sizing belongs with the figure, not with the grid.
//
// The two failure modes are both stated on the page rather than swallowed. A
// key that build_figures.py recorded as a failure renders as a card saying the
// script did not produce it; a key that is in neither the figures nor the
// failures is a typo in the HTML, and says that instead of rendering nothing.

(function () {
  const S = window.AssayLoop;
  const SRC = "assets/data/figures.json?v=f4adcd1a";

  function escapeAttr(text) {
    return String(text).replace(/&/g, "&amp;").replace(/"/g, "&quot;")
                       .replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function card(fig) {
    // The caption text is authored in build_figures.py and contains HTML
    // entities on purpose (&ndash;, &kappa;), so it goes in unescaped. The
    // alt text is an attribute and does not.
    //
    // Neither the image nor the caption links to the vector PDF. The PDF is
    // still built and still shipped in assets/figures/ -- and figures.json
    // still records its path -- but a "Vector PDF" line under every caption is
    // a row of repeated chrome on a page whose job is to show the figures. The
    // people who want the vector file are reading the repo, not the site.
    const img = fig.image
      ? `<img src="${fig.image}" alt="${escapeAttr(fig.title)}" loading="lazy"/>`
      : "";
    return `<figure class="fig-card">
      ${img}
      <figcaption>
        <span class="fig-section">${fig.section}</span>
        <strong>${fig.title}</strong> ${fig.caption}
      </figcaption>
    </figure>`;
  }

  function missing(key, detail) {
    return `<figure class="fig-card fig-card-missing">
      <figcaption>
        <span class="fig-section">Not available</span>
        <strong>${escapeAttr(key)}</strong> ${escapeAttr(detail)}
      </figcaption>
    </figure>`;
  }

  async function init() {
    const slots = Array.from(document.querySelectorAll("[data-figures]"));
    if (!slots.length) return;

    let data;
    try {
      data = await S.fetchJSON(SRC);
    } catch (err) {
      for (const slot of slots) S.showError(slot, err);
      return;
    }

    const byKey = S.indexBy(data.figures, "key");
    const failed = S.indexBy(data.failures || [], "key");

    for (const slot of slots) {
      const keys = slot.dataset.figures.split(",").map((k) => k.trim())
                                                  .filter(Boolean);
      slot.classList.add("figure-strip");
      if (slot.dataset.figureWidth) {
        slot.style.maxWidth = `${slot.dataset.figureWidth}px`;
      }
      slot.innerHTML = keys.map((key) => {
        if (byKey[key]) return card(byKey[key]);
        if (failed[key]) {
          return missing(key,
            "This figure's script did not run on the build that made this " +
            "site, so there is nothing to show here. It is in the paper.");
        }
        console.error(`figure-strip: no figure or failure named "${key}"`);
        return missing(key,
          "This page asked for a figure that figures.json does not have. " +
          "That is a bug in the site, not a missing result.");
      }).join("");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
