// Headline numbers, for any page carrying [data-stat] nodes.
//
// The stat grid and the two figures in the TL;DR are written into the HTML
// as the paper's values, so the page reads correctly with JavaScript off and
// before the fetch resolves. This script then overwrites them from
// assets/data/summary.json, which build_data.py derives from the same table
// run that results.html and diversity.html read. If a rerun moves the best EF,
// every place the site reports the headline enrichment moves with it.
//
// If summary.json cannot be loaded, the static values stay on screen and a
// visible note says so. That is the one case where the site shows a number it
// did not just derive, and it is not allowed to be silent about it.

(function () {
  const S = window.AssayLoop;

  /** Fill every [data-stat] node, and report the ones summary.json has no value for. */
  function apply(summary) {
    const value = {
      best_ef: (s) => `${s.best_ef.value.toFixed(2)}×`,
      best_frac_hits: (s) => `${(s.best_frac_hits.value * 100).toFixed(1)}%`,
      n_methods: (s) => s.n_methods.toLocaleString(),
      n_test_screens: (s) => String(s.n_test_screens),
      rounds_by_batch: (s) => `${s.n_rounds} × ${s.batch_size}`,
    };
    const unresolved = [];
    for (const node of document.querySelectorAll("[data-stat]")) {
      const key = node.dataset.stat;
      const fn = value[key];
      if (!fn) { unresolved.push(key); continue; }
      let text;
      try {
        text = fn(summary);
      } catch (err) {
        // A null or absent field, not a broken selector: name the key rather
        // than writing "undefined" into the headline.
        unresolved.push(key);
        continue;
      }
      node.textContent = text;
    }
    return unresolved;
  }

  function note(message) {
    const el = document.getElementById("summary-note");
    if (!el) return;
    el.hidden = false;
    el.textContent = message;
  }

  async function init() {
    let summary;
    try {
      summary = await S.fetchJSON("assets/data/summary.json");
    } catch (err) {
      console.error(err);
      note("The numbers below are the paper's published values: " +
           "assets/data/summary.json could not be loaded, so they are not " +
           "being read back from the results table. Run docs/build_data.py.");
      return;
    }
    const unresolved = apply(summary);
    if (unresolved.length) {
      note("Still showing the paper's published values for: " +
           unresolved.join(", ") + ". summary.json has no field for them.");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
