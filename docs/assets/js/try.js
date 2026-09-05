(function () {
  "use strict";

  const apiBase = (document.querySelector('meta[name="assayloop-api-base"]')?.content || "")
    .trim().replace(/\/$/, "");
  const apiURL = (path) => apiBase ? `${apiBase}${path}` : path.replace(/^\//, "");
  const $ = (id) => document.getElementById(id);
  let examples = [];
  let latestRows = [];
  let selectedExampleId = "";
  let customScreensEnabled = false;
  let tourIndex = -1;
  let tourPositionTimer = null;

  const DESCRIPTION_INPUTS = [
    "phenotype", "cell-line", "cell-type", "organism", "condition", "description",
  ];
  const TOUR_STEPS = [
    {
      target: "example-select",
      title: "Start with a public example",
      copy: "Choose any of the five public screens. The first example is already loaded for you.",
    },
    {
      target: "observations",
      title: "Review the feedback",
      copy: "These hit and non-hit results come from the first two batches. You can edit them or leave them as they are.",
    },
    {
      target: "rank-button",
      title: "Rank the next batch",
      copy: "Click this button to run AssayFormer in your browser and fill the predicted-gene table.",
    },
  ];

  async function jsonRequest(path, options) {
    const response = await fetch(apiURL(path), options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* use status below */ }
    if (!response.ok) {
      throw new Error(body?.detail || `Inference service returned ${response.status}.`);
    }
    return body;
  }

  function parseBoolean(value) {
    const normalized = String(value).trim().toLowerCase();
    if (["1", "true", "hit", "yes", "positive", "+"].includes(normalized)) return true;
    if (["0", "false", "non-hit", "nonhit", "no", "negative", "-"].includes(normalized)) return false;
    return null;
  }

  function splitLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return [];
    if (trimmed.includes("\t")) return trimmed.split("\t").map((v) => v.trim());
    if (trimmed.includes(",")) return trimmed.split(",").map((v) => v.trim());
    return trimmed.split(/\s+/);
  }

  function parseObservations() {
    const mode = $("value-mode").value;
    const threshold = Number($("threshold").value);
    const direction = $("direction").value;
    const rows = [];
    const errors = [];
    const seen = new Set();
    const lines = $("observations").value.split(/\r?\n/);

    lines.forEach((line, index) => {
      const columns = splitLine(line);
      if (!columns.length) return;
      if (index === 0 && /^(gene|symbol|gene_symbol)$/i.test(columns[0])) return;
      const gene = columns[0].toUpperCase();
      if (!/^[A-Z0-9][A-Z0-9._-]*$/.test(gene)) {
        errors.push(`line ${index + 1}: invalid gene symbol`);
        return;
      }
      if (seen.has(gene)) return;
      seen.add(gene);

      let hit = false;
      if (mode === "binary") {
        if (columns.length < 2) {
          errors.push(`line ${index + 1}: missing hit label`);
          return;
        }
        hit = parseBoolean(columns[1]);
        if (hit === null) {
          errors.push(`line ${index + 1}: use hit/non-hit or 1/0`);
          return;
        }
      } else if (mode === "continuous") {
        const score = Number(columns[1]);
        if (columns.length < 2 || !Number.isFinite(score) || !Number.isFinite(threshold)) {
          errors.push(`line ${index + 1}: invalid score or threshold`);
          return;
        }
        hit = direction === "high" ? score >= threshold : score <= threshold;
      }
      rows.push({ gene, hit });
    });

    if (rows.length > 1024) errors.push("at most 1,024 unique observations are supported");
    return { rows: rows.slice(0, 1024), errors };
  }

  function updateParseSummary() {
    const parsed = parseObservations();
    const hits = parsed.rows.filter((row) => row.hit).length;
    let message = parsed.rows.length
      ? `${parsed.rows.length} unique genes · ${hits} hits · ${parsed.rows.length - hits} non-hits`
      : "No observations pasted; cold-start ranking is supported.";
    if (parsed.errors.length) message += ` · ${parsed.errors.length} problem(s)`;
    $("parse-summary").textContent = message;
    $("parse-summary").classList.toggle("has-error", parsed.errors.length > 0);
  }

  function setValue(id, value) {
    $(id).value = value == null ? "" : value;
  }

  function setDescriptionEditable(enabled) {
    DESCRIPTION_INPUTS.forEach((id) => { $(id).readOnly = !enabled; });
    $("methodology").disabled = !enabled;
    const customOption = $("example-select").querySelector('option[value="custom"]');
    customOption.disabled = !enabled;
    $("description-mode").textContent = enabled
      ? "A text-embedding-3-small endpoint is configured, so you may select Custom screen and edit these fields."
      : "Public examples use released text embeddings and run in your browser. The first ranking downloads the 20 MB model; configure an embedding endpoint to enter a different screen.";
  }

  function clearScreen() {
    ["phenotype", "cleaned-phenotype", "cell-line", "cell-type", "condition", "contrast-label", "description"]
      .forEach((id) => setValue(id, ""));
    setValue("methodology", "");
    setValue("organism", "Homo sapiens");
    setValue("observations", "");
    selectedExampleId = "";
    $("example-source").textContent = "No public example is selected.";
    updateParseSummary();
  }

  function applyExample(index) {
    const example = examples[index];
    if (!example) return;
    selectedExampleId = example.screen_id;
    const screen = example.screen || {};
    setValue("phenotype", screen.phenotype);
    setValue("cleaned-phenotype", screen.cleaned_phenotype);
    setValue("cell-line", screen.cell_line);
    setValue("cell-type", screen.cell_type);
    setValue("methodology", screen.library_methodology);
    setValue("organism", screen.organism);
    setValue("condition", screen.condition_clause);
    setValue("contrast-label", screen.contrast_label);
    setValue("description", screen.description);
    setValue("observations", example.paste);
    setValue("value-mode", "binary");
    $("example-source").textContent = `Selected example source: ${example.source}.`;
    toggleThreshold();
    updateParseSummary();
  }

  function activeTourTarget() {
    return tourIndex < 0 ? null : $(TOUR_STEPS[tourIndex].target);
  }

  function positionTour() {
    const target = activeTourTarget();
    const popover = $("tour-popover");
    if (!target || popover.hidden) return;
    const targetRect = target.getBoundingClientRect();
    const popoverRect = popover.getBoundingClientRect();
    const margin = 12;
    const gap = 20;
    const roomBelow = window.innerHeight - targetRect.bottom;
    const placement = roomBelow >= popoverRect.height + gap + margin ? "bottom" : "top";
    const desiredLeft = targetRect.left + targetRect.width / 2 - popoverRect.width / 2;
    const left = Math.max(
      margin,
      Math.min(desiredLeft, window.innerWidth - popoverRect.width - margin)
    );
    const desiredTop = placement === "bottom"
      ? targetRect.bottom + gap
      : targetRect.top - popoverRect.height - gap;
    const top = Math.max(
      margin,
      Math.min(desiredTop, window.innerHeight - popoverRect.height - margin)
    );
    popover.dataset.placement = placement;
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    const arrowCenter = targetRect.left + targetRect.width / 2 - left - 8;
    $("tour-arrow").style.left =
      `${Math.max(20, Math.min(arrowCenter, popoverRect.width - 36))}px`;
  }

  function showTourStep(index) {
    document.querySelectorAll(".is-tour-target").forEach((node) => {
      node.classList.remove("is-tour-target");
    });
    tourIndex = Math.max(0, Math.min(index, TOUR_STEPS.length - 1));
    const step = TOUR_STEPS[tourIndex];
    const target = $(step.target);
    $("tour-progress").textContent = `Step ${tourIndex + 1} of ${TOUR_STEPS.length}`;
    $("tour-title").textContent = step.title;
    $("tour-copy").textContent = step.copy;
    $("tour-back").hidden = tourIndex === 0;
    $("tour-next").textContent = tourIndex === TOUR_STEPS.length - 1 ? "Done" : "Next";
    $("tour-scrim").hidden = false;
    $("tour-popover").hidden = false;
    target.classList.add("is-tour-target");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    window.clearTimeout(tourPositionTimer);
    window.requestAnimationFrame(positionTour);
    tourPositionTimer = window.setTimeout(positionTour, 350);
    $("tour-popover").focus({ preventScroll: true });
  }

  function closeTour({ focusTarget = false } = {}) {
    const target = activeTourTarget();
    window.clearTimeout(tourPositionTimer);
    document.querySelectorAll(".is-tour-target").forEach((node) => {
      node.classList.remove("is-tour-target");
    });
    $("tour-scrim").hidden = true;
    $("tour-popover").hidden = true;
    tourIndex = -1;
    if (focusTarget && target) target.focus({ preventScroll: true });
  }

  function nextTourStep() {
    if (tourIndex >= TOUR_STEPS.length - 1) closeTour({ focusTarget: true });
    else showTourStep(tourIndex + 1);
  }

  function toggleThreshold() {
    const show = $("value-mode").value === "continuous";
    $("threshold-field").classList.toggle("is-hidden", !show);
    $("direction-field").classList.toggle("is-hidden", !show);
    updateParseSummary();
  }

  function renderResults(payload) {
    latestRows = payload.results || [];
    const body = $("results-body");
    body.replaceChildren();
    latestRows.forEach((row) => {
      const tr = document.createElement("tr");
      const values = [
        row.rank,
        row.gene,
        Number(row.score).toFixed(5),
        `${(100 * Number(row.batch_inclusion_probability)).toFixed(1)}%`,
        row.retrospective_hit === true ? "hit" : (row.retrospective_hit === false ? "non-hit" : "—"),
        row.public_screens == null ? "—" : `${row.public_hits ?? 0} / ${row.public_screens}`,
        row.common_essential ? "yes" : "no",
        row.pathway || "—",
        row.complex || "—",
      ];
      values.forEach((value, index) => {
        const td = document.createElement("td");
        td.textContent = value;
        if (index === 1) td.className = "gene-cell";
        if (index === 2) td.className = "score-cell";
        if (index === 4 && row.retrospective_hit === true) td.className = "retrospective-hit";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    const unknown = payload.ignored_unknown_genes || [];
    $("results-summary").textContent =
      `${payload.n_context.toLocaleString()} observations used · ` +
      `${payload.n_candidates.toLocaleString()} candidate genes scored · ` +
      `${latestRows.length} returned · inclusion probability from ` +
      `${payload.probability_samples.toLocaleString()} Gumbel-top-k samples (not hit probability)` +
      (payload.inference_runtime ? ` · ${payload.inference_runtime}` : "") +
      (unknown.length ? ` · ${unknown.length} unknown symbol(s) ignored: ${unknown.slice(0, 8).join(", ")}${unknown.length > 8 ? "…" : ""}` : "");
    $("download-button").disabled = latestRows.length === 0;
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadCSV() {
    if (!latestRows.length) return;
    const headers = ["rank", "gene", "acquisition_score", "batch_inclusion_probability", "retrospective_hit", "public_hits", "public_screens", "common_essential", "reactome_pathway", "corum_complex"];
    const lines = [headers.join(",")];
    latestRows.forEach((row) => lines.push([
      row.rank, row.gene, row.score, row.batch_inclusion_probability, row.retrospective_hit,
      row.public_hits, row.public_screens,
      row.common_essential, row.pathway, row.complex,
    ].map(csvEscape).join(",")));
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "assayformer_next_batch.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  async function submit(event) {
    event.preventDefault();
    if (tourIndex >= 0) closeTour();
    const parsed = parseObservations();
    if (parsed.errors.length) {
      $("form-error").textContent = parsed.errors.slice(0, 4).join("; ");
      return;
    }
    const button = $("rank-button");
    button.disabled = true;
    button.textContent = "Ranking…";
    $("form-error").textContent = "";
    try {
      const payload = {
        screen: {
          phenotype: $("phenotype").value.trim(),
          cleaned_phenotype: $("cleaned-phenotype").value,
          cell_line: $("cell-line").value.trim(),
          cell_type: $("cell-type").value.trim(),
          organism: $("organism").value.trim(),
          library_methodology: $("methodology").value,
          condition_clause: $("condition").value.trim(),
          contrast_label: $("contrast-label").value,
          description: $("description").value.trim(),
        },
        observations: parsed.rows,
        batch_size: Number($("batch-size").value),
        exclude_common_essential: $("exclude-essential").checked,
        example_id: selectedExampleId,
      };
      let ranked;
      const example = examples.find((row) => row.screen_id === selectedExampleId);
      if (example && window.AssayFormerBrowser) {
        button.textContent = "Ranking in your browser…";
        try {
          ranked = await window.AssayFormerBrowser.rank(payload, example);
        } catch (browserError) {
          try {
            ranked = await jsonRequest("/api/rank", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
          } catch (apiError) {
            throw new Error(`Browser inference failed: ${browserError.message}`);
          }
        }
      } else {
        ranked = await jsonRequest("/api/rank", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      renderResults(ranked);
    } catch (error) {
      $("form-error").textContent = error.message;
    } finally {
      button.disabled = false;
      button.textContent = "Rank the next batch";
    }
  }

  async function init() {
    $("rank-form").addEventListener("submit", submit);
    $("value-mode").addEventListener("change", toggleThreshold);
    $("threshold").addEventListener("input", updateParseSummary);
    $("direction").addEventListener("change", updateParseSummary);
    $("observations").addEventListener("input", updateParseSummary);
    $("download-button").addEventListener("click", downloadCSV);
    $("tour-close").addEventListener("click", closeTour);
    $("tour-scrim").addEventListener("click", closeTour);
    $("tour-back").addEventListener("click", () => showTourStep(tourIndex - 1));
    $("tour-next").addEventListener("click", nextTourStep);
    window.addEventListener("resize", positionTour);
    window.addEventListener("scroll", positionTour, { passive: true });
    document.addEventListener("keydown", (event) => {
      if (tourIndex < 0) return;
      if (event.key === "Escape") closeTour();
      if (event.key === "ArrowRight") nextTourStep();
      if (event.key === "ArrowLeft" && tourIndex > 0) showTourStep(tourIndex - 1);
    });
    $("example-select").addEventListener("change", (event) => {
      if (event.target.value === "custom") {
        clearScreen();
        setDescriptionEditable(customScreensEnabled);
      } else {
        applyExample(Number(event.target.value));
      }
    });

    try {
      const examplePayload = await window.AssayLoop.fetchJSON("assets/data/try_examples.json");
      examples = examplePayload.examples || [];
      examples.forEach((example, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = example.name;
        $("example-select").appendChild(option);
      });
      if (examples.length) {
        $("example-select").value = "0";
        applyExample(0);
      }
    } catch (error) {
      $("form-error").textContent = "The public examples could not be loaded.";
      console.error(error);
    }

    try {
      const health = await jsonRequest("/api/health");
      customScreensEnabled = Boolean(health.custom_screens_enabled);
    } catch (error) {
      customScreensEnabled = false;
    }
    setDescriptionEditable(customScreensEnabled);
    if (examples.length) window.setTimeout(() => showTourStep(0), 250);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
