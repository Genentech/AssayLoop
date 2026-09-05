(function () {
  "use strict";

  const DATA_PATH = "assets/data/assayformer_browser.json";
  const ANNOTATIONS_PATH = "assets/data/gene_umap.json";
  let bundlePromise = null;
  let sessionPromise = null;
  let annotationsPromise = null;
  let geneIdByUpper = null;

  function bundle() {
    if (!bundlePromise) bundlePromise = window.AssayLoop.fetchJSON(DATA_PATH);
    return bundlePromise;
  }

  function annotations() {
    if (!annotationsPromise) {
      annotationsPromise = window.AssayLoop.fetchJSON(ANNOTATIONS_PATH).then((payload) => {
        const byGene = new Map();
        (payload.genes || []).forEach((row) => byGene.set(String(row.gene).toUpperCase(), row));
        return byGene;
      });
    }
    return annotationsPromise;
  }

  function modelSession(settings) {
    if (!window.ort) throw new Error("The ONNX browser runtime could not be loaded.");
    if (!sessionPromise) {
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
      sessionPromise = window.ort.InferenceSession.create(settings.model.path, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }
    return sessionPromise;
  }

  function decodeFloat32(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] === 1) {
      return new Float32Array(bytes.buffer);
    }
    const values = new Float32Array(bytes.length / 4);
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < values.length; index += 1) {
      values[index] = view.getFloat32(index * 4, true);
    }
    return values;
  }

  function seededRandom(seed) {
    let state = seed >>> 0;
    return function () {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function siftUp(values, indices, position) {
    let child = position;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (values[parent] <= values[child]) break;
      [values[parent], values[child]] = [values[child], values[parent]];
      [indices[parent], indices[child]] = [indices[child], indices[parent]];
      child = parent;
    }
  }

  function siftDown(values, indices, size) {
    let parent = 0;
    while (true) {
      const left = parent * 2 + 1;
      if (left >= size) break;
      const right = left + 1;
      const child = right < size && values[right] < values[left] ? right : left;
      if (values[parent] <= values[child]) break;
      [values[parent], values[child]] = [values[child], values[parent]];
      [indices[parent], indices[child]] = [indices[child], indices[parent]];
      parent = child;
    }
  }

  function inclusionProbabilities(scores, batchSize, samples) {
    const candidateCount = scores.length;
    const k = Math.min(batchSize, candidateCount);
    const counts = new Uint32Array(candidateCount);
    const heapValues = new Float64Array(k);
    const heapIndices = new Int32Array(k);
    const random = seededRandom(0);
    for (let sample = 0; sample < samples; sample += 1) {
      let size = 0;
      for (let index = 0; index < candidateCount; index += 1) {
        const uniform = Math.max(random(), 1e-12);
        const perturbed = scores[index] - Math.log(-Math.log(uniform));
        if (size < k) {
          heapValues[size] = perturbed;
          heapIndices[size] = index;
          siftUp(heapValues, heapIndices, size);
          size += 1;
        } else if (perturbed > heapValues[0]) {
          heapValues[0] = perturbed;
          heapIndices[0] = index;
          siftDown(heapValues, heapIndices, size);
        }
      }
      for (let index = 0; index < size; index += 1) counts[heapIndices[index]] += 1;
    }
    return Array.from(counts, (count) => count / samples);
  }

  function screenMatches(actual, expected) {
    return Object.entries(expected || {}).every(
      ([key, value]) => String(actual[key] ?? "").trim() === String(value ?? "").trim()
    );
  }

  async function rank(request, example) {
    const settings = await bundle();
    if (!example || !screenMatches(request.screen, example.screen)) {
      throw new Error("This screen needs a configured description-embedding endpoint.");
    }
    const encodedEmbedding = settings.example_embeddings[example.screen_id];
    if (!encodedEmbedding) throw new Error("No released embedding exists for this example.");

    if (!geneIdByUpper) {
      geneIdByUpper = new Map();
      settings.vocab.forEach((gene, index) => geneIdByUpper.set(String(gene).toUpperCase(), index));
    }
    const seen = new Set();
    const observedIds = [];
    const observedHits = [];
    const unknown = [];
    request.observations.forEach((row) => {
      const key = String(row.gene).toUpperCase();
      const geneId = geneIdByUpper.get(key);
      if (geneId == null) {
        unknown.push(key);
      } else if (!seen.has(key)) {
        seen.add(key);
        observedIds.push(BigInt(geneId));
        observedHits.push(BigInt(row.hit ? 1 : 0));
      }
    });

    const annotationByGene = await annotations();
    const candidateGenes = settings.candidate_genes.filter((gene) => {
      const key = String(gene).toUpperCase();
      if (seen.has(key)) return false;
      return !request.exclude_common_essential || !annotationByGene.get(key)?.essential;
    });
    if (!candidateGenes.length) throw new Error("No candidate genes remain after exclusions.");
    const candidateIds = BigInt64Array.from(
      candidateGenes,
      (gene) => BigInt(geneIdByUpper.get(String(gene).toUpperCase()))
    );
    const runtime = await modelSession(settings);
    const output = await runtime.run({
      description_embedding: new window.ort.Tensor(
        "float32", decodeFloat32(encodedEmbedding), [1, 1536]
      ),
      observed_gene_ids: new window.ort.Tensor(
        "int64", BigInt64Array.from(observedIds), [1, observedIds.length]
      ),
      observed_hit_labels: new window.ort.Tensor(
        "int64", BigInt64Array.from(observedHits), [1, observedHits.length]
      ),
      candidate_gene_ids: new window.ort.Tensor(
        "int64", candidateIds, [1, candidateIds.length]
      ),
    });
    const scores = output.acquisition_scores.data;
    const samples = settings.probability_samples;
    const probabilities = inclusionProbabilities(scores, request.batch_size, samples);
    const ranking = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
    const knownLabels = example.next_batch_labels || {};
    const results = ranking.slice(0, request.batch_size).map((candidateIndex, rankIndex) => {
      const gene = candidateGenes[candidateIndex];
      const annotation = annotationByGene.get(String(gene).toUpperCase()) || {};
      return {
        rank: rankIndex + 1,
        gene,
        score: scores[candidateIndex],
        batch_inclusion_probability: probabilities[candidateIndex],
        retrospective_hit: Object.prototype.hasOwnProperty.call(knownLabels, gene)
          ? knownLabels[gene] : null,
        public_screens: annotation.screens,
        public_hits: annotation.hits,
        common_essential: Boolean(annotation.essential),
        pathway: annotation.pathway || "",
        complex: annotation.complex || "",
      };
    });
    return {
      model: "AssayFormer (released paper checkpoint, s19; ONNX browser graph)",
      score_label: "Acquisition score",
      probability_label: "Probability of inclusion in the next batch",
      probability_method: "Monte Carlo Gumbel-top-k (Plackett-Luce)",
      probability_samples: samples,
      probability_temperature: 1.0,
      n_context: observedIds.length,
      n_candidates: candidateGenes.length,
      ignored_unknown_genes: unknown,
      example_id: example.screen_id,
      inference_runtime: "ONNX Runtime Web",
      results,
    };
  }

  window.AssayFormerBrowser = { rank };
})();
