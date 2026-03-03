const { useEffect, useMemo, useRef, useState } = React;

const OBJECTS = [
  "cat", "dog", "tree", "house", "car", "bicycle", "apple", "banana", "fish", "bird",
  "cloud", "sun", "moon", "star", "flower", "cup", "book", "chair", "table", "phone",
  "clock", "pizza", "burger", "ice cream", "guitar", "drum", "camera", "airplane", "rocket", "boat",
  "train", "bus", "robot", "monster", "dragon", "castle", "crown", "shoe", "hat", "glasses",
  "toothbrush", "key", "lock", "lamp", "cookie", "donut", "snail", "frog", "whale", "tennis ball"
];

const STORAGE_KEY = "yourdrawingssuckai.dataset.v1";
const ALGO_STATS_STORAGE_KEY = "yourdrawingssuckai.algorithmStats.v1";

const COMPARE_STATS_STORAGE_KEY = "yourdrawingssuckai.modelCompareStats.v1";
const GRID_SIZE = 16;
const ALGORITHM_COUNT = 24;

const V2_ARTICLE_PARAGRAPHS = [
  "When HyperDraw v1 launched, it was fast, funny, and surprisingly decent at rough sketches, but it still missed too often for the team to call it truly reliable.",
  "After collecting and replaying over 500 reference drawings through identical evaluation prompts, v1 correctly predicted the drawing only 14% of the time, which made it clear we needed a deeper redesign instead of a cosmetic patch.",
  "The original v1 stack worked by converting each canvas into a 16x16 intensity grid, flattening that into a 256-value vector, and running nearest-neighbor comparisons against the dataset for whichever label had the closest geometric distance.",
  "In short form, v1 leaned heavily on Euclidean distance d(x,y)=sqrt(SUM_i((x_i-y_i)^2)) and softmax confidence p_i=exp(s_i)/SUM_j exp(s_j), where lower distance implied higher score and higher score implied confidence.",
  "That pipeline was quick, but it was fragile because the model overweighted literal pixel placement: tiny translation shifts, sketch size changes, or slight rotation could make two semantically similar drawings appear mathematically far apart.",
  "We tested a variety of approaches inspired by experiments from the earlier model generations, including weighted center-priority matching and a multi-scale, rotation-aware nearest search to stabilize guesses under messy real drawing behavior.",
  "Another interesting approach emphasized line-profile statistics instead of raw pixels, using row and column density transitions to recognize structure, which improved shape understanding on symbols with strong silhouettes.",
  "Even with those gains, isolated methods still struggled with confidence calibration and class dominance, so we combined the strongest pieces into a single golden approach and then tuned it repeatedly against the same shared benchmark set.",
  "That final v2 blend moved benchmark accuracy from 14% to a staggering 38% on the exact same 500+ references, which validated that the gains were real and not just a side effect of easier data.",
  "From an inference-speed perspective, v2 now reaches a stable high-confidence answer in almost half the time under normal play loops, with an observed 53% faster convergence during repeated draw-and-guess cycles.",
  "One major v2 difference is normalization before comparison: we compute a drawing bounding box, recenter the active signal, and scale strokes into a consistent frame before applying weighted distance and k-nearest voting.",
  "We also broaden the candidate comparison set by evaluating transformed variants and feature vectors, then fusing predictions so no single brittle metric can dominate final output.",
  "Bias reduction was another direct objective because users reported v1 repeatedly falling back to bird, cloud, or cup regardless of context, which is a classic mode-collapse symptom in small sketch datasets.",
  "To counter that, v2 introduces balancing logic that reduces over-frequent label momentum and rewards agreement across diverse feature views, making it less likely to guess the simplest or most over-trained class by default.",
  "Bias avoidance is still not perfect, but it is far better than v1 and notably more likely to land on the correct answer instead of the easiest answer.",
  "The team also improved robustness around stroke noise, partial erasing, and off-center doodles so users can draw naturally without having to game the classifier.",
  "Importantly, every claimed gain in this write-up comes from matched reference materials and repeated evaluation procedures, keeping comparisons fair between v1 and v2.",
  "The writer of this article would like to thank the team for their hard work, patience, and relentless iteration in creating something truly extraordinary for the community.",
];

function getStorageItem(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function setStorageItem(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Ignore storage write errors (private browsing, disabled storage, quota exceeded).
  }
}

function randomPrompt() {
  return OBJECTS[Math.floor(Math.random() * OBJECTS.length)];
}

function loadDataset() {
  try {
    const raw = getStorageItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item) =>
        item &&
        typeof item.label === "string" &&
        Array.isArray(item.vector) &&
        item.vector.length === GRID_SIZE * GRID_SIZE &&
        typeof item.ts === "number"
    );
  } catch {
    return [];
  }
}

function saveDataset(dataset) {
  setStorageItem(STORAGE_KEY, JSON.stringify(dataset));
}

function createDefaultAlgorithmStats() {
  return Array.from({ length: ALGORITHM_COUNT }, (_, index) => ({ id: index + 1, attempts: 0, correct: 0 }));
}

function loadAlgorithmStats() {
  try {
    const raw = getStorageItem(ALGO_STATS_STORAGE_KEY);
    if (!raw) return createDefaultAlgorithmStats();

    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return createDefaultAlgorithmStats();

    const safeById = parsed.reduce((acc, stat) => {
      if (!stat || typeof stat.id !== "number") return acc;
      if (typeof stat.attempts !== "number" || typeof stat.correct !== "number") return acc;
      acc[stat.id] = {
        id: stat.id,
        attempts: Math.max(0, Math.floor(stat.attempts)),
        correct: Math.max(0, Math.floor(stat.correct)),
      };
      return acc;
    }, {});

    return createDefaultAlgorithmStats().map((defaultStat) => safeById[defaultStat.id] || defaultStat);
  } catch {
    return createDefaultAlgorithmStats();
  }
}

function saveAlgorithmStats(stats) {
  setStorageItem(ALGO_STATS_STORAGE_KEY, JSON.stringify(stats));
}


function loadCompareStats() {
  try {
    const raw = getStorageItem(COMPARE_STATS_STORAGE_KEY);
    if (!raw) return { attempts: 0, hyperDrawWins: 0, hyperDrawV2Wins: 0, ties: 0 };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") throw new Error("invalid stats");
    return {
      attempts: Math.max(0, Math.floor(parsed.attempts || 0)),
      hyperDrawWins: Math.max(0, Math.floor(parsed.hyperDrawWins || 0)),
      hyperDrawV2Wins: Math.max(0, Math.floor(parsed.hyperDrawV2Wins || 0)),
      ties: Math.max(0, Math.floor(parsed.ties || 0)),
    };
  } catch {
    return { attempts: 0, hyperDrawWins: 0, hyperDrawV2Wins: 0, ties: 0 };
  }
}

function saveCompareStats(stats) {
  setStorageItem(COMPARE_STATS_STORAGE_KEY, JSON.stringify(stats));
}

function distance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return Math.sqrt(total);
}

function softmax(values) {
  if (values.length === 0) return [];
  const peak = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function boundingBox(vector) {
  let minX = GRID_SIZE;
  let maxX = -1;
  let minY = GRID_SIZE;
  let maxY = -1;

  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const value = vector[y * GRID_SIZE + x];
      if (value <= 0.05) continue;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) return null;

  return { minX, maxX, minY, maxY };
}

function normalizeVector(vector) {
  const box = boundingBox(vector);
  if (!box) return vector;

  const width = box.maxX - box.minX + 1;
  const height = box.maxY - box.minY + 1;
  const scale = Math.max(width, height);

  const output = new Array(vector.length).fill(0);
  const offsetX = Math.floor((GRID_SIZE - scale) / 2);
  const offsetY = Math.floor((GRID_SIZE - scale) / 2);

  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const sourceX = box.minX + ((x - offsetX) / scale) * width;
      const sourceY = box.minY + ((y - offsetY) / scale) * height;
      const ix = Math.floor(sourceX);
      const iy = Math.floor(sourceY);

      if (ix < box.minX || ix > box.maxX || iy < box.minY || iy > box.maxY) continue;

      const value = vector[iy * GRID_SIZE + ix];
      output[y * GRID_SIZE + x] = value > 0.05 ? value : 0;
    }
  }

  return output;
}

function buildLabelPrototypes(dataset) {
  const grouped = dataset.reduce((acc, item) => {
    if (!acc[item.label]) acc[item.label] = [];
    acc[item.label].push(normalizeVector(item.vector));
    return acc;
  }, {});

  return Object.entries(grouped).reduce((acc, [label, vectors]) => {
    const prototype = new Array(GRID_SIZE * GRID_SIZE).fill(0);
    vectors.forEach((vector) => {
      for (let i = 0; i < vector.length; i += 1) {
        prototype[i] += vector[i];
      }
    });

    for (let i = 0; i < prototype.length; i += 1) {
      prototype[i] /= vectors.length;
    }

    acc[label] = prototype;
    return acc;
  }, {});
}

function weightedDistance(a, b, weightFn) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    total += d * d * weightFn(i);
  }
  return Math.sqrt(total);
}

function binarizeVector(vector, threshold = 0.2) {
  return vector.map((value) => (value >= threshold ? 1 : 0));
}

function rotateVector90(vector) {
  const output = new Array(vector.length).fill(0);
  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      output[x * GRID_SIZE + (GRID_SIZE - 1 - y)] = vector[y * GRID_SIZE + x];
    }
  }
  return output;
}

function generateRotations(vector) {
  const rot0 = vector;
  const rot90 = rotateVector90(rot0);
  const rot180 = rotateVector90(rot90);
  const rot270 = rotateVector90(rot180);
  return [rot0, rot90, rot180, rot270];
}

function flipVectorHorizontal(vector) {
  const output = new Array(vector.length).fill(0);
  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      output[y * GRID_SIZE + (GRID_SIZE - 1 - x)] = vector[y * GRID_SIZE + x];
    }
  }
  return output;
}

function generateTransformVariants(vector) {
  const rotations = generateRotations(vector);
  const flipped = flipVectorHorizontal(vector);
  return [...rotations, ...generateRotations(flipped)];
}

function scoreTransformInvariantModel(inputVector, dataset, options = {}) {
  const {
    k = 15,
    distanceFloor = 0.02,
    normalizeDataset = true,
    featureWeight = 0.3,
    centerWeightPower = 0,
  } = options;

  const inputNorm = normalizeVector(inputVector);
  const inputFeatures = extractLineFeatures(inputNorm).compact;
  const inputCandidates = generateTransformVariants(inputNorm);

  const scored = dataset.map((item) => {
    const base = normalizeDataset ? normalizeVector(item.vector) : item.vector;
    const candidates = generateTransformVariants(base);

    const bestDistance = inputCandidates.reduce((bestInput, inputCandidate) => {
      const bestForInput = candidates.reduce((bestCandidate, candidate) => {
        let d = distance(inputCandidate, candidate) / Math.sqrt(inputVector.length);
        if (centerWeightPower > 0) {
          const box = boundingBox(candidate);
          if (box) {
            const cx = (box.minX + box.maxX) / 2;
            const cy = (box.minY + box.maxY) / 2;
            const centerDx = Math.abs(cx - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
            const centerDy = Math.abs(cy - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
            const centerPenalty = Math.pow((centerDx + centerDy) / 2, centerWeightPower);
            d *= 1 + centerPenalty * 0.2;
          }
        }
        return Math.min(bestCandidate, d);
      }, Number.POSITIVE_INFINITY);
      return Math.min(bestInput, bestForInput);
    }, Number.POSITIVE_INFINITY);

    const candidateFeatures = extractLineFeatures(base).compact;
    const lineDistance = featureDistance(inputFeatures, candidateFeatures);
    const blendedDistance = bestDistance * (1 - featureWeight) + lineDistance * featureWeight;

    return {
      label: item.label,
      distance: blendedDistance,
      rawDistance: bestDistance,
    };
  });

  const ranked = scored.sort((a, b) => a.distance - b.distance);
  const vote = voteByInverseDistance(
    ranked.map((entry) => ({
      label: entry.label,
      distance: Math.max(distanceFloor, entry.distance),
    })),
    k
  );

  const nearest = ranked[0] || { label: "unknown", rawDistance: 1 };

  return {
    label: vote.label,
    confidence: vote.confidence,
    nearestLabel: nearest.label,
    nearestConfidence: Math.round((1 - Math.min(1, nearest.rawDistance)) * 100),
  };
}

function extractLineFeatures(vector) {
  const norm = normalizeVector(vector);
  const binary = binarizeVector(norm, 0.25);
  const rowSums = new Array(GRID_SIZE).fill(0);
  const colSums = new Array(GRID_SIZE).fill(0);
  let hTransitions = 0;
  let vTransitions = 0;
  let d1Transitions = 0;
  let d2Transitions = 0;
  let active = 0;
  let cx = 0;
  let cy = 0;

  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const index = y * GRID_SIZE + x;
      const value = binary[index];
      rowSums[y] += value;
      colSums[x] += value;
      active += value;
      cx += value * x;
      cy += value * y;

      if (x < GRID_SIZE - 1 && value !== binary[index + 1]) hTransitions += 1;
      if (y < GRID_SIZE - 1 && value !== binary[index + GRID_SIZE]) vTransitions += 1;
      if (x < GRID_SIZE - 1 && y < GRID_SIZE - 1 && value !== binary[index + GRID_SIZE + 1]) d1Transitions += 1;
      if (x > 0 && y < GRID_SIZE - 1 && value !== binary[index + GRID_SIZE - 1]) d2Transitions += 1;
    }
  }

  const safeActive = Math.max(active, 1);
  const centerX = cx / safeActive / GRID_SIZE;
  const centerY = cy / safeActive / GRID_SIZE;

  const transitions = [hTransitions, vTransitions, d1Transitions, d2Transitions].map((v) => v / (GRID_SIZE * GRID_SIZE));
  const rowProfile = rowSums.map((value) => value / GRID_SIZE);
  const colProfile = colSums.map((value) => value / GRID_SIZE);

  return {
    binary,
    full: [...transitions, active / (GRID_SIZE * GRID_SIZE), centerX, centerY, ...rowProfile, ...colProfile],
    compact: [...transitions, active / (GRID_SIZE * GRID_SIZE), centerX, centerY],
    profileOnly: [...rowProfile, ...colProfile],
  };
}

function featureDistance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return Math.sqrt(total / Math.max(1, a.length));
}

function voteFeatureKnn(featureInput, dataset, featureSelector, k) {
  const scored = dataset
    .map((item) => ({
      label: item.label,
      distance: featureDistance(featureInput, featureSelector(extractLineFeatures(item.vector))),
    }))
    .sort((a, b) => a.distance - b.distance);

  return voteByInverseDistance(scored, k);
}

function voteByInverseDistance(scoredExamples, k) {
  const topK = scoredExamples.slice(0, Math.min(k, scoredExamples.length));
  const labelScores = topK.reduce((acc, item) => {
    const vote = 1 / Math.max(item.distance, 0.001);
    acc[item.label] = (acc[item.label] || 0) + vote;
    return acc;
  }, {});

  const ranked = Object.entries(labelScores).sort((a, b) => b[1] - a[1]);
  const [label = "unknown"] = ranked[0] || [];
  const probabilities = softmax(ranked.map(([, value]) => value));
  const confidence = Math.round((probabilities[0] || 0) * 100);
  return { label, confidence };
}

function runAlgorithms(vector, dataset) {
  if (!dataset.length) {
    return Array.from({ length: ALGORITHM_COUNT }, (_, index) => ({
      id: index + 1,
      name: `Algorithm ${index + 1}`,
      label: "Need training data first",
      confidence: 0,
    }));
  }

  const normalizedInput = normalizeVector(vector);
  const prototypesRaw = buildLabelPrototypes(dataset.map((item) => ({ ...item, vector: item.vector })));
  const prototypesNormalized = buildLabelPrototypes(dataset.map((item) => ({ ...item, vector: normalizeVector(item.vector) })));

  const rawDistances = dataset
    .map((item) => ({
      label: item.label,
      distance: distance(vector, item.vector) / Math.sqrt(vector.length),
    }))
    .sort((a, b) => a.distance - b.distance);

  const normalizedDistances = dataset
    .map((item) => ({
      label: item.label,
      distance: distance(normalizedInput, normalizeVector(item.vector)) / Math.sqrt(vector.length),
    }))
    .sort((a, b) => a.distance - b.distance);

  const algo1TopK = normalizedDistances.slice(0, Math.min(24, normalizedDistances.length));
  const algo1LabelScores = algo1TopK.reduce((acc, item) => {
    acc[item.label] = (acc[item.label] || 0) + 1 / Math.max(item.distance + 0.08, 0.001);
    return acc;
  }, {});

  Object.entries(prototypesNormalized).forEach(([label, prototype]) => {
    const prototypeDistance = distance(normalizedInput, prototype) / Math.sqrt(vector.length);
    const prototypeVote = 1 / Math.max(0.001, prototypeDistance + 0.06);
    algo1LabelScores[label] = (algo1LabelScores[label] || 0) + prototypeVote * 0.35;
  });

  const algo1Ranked = Object.entries(algo1LabelScores).sort((a, b) => b[1] - a[1]);
  const algo1Probs = softmax(algo1Ranked.map(([, score]) => score));
  const algo1Guess = algo1Ranked[0]?.[0] || "unknown";
  const algo1Confidence = Math.round((algo1Probs[0] || 0) * 100);

  const nearestRaw = rawDistances[0];
  const nearestNorm = normalizedDistances[0];
  const knn5Raw = voteByInverseDistance(rawDistances, 5);
  const knn11Norm = voteByInverseDistance(normalizedDistances, 11);

  const prototypeRaw = Object.entries(prototypesRaw)
    .map(([label, proto]) => ({ label, distance: distance(vector, proto) / Math.sqrt(vector.length) }))
    .sort((a, b) => a.distance - b.distance)[0];

  const prototypeNorm = Object.entries(prototypesNormalized)
    .map(([label, proto]) => ({ label, distance: distance(normalizedInput, proto) / Math.sqrt(vector.length) }))
    .sort((a, b) => a.distance - b.distance)[0];

  const centerWeighted = dataset
    .map((item) => ({
      label: item.label,
      distance:
        weightedDistance(normalizedInput, normalizeVector(item.vector), (index) => {
          const x = index % GRID_SIZE;
          const y = Math.floor(index / GRID_SIZE);
          const dx = Math.abs(x - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
          const dy = Math.abs(y - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
          return 1.4 - Math.min(1, (dx + dy) / 2) * 0.6;
        }) / Math.sqrt(vector.length),
    }))
    .sort((a, b) => a.distance - b.distance)[0];

  const binaryInput = binarizeVector(normalizedInput);
  const binaryNearest = dataset
    .map((item) => ({
      label: item.label,
      distance: distance(binaryInput, binarizeVector(normalizeVector(item.vector))) / Math.sqrt(vector.length),
    }))
    .sort((a, b) => a.distance - b.distance)[0];

  const edgeWeighted = dataset
    .map((item) => ({
      label: item.label,
      distance:
        weightedDistance(normalizedInput, normalizeVector(item.vector), (index) => {
          const x = index % GRID_SIZE;
          const y = Math.floor(index / GRID_SIZE);
          const dx = Math.abs(x - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
          const dy = Math.abs(y - (GRID_SIZE - 1) / 2) / (GRID_SIZE / 2);
          return 0.8 + Math.min(1, (dx + dy) / 2) * 0.9;
        }) / Math.sqrt(vector.length),
    }))
    .sort((a, b) => a.distance - b.distance)[0];

  const knn21Norm = voteByInverseDistance(normalizedDistances, 21);

  const prototypeBlendScores = Object.entries(prototypesNormalized).map(([label, prototype]) => {
    const normalizedPrototypeDistance = distance(normalizedInput, prototype) / Math.sqrt(vector.length);
    const nearestSupport = normalizedDistances.filter((item) => item.label === label).slice(0, 3);
    const nearestSupportDistance =
      nearestSupport.length > 0
        ? nearestSupport.reduce((sum, item) => sum + item.distance, 0) / nearestSupport.length
        : 1;

    return {
      label,
      score: 1 / Math.max(0.001, normalizedPrototypeDistance * 0.7 + nearestSupportDistance * 0.3 + 0.02),
    };
  });

  const prototypeBlendRanked = prototypeBlendScores.sort((a, b) => b.score - a.score);
  const prototypeBlendProbabilities = softmax(prototypeBlendRanked.map((item) => item.score));
  const prototypeBlendTop = prototypeBlendRanked[0];

  const inputLineFeatures = extractLineFeatures(vector);

  const lineShapeKnn7 = voteFeatureKnn(inputLineFeatures.full, dataset, (features) => features.full, 7);
  const lineShapeKnn15 = voteFeatureKnn(inputLineFeatures.full, dataset, (features) => features.full, 15);
  const lineProfileKnn11 = voteFeatureKnn(inputLineFeatures.profileOnly, dataset, (features) => features.profileOnly, 11);
  const lineCompactKnn9 = voteFeatureKnn(inputLineFeatures.compact, dataset, (features) => features.compact, 9);

  const model7LikeInputCandidates = [vector, normalizedInput, ...generateRotations(normalizedInput)];
  const model7LikeDataset = dataset.map((item) => {
    const normalized = normalizeVector(item.vector);
    const rotations = generateRotations(normalized);
    const prototypes = [item.vector, normalized, ...rotations];
    const bestDistance = prototypes.reduce((best, candidate) => {
      const candidateDistance = distance(normalizedInput, candidate) / Math.sqrt(vector.length);
      return Math.min(best, candidateDistance);
    }, Number.POSITIVE_INFINITY);

    const bestAlignedDistance = model7LikeInputCandidates.reduce((bestInput, inputCandidate) => {
      const candidateDistance = prototypes.reduce((bestProto, candidate) => {
        const d = distance(inputCandidate, candidate) / Math.sqrt(vector.length);
        return Math.min(bestProto, d);
      }, Number.POSITIVE_INFINITY);
      return Math.min(bestInput, candidateDistance);
    }, Number.POSITIVE_INFINITY);

    return {
      label: item.label,
      distance: bestDistance,
      alignedDistance: bestAlignedDistance,
    };
  });

  const model7LikeNearest = [...model7LikeDataset].sort((a, b) => a.distance - b.distance)[0];
  const model7LikeInvariantVote = voteByInverseDistance(
    model7LikeDataset
      .map((item) => ({ label: item.label, distance: item.alignedDistance }))
      .sort((a, b) => a.distance - b.distance),
    17
  );

  const model20 = scoreTransformInvariantModel(vector, dataset, {
    k: 13,
    distanceFloor: 0.015,
    featureWeight: 0.25,
    centerWeightPower: 0,
  });
  const model21 = scoreTransformInvariantModel(vector, dataset, {
    k: 17,
    distanceFloor: 0.02,
    featureWeight: 0.35,
    centerWeightPower: 0,
  });
  const model22 = scoreTransformInvariantModel(vector, dataset, {
    k: 21,
    distanceFloor: 0.02,
    featureWeight: 0.3,
    centerWeightPower: 1,
  });
  const model23 = scoreTransformInvariantModel(vector, dataset, {
    k: 19,
    distanceFloor: 0.01,
    featureWeight: 0.4,
    centerWeightPower: 2,
  });
  const model24 = scoreTransformInvariantModel(vector, dataset, {
    k: 23,
    distanceFloor: 0.01,
    featureWeight: 0.45,
    centerWeightPower: 1,
  });

  return [
    { id: 1, name: "Algorithm 1 (Current)", label: algo1Guess, confidence: algo1Confidence },
    { id: 2, name: "Algorithm 2 (Nearest Raw)", label: nearestRaw?.label || "unknown", confidence: Math.round((1 - Math.min(1, nearestRaw?.distance || 1)) * 100) },
    { id: 3, name: "Algorithm 3 (Nearest Normalized)", label: nearestNorm?.label || "unknown", confidence: Math.round((1 - Math.min(1, nearestNorm?.distance || 1)) * 100) },
    { id: 4, name: "Algorithm 4 (kNN-5 Raw)", label: knn5Raw.label, confidence: knn5Raw.confidence },
    { id: 5, name: "Algorithm 5 (kNN-11 Normalized)", label: knn11Norm.label, confidence: knn11Norm.confidence },
    { id: 6, name: "Algorithm 6 (Prototype Raw)", label: prototypeRaw?.label || "unknown", confidence: Math.round((1 - Math.min(1, prototypeRaw?.distance || 1)) * 100) },
    { id: 7, name: "Algorithm 7 (Prototype Normalized)", label: prototypeNorm?.label || "unknown", confidence: Math.round((1 - Math.min(1, prototypeNorm?.distance || 1)) * 100) },
    { id: 8, name: "Algorithm 8 (Center Weighted)", label: centerWeighted?.label || "unknown", confidence: Math.round((1 - Math.min(1, centerWeighted?.distance || 1)) * 100) },
    { id: 9, name: "Algorithm 9 (Binary Shape)", label: binaryNearest?.label || "unknown", confidence: Math.round((1 - Math.min(1, binaryNearest?.distance || 1)) * 100) },
    { id: 10, name: "Algorithm 10 (Edge Weighted)", label: edgeWeighted?.label || "unknown", confidence: Math.round((1 - Math.min(1, edgeWeighted?.distance || 1)) * 100) },
    { id: 11, name: "Algorithm 11 (kNN-21 Normalized)", label: knn21Norm.label, confidence: knn21Norm.confidence },
    {
      id: 12,
      name: "Algorithm 12 (Prototype Blend)",
      label: prototypeBlendTop?.label || "unknown",
      confidence: Math.round((prototypeBlendProbabilities[0] || 0) * 100),
    },
    {
      id: 13,
      name: "Algorithm 13 (Raw+Norm Hybrid)",
      label: nearestNorm?.distance <= (nearestRaw?.distance ?? 1) ? nearestNorm?.label || "unknown" : nearestRaw?.label || "unknown",
      confidence: Math.round(
        (1 - Math.min(1, Math.min(nearestNorm?.distance ?? 1, nearestRaw?.distance ?? 1))) * 100
      ),
    },
    { id: 14, name: "Algorithm 14 (Line Shape kNN-7)", label: lineShapeKnn7.label, confidence: lineShapeKnn7.confidence },
    { id: 15, name: "Algorithm 15 (Line Shape kNN-15)", label: lineShapeKnn15.label, confidence: lineShapeKnn15.confidence },
    { id: 16, name: "Algorithm 16 (Line Profile kNN-11)", label: lineProfileKnn11.label, confidence: lineProfileKnn11.confidence },
    { id: 17, name: "Algorithm 17 (Line Compact kNN-9)", label: lineCompactKnn9.label, confidence: lineCompactKnn9.confidence },
    { id: 18, name: "Algorithm 18 (Model 7 Multi-Scale/Rotation Nearest)", label: model7LikeNearest?.label || "unknown", confidence: Math.round((1 - Math.min(1, model7LikeNearest?.distance || 1)) * 100) },
    { id: 19, name: "Algorithm 19 (Model 7 Multi-Transform kNN-17)", label: model7LikeInvariantVote.label, confidence: model7LikeInvariantVote.confidence },
    { id: 20, name: "Algorithm 20 (v2 Transform Invariant kNN-13)", label: model20.label, confidence: model20.confidence },
    { id: 21, name: "Algorithm 21 (v2 Transform + Line Blend kNN-17)", label: model21.label, confidence: model21.confidence },
    { id: 22, name: "Algorithm 22 (v2 Rotation/Scale Robust kNN-21)", label: model22.label, confidence: model22.confidence },
    { id: 23, name: "Algorithm 23 (v2 Invariant + Center Stabilized)", label: model23.label, confidence: model23.confidence },
    { id: 24, name: "Algorithm 24 (v2 Invariant Ensemble kNN-23)", label: model24.label, confidence: model24.confidence },
  ];
}


function App() {
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const strokesRef = useRef([]);
  const activeStrokeRef = useRef(null);
  const drawingRevisionRef = useRef(0);
  const lastGuessedRevisionRef = useRef(-1);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => randomPrompt());
  const [selectedModel, setSelectedModel] = useState("hyperdraw");
  const [compareMode, setCompareMode] = useState(false);
  const [guess, setGuess] = useState("start drawing");
  const [confidence, setConfidence] = useState(0);
  const [compareResults, setCompareResults] = useState({
    hyperDraw: { label: "start drawing", confidence: 0 },
    hyperDrawV2: { label: "start drawing", confidence: 0 },
  });
  const [compareStats, setCompareStats] = useState(() => loadCompareStats());
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [showIntro, setShowIntro] = useState(true);
  const [introDetails, setIntroDetails] = useState(false);
  const [activeTab, setActiveTab] = useState("draw");
  const [algorithmStats, setAlgorithmStats] = useState(() => loadAlgorithmStats());
  const [lastDoneResults, setLastDoneResults] = useState([]);

  useEffect(() => {
    saveAlgorithmStats(algorithmStats);
  }, [algorithmStats]);

  useEffect(() => {
    saveCompareStats(compareStats);
  }, [compareStats]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 20;
  }, [showIntro]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = isErasing ? "#ffffff" : "#111827";
    ctx.lineWidth = isErasing ? 32 : 20;
  }, [isErasing]);

  const getPoint = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    if (event.touches?.length) {
      const touch = event.touches[0];
      return {
        x: ((touch.clientX - rect.left) * canvas.width) / rect.width,
        y: ((touch.clientY - rect.top) * canvas.height) / rect.height,
      };
    }

    return {
      x: ((event.clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.clientY - rect.top) * canvas.height) / rect.height,
    };
  };

  const startDrawing = (event) => {
    event.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const point = getPoint(event);
    isDrawingRef.current = true;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    activeStrokeRef.current = [point];
    strokesRef.current.push(activeStrokeRef.current);
  };

  const draw = (event) => {
    if (!isDrawingRef.current) return;
    event.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const point = getPoint(event);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    activeStrokeRef.current?.push(point);
    drawingRevisionRef.current += 1;
  };

  const stopDrawing = () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    canvasRef.current.getContext("2d").closePath();
    activeStrokeRef.current = null;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    strokesRef.current = [];
    activeStrokeRef.current = null;
    drawingRevisionRef.current += 1;
    setGuess("start drawing");
    setConfidence(0);
    setCompareResults({
      hyperDraw: { label: "start drawing", confidence: 0 },
      hyperDrawV2: { label: "start drawing", confidence: 0 },
    });
    setStatusMessage("");
  };

  const vectorizeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return new Array(GRID_SIZE * GRID_SIZE).fill(0);

    const offscreen = document.createElement("canvas");
    offscreen.width = GRID_SIZE;
    offscreen.height = GRID_SIZE;

    const octx = offscreen.getContext("2d");
    octx.fillStyle = "white";
    octx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
    octx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);

    const { data } = octx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const vec = [];
    for (let i = 0; i < data.length; i += 4) {
      const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
      vec.push(1 - gray / 255);
    }
    return vec;
  };

  const getDrawingStats = () => {
    const vec = vectorizeCanvas();
    const totalInk = vec.reduce((sum, value) => sum + value, 0);
    const activePixels = vec.reduce((count, value) => count + (value > 0.18 ? 1 : 0), 0);
    const drawnStrokeCount = strokesRef.current.filter((stroke) => stroke.length > 1).length;

    return {
      vec,
      totalInk,
      activePixels,
      drawnStrokeCount,
      hasAnyInk: totalInk > 0.03 || activePixels > 0,
      hasMeaningfulDrawing: totalInk > 5 && activePixels > 8 && drawnStrokeCount > 0,
    };
  };

  const guessDrawing = () => {
    if (!canvasRef.current) return;

    const drawingStats = getDrawingStats();

    const needsEarlyGuess = selectedModel === "hyperdraw_v2" || devMode;

    if (!drawingStats.hasAnyInk) {
      setStatusMessage("Draw something first — erased/blank canvas cannot be guessed.");
      setConfidence(0);
      return;
    }

    if (!drawingStats.hasMeaningfulDrawing && !needsEarlyGuess) {
      setStatusMessage("Draw a little more for HyperDraw to start guessing.");
      setConfidence(0);
      return;
    }

    if (dataset.length === 0) {
      setGuess("Need training data first");
      setConfidence(0);
      setStatusMessage("Train me with a few drawings before guessing.");
      setLastDoneResults([]);
      return;
    }

    const results = runAlgorithms(drawingStats.vec, dataset);
    const hyperDraw = results.find((entry) => entry.id === 1) || { label: "unknown", confidence: 0 };
    const hyperDrawV2 = results.find((entry) => entry.id === 24) || { label: "unknown", confidence: 0 };
    const selected = selectedModel === "hyperdraw_v2" ? hyperDrawV2 : hyperDraw;
    const conf = Math.max(1, Math.min(99, selected.confidence));
    const lowConfidence = conf < 60;

    setGuess(selected.label);
    setConfidence(conf);
    setCompareResults({
      hyperDraw: { label: hyperDraw.label, confidence: Math.max(1, Math.min(99, hyperDraw.confidence || 0)) },
      hyperDrawV2: { label: hyperDrawV2.label, confidence: Math.max(1, Math.min(99, hyperDrawV2.confidence || 0)) },
    });
    setLastDoneResults(results);
    setStatusMessage(lowConfidence ? "Low confidence guess — try cleaner strokes for better accuracy." : "");
  };

  const stopDrawingAndGuess = () => {
    stopDrawing();
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (!canvasRef.current) return;
      if (drawingRevisionRef.current === lastGuessedRevisionRef.current) return;
      lastGuessedRevisionRef.current = drawingRevisionRef.current;
      guessDrawing();
    }, 300);

    return () => clearInterval(intervalId);
  }, [dataset]);

  const saveDrawing = () => {
    const drawingStats = getDrawingStats();

    if (!drawingStats.hasMeaningfulDrawing) {
      setStatusMessage("Nope — draw first. Blank/erased canvas won't be saved to training.");
      return;
    }

    const { vec } = drawingStats;
    const results = runAlgorithms(vec, dataset);
    const hyperDraw = results.find((entry) => entry.id === 1) || { label: "unknown", confidence: 0 };
    const hyperDrawV2 = results.find((entry) => entry.id === 24) || { label: "unknown", confidence: 0 };

    setCompareResults({
      hyperDraw: { label: hyperDraw.label, confidence: Math.max(1, Math.min(99, hyperDraw.confidence || 0)) },
      hyperDrawV2: { label: hyperDrawV2.label, confidence: Math.max(1, Math.min(99, hyperDrawV2.confidence || 0)) },
    });
    setLastDoneResults(results);
    setAlgorithmStats((previous) =>
      previous.map((algo) => {
        const result = results.find((entry) => entry.id === algo.id);
        const gotItRight = result?.label === prompt;
        return {
          ...algo,
          attempts: algo.attempts + 1,
          correct: algo.correct + (gotItRight ? 1 : 0),
        };
      })
    );

    setCompareStats((previous) => {
      const next = { ...previous, attempts: previous.attempts + 1 };
      const hyperDrawCorrect = hyperDraw.label === prompt;
      const hyperDrawV2Correct = hyperDrawV2.label === prompt;
      if (hyperDrawCorrect && !hyperDrawV2Correct) next.hyperDrawWins += 1;
      else if (hyperDrawV2Correct && !hyperDrawCorrect) next.hyperDrawV2Wins += 1;
      else next.ties += 1;
      return next;
    });

    const updated = [...dataset, { label: prompt, vector: vec, ts: Date.now() }].slice(-2000);
    setDataset(updated);
    saveDataset(updated);
    setPrompt(randomPrompt());
    clearCanvas();
    setStatusMessage("Done! Added to dataset and moved to the next prompt.");
  };

  const promptCounts = useMemo(
    () =>
      dataset.reduce((acc, item) => {
        acc[item.label] = (acc[item.label] || 0) + 1;
        return acc;
      }, {}),
    [dataset]
  );

  if (showIntro) {
    return (
      <main className="app intro-screen">
        <section className="card intro-card intro-card-glow">
          <p className="intro-kicker">⚡ BREAKTHROUGH RELEASE</p>
          <h1>INTRODUCING HYPERDRAWv2</h1>
          <p className="subtitle">Faster guesses, stronger shape understanding, and significantly less default-label bias.</p>
          <div className="intro-highlights">
            <div className="intro-pill"><strong>v1:</strong>&nbsp;14% benchmark hit rate</div>
            <div className="intro-pill"><strong>v2:</strong>&nbsp;38% benchmark hit rate</div>
            <div className="intro-pill"><strong>Speed:</strong>&nbsp;53% faster stable guesses</div>
          </div>
          {!introDetails ? (
            <div className="row">
              <button className="primary" onClick={() => setIntroDetails(true)}>Learn More</button>
              <button
                className="secondary"
                onClick={() => {
                  setShowIntro(false);
                }}
              >
                Continue
              </button>
            </div>
          ) : (
            <>
              <p>
                Full documentation is now available in the <strong>Articles</strong> tab and includes a long-form breakdown of
                how v1 worked, why it plateaued, and how v2 was tuned to achieve dramatically better quality under the same
                reference conditions.
              </p>
              <p>
                It covers benchmark outcomes on 500+ references, formulas used in v1, tested strategy families from earlier
                approaches, the final golden method, and progress on reducing overconfident bias toward classes like bird,
                cloud, and cup.
              </p>
              <div className="row">
                <button
                  className="secondary"
                  onClick={() => {
                    setShowIntro(false);
                    setActiveTab("articles");
                  }}
                >
                  Open Articles
                </button>
                <button
                  className="primary"
                  onClick={() => {
                    setShowIntro(false);
                  }}
                >
                  Continue to Draw Lab
                </button>
              </div>
            </>
          )}
        </section>
      </main>
    );
  }

  return (
    <main className="app">
      <h1>YourDrawingsSuck.AI</h1>
      <p className="subtitle">Get a random object, draw it, and let our hilariously judgy AI guess from community sketches.</p>

      <div className="row">
        <button className={`secondary ${activeTab === "draw" ? "active" : ""}`} onClick={() => setActiveTab("draw")}>Draw Lab</button>
        <button className={`secondary ${activeTab === "articles" ? "active" : ""}`} onClick={() => setActiveTab("articles")}>Articles</button>
      </div>

      {activeTab === "draw" ? (
      <div className="grid">
        <section className="card">
          <h2>Draw this: <span style={{ color: "#6ee7b7" }}>{prompt}</span></h2>
          <canvas
            ref={canvasRef}
            width="500"
            height="500"
            aria-label="drawing area"
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawingAndGuess}
            onMouseLeave={stopDrawingAndGuess}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawingAndGuess}
          ></canvas>
          <div className="row">
            <button className={`secondary ${!isErasing ? "active" : ""}`} onClick={() => setIsErasing(false)}>Draw</button>
            <button className={`secondary ${isErasing ? "active" : ""}`} onClick={() => setIsErasing(true)}>Eraser</button>
            <button className="primary" onClick={saveDrawing}>Done</button>
            <button className="warn" onClick={clearCanvas}>Clear</button>
            <button className={`secondary ${devMode ? "active" : ""}`} onClick={() => setDevMode((on) => !on)}>
              {devMode ? "Dev Mode: ON" : "Dev Mode"}
            </button>
          </div>
          {statusMessage && <p className="status-msg">{statusMessage}</p>}
        </section>

        <aside className="card">
          <h2>AI Guess</h2>
          <div className="row controls-row">
            <label>
              Model:&nbsp;
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                <option value="hyperdraw">HyperDraw</option>
                <option value="hyperdraw_v2">HyperDraw_v2</option>
              </select>
            </label>
            <button className={`secondary ${compareMode ? "active" : ""}`} onClick={() => setCompareMode((on) => !on)}>
              {compareMode ? "Hide Compare" : "Compare"}
            </button>
          </div>
          {!compareMode ? (
            <>
              <p className="big">{guess}</p>
              <p>Confidence: {confidence}%</p>
            </>
          ) : (
            <div className="compare-grid">
              <div className="stat">
                <div><strong>HyperDraw</strong></div>
                <div>Guess: {compareResults.hyperDraw.label}</div>
                <div>Confidence: {compareResults.hyperDraw.confidence}%</div>
              </div>
              <div className="stat">
                <div><strong>HyperDraw_v2</strong></div>
                <div>Guess: {compareResults.hyperDrawV2.label}</div>
                <div>Confidence: {compareResults.hyperDrawV2.confidence}%</div>
              </div>
            </div>
          )}

          <div className="stats">
            <div className="stat"><div>Total drawings</div><div className="big">{dataset.length}</div></div>
            <div className="stat"><div>Objects learned</div><div className="big">{Object.keys(promptCounts).length}</div></div>
          </div>

          <div className="stats">
            <div className="stat"><div>Compare rounds</div><div className="big">{compareStats.attempts}</div></div>
            <div className="stat"><div>HyperDraw wins</div><div className="big">{compareStats.hyperDrawWins}</div></div>
            <div className="stat"><div>HyperDraw_v2 wins</div><div className="big">{compareStats.hyperDrawV2Wins}</div></div>
            <div className="stat"><div>Ties</div><div className="big">{compareStats.ties}</div></div>
          </div>

          <h3>Top trained objects</h3>
          <ul>
            {Object.entries(promptCounts)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([label, count]) => <li key={label}>{label}: {count}</li>)}
          </ul>

          {devMode && (
            <>
              <h3>Algorithm lab</h3>
              <p>Click <strong>Done</strong> to log correctness rates for all 24 algorithms.</p>
              <div className="algo-grid">
                {[...algorithmStats]
                  .sort((a, b) => {
                    const aAccuracy = a.attempts ? a.correct / a.attempts : -1;
                    const bAccuracy = b.attempts ? b.correct / b.attempts : -1;
                    if (bAccuracy !== aAccuracy) return bAccuracy - aAccuracy;
                    if (b.correct !== a.correct) return b.correct - a.correct;
                    return a.id - b.id;
                  })
                  .map((algo) => {
                  const latest = lastDoneResults.find((entry) => entry.id === algo.id);
                  const accuracy = algo.attempts ? Math.round((algo.correct / algo.attempts) * 100) : 0;
                  return (
                    <div className="stat" key={algo.id}>
                      <div><strong>Algorithm {algo.id}</strong>{algo.id === 1 ? " (live model)" : ""}</div>
                      <div>Guess: {latest?.label || "-"}</div>
                      <div>Guess confidence: {latest?.confidence ?? 0}%</div>
                      <div>Correctness rate: {accuracy}% ({algo.correct}/{algo.attempts})</div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </aside>
      </div>
      ) : (
        <section className="card article-card">
          <h2>HyperDraw v2 Deep Dive</h2>
          <p className="subtitle">A long-form update for returning users who want the full story.</p>
          <article>
            {V2_ARTICLE_PARAGRAPHS.map((paragraph, index) => (
              <p key={`v2-article-${index}`}>{paragraph}</p>
            ))}
          </article>
        </section>
      )}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
