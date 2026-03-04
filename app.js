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

const ACTIVE_ALGORITHM_IDS = [1, 7, 45, 57, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76];
const HYPERDRAW_ALGORITHM_ID = 1;
const HYPERDRAW_V2_ALGORITHM_ID = 7;

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
    const downscale32To16 = (vector) => {
      if (!Array.isArray(vector) || vector.length !== 32 * 32) return vector;
      const output = new Array(GRID_SIZE * GRID_SIZE).fill(0);
      for (let y = 0; y < GRID_SIZE; y += 1) {
        for (let x = 0; x < GRID_SIZE; x += 1) {
          let sum = 0;
          for (let oy = 0; oy < 2; oy += 1) {
            for (let ox = 0; ox < 2; ox += 1) {
              const sourceX = x * 2 + ox;
              const sourceY = y * 2 + oy;
              sum += vector[sourceY * 32 + sourceX] || 0;
            }
          }
          output[y * GRID_SIZE + x] = sum / 4;
        }
      }
      return output;
    };

    return parsed.filter(
      (item) =>
        item &&
        typeof item.label === "string" &&
        Array.isArray(item.vector) &&
        typeof item.ts === "number"
    ).map((item) => ({
      ...item,
      vector: downscale32To16(item.vector),
    })).filter((item) => item.vector.length === GRID_SIZE * GRID_SIZE);
  } catch {
    return [];
  }
}

function saveDataset(dataset) {
  setStorageItem(STORAGE_KEY, JSON.stringify(dataset));
}

function createDefaultAlgorithmStats() {
  return ACTIVE_ALGORITHM_IDS.map((id) => ({ id, attempts: 0, correct: 0 }));
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

function sampleVectorBilinear(vector, x, y) {
  if (x < 0 || x > GRID_SIZE - 1 || y < 0 || y > GRID_SIZE - 1) return 0;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(GRID_SIZE - 1, x0 + 1);
  const y1 = Math.min(GRID_SIZE - 1, y0 + 1);
  const dx = x - x0;
  const dy = y - y0;

  const v00 = vector[y0 * GRID_SIZE + x0] || 0;
  const v10 = vector[y0 * GRID_SIZE + x1] || 0;
  const v01 = vector[y1 * GRID_SIZE + x0] || 0;
  const v11 = vector[y1 * GRID_SIZE + x1] || 0;

  const top = v00 * (1 - dx) + v10 * dx;
  const bottom = v01 * (1 - dx) + v11 * dx;
  return top * (1 - dy) + bottom * dy;
}

function transformVector(vector, { translateX = 0, translateY = 0, angle = 0, scale = 1 } = {}) {
  const center = (GRID_SIZE - 1) / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const safeScale = Math.max(scale, 0.01);
  const output = new Array(vector.length).fill(0);

  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const tx = x - center - translateX;
      const ty = y - center - translateY;
      const scaledX = tx / safeScale;
      const scaledY = ty / safeScale;
      const sourceX = center + (scaledX * cos + scaledY * sin);
      const sourceY = center + (-scaledX * sin + scaledY * cos);
      output[y * GRID_SIZE + x] = sampleVectorBilinear(vector, sourceX, sourceY);
    }
  }

  return output;
}

function centroid(vector) {
  let weight = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < GRID_SIZE; y += 1) {
    for (let x = 0; x < GRID_SIZE; x += 1) {
      const value = Math.max(0, vector[y * GRID_SIZE + x] || 0);
      if (value <= 0.01) continue;
      weight += value;
      sumX += x * value;
      sumY += y * value;
    }
  }

  if (weight <= 0.0001) {
    const mid = (GRID_SIZE - 1) / 2;
    return { x: mid, y: mid };
  }

  return { x: sumX / weight, y: sumY / weight };
}

function centroidForSize(vector, size) {
  let weight = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = Math.max(0, vector[y * size + x] || 0);
      if (value <= 0.01) continue;
      weight += value;
      sumX += x * value;
      sumY += y * value;
    }
  }

  if (weight <= 0.0001) {
    const mid = (size - 1) / 2;
    return { x: mid, y: mid };
  }

  return { x: sumX / weight, y: sumY / weight };
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

function resizeVector(vector, fromSize, toSize) {
  if (fromSize === toSize) return [...vector];

  const output = new Array(toSize * toSize).fill(0);
  for (let y = 0; y < toSize; y += 1) {
    const sourceY = Math.min(fromSize - 1, Math.floor((y / Math.max(1, toSize - 1)) * (fromSize - 1)));
    for (let x = 0; x < toSize; x += 1) {
      const sourceX = Math.min(fromSize - 1, Math.floor((x / Math.max(1, toSize - 1)) * (fromSize - 1)));
      output[y * toSize + x] = vector[sourceY * fromSize + sourceX] || 0;
    }
  }
  return output;
}

function boundingBoxForSize(vector, size) {
  let minX = size;
  let maxX = -1;
  let minY = size;
  let maxY = -1;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = vector[y * size + x];
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

function normalizeVectorForSize(vector, size) {
  const box = boundingBoxForSize(vector, size);
  if (!box) return vector;

  const width = box.maxX - box.minX + 1;
  const height = box.maxY - box.minY + 1;
  const scale = Math.max(width, height);
  const output = new Array(size * size).fill(0);
  const offsetX = Math.floor((size - scale) / 2);
  const offsetY = Math.floor((size - scale) / 2);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const sourceX = box.minX + ((x - offsetX) / scale) * width;
      const sourceY = box.minY + ((y - offsetY) / scale) * height;
      const ix = Math.floor(sourceX);
      const iy = Math.floor(sourceY);
      if (ix < box.minX || ix > box.maxX || iy < box.minY || iy > box.maxY) continue;
      const value = vector[iy * size + ix];
      output[y * size + x] = value > 0.05 ? value : 0;
    }
  }

  return output;
}

function rotateVector90ForSize(vector, size) {
  const output = new Array(vector.length).fill(0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      output[x * size + (size - 1 - y)] = vector[y * size + x];
    }
  }
  return output;
}

function generateTransformVariantsForSize(vector, size) {
  const rot0 = vector;
  const rot90 = rotateVector90ForSize(rot0, size);
  const rot180 = rotateVector90ForSize(rot90, size);
  const rot270 = rotateVector90ForSize(rot180, size);
  const flipped = new Array(vector.length).fill(0);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      flipped[y * size + (size - 1 - x)] = vector[y * size + x];
    }
  }

  const flipped90 = rotateVector90ForSize(flipped, size);
  const flipped180 = rotateVector90ForSize(flipped90, size);
  const flipped270 = rotateVector90ForSize(flipped180, size);

  return [rot0, rot90, rot180, rot270, flipped, flipped90, flipped180, flipped270];
}

function sampleBilinear(vector, size, x, y) {
  if (x < 0 || y < 0 || x > size - 1 || y > size - 1) return 0;

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;

  const a = vector[y0 * size + x0] || 0;
  const b = vector[y0 * size + x1] || 0;
  const c = vector[y1 * size + x0] || 0;
  const d = vector[y1 * size + x1] || 0;

  const top = a * (1 - tx) + b * tx;
  const bottom = c * (1 - tx) + d * tx;
  return top * (1 - ty) + bottom * ty;
}

function canonicalizeByMoments(vector, size = GRID_SIZE, r0 = 9) {
  let m00 = 0;
  let xSum = 0;
  let ySum = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const w = vector[y * size + x] || 0;
      m00 += w;
      xSum += x * w;
      ySum += y * w;
    }
  }

  if (m00 < 1e-4) return [...vector];

  const centroidX = xSum / m00;
  const centroidY = ySum / m00;

  let mu20 = 0;
  let mu02 = 0;
  let mu11 = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const w = vector[y * size + x] || 0;
      const u = x - centroidX;
      const v = y - centroidY;
      mu20 += u * u * w;
      mu02 += v * v * w;
      mu11 += u * v * w;
    }
  }

  const theta = 0.5 * Math.atan2(2 * mu11, mu20 - mu02);
  const radius = Math.sqrt((mu20 + mu02) / Math.max(m00, 1e-6));
  const scale = r0 / Math.max(radius, 1e-4);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const center = (size - 1) / 2;

  const output = new Array(size * size).fill(0);
  for (let yo = 0; yo < size; yo += 1) {
    for (let xo = 0; xo < size; xo += 1) {
      const px = xo - center;
      const py = yo - center;
      const invX = px / scale;
      const invY = py / scale;
      const srcX = cosT * invX - sinT * invY + centroidX;
      const srcY = sinT * invX + cosT * invY + centroidY;
      output[yo * size + xo] = sampleBilinear(vector, size, srcX, srcY);
    }
  }

  return output;
}

function blur3x3(vector, size) {
  const output = new Array(size * size).fill(0);
  const kernel = [1, 2, 1];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      let total = 0;
      let weight = 0;

      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const sx = x + kx;
          const sy = y + ky;
          if (sx < 0 || sy < 0 || sx >= size || sy >= size) continue;
          const w = kernel[kx + 1] * kernel[ky + 1];
          total += (vector[sy * size + sx] || 0) * w;
          weight += w;
        }
      }

      output[y * size + x] = total / Math.max(1, weight);
    }
  }

  return output;
}

function sobelEdges(vector, size, threshold = 0.14) {
  const edges = new Array(size * size).fill(0);

  for (let y = 1; y < size - 1; y += 1) {
    for (let x = 1; x < size - 1; x += 1) {
      const p = (dx, dy) => vector[(y + dy) * size + (x + dx)] || 0;
      const gx = -p(-1, -1) + p(1, -1) - 2 * p(-1, 0) + 2 * p(1, 0) - p(-1, 1) + p(1, 1);
      const gy = p(-1, -1) + 2 * p(0, -1) + p(1, -1) - p(-1, 1) - 2 * p(0, 1) - p(1, 1);
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > threshold) edges[y * size + x] = 1;
    }
  }

  return edges;
}

function distanceTransformChamfer(edgeMap, size) {
  const INF = 1e6;
  const dt = edgeMap.map((v) => (v > 0 ? 0 : INF));
  const sqrt2 = Math.SQRT2;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = y * size + x;
      let best = dt[i];
      if (x > 0) best = Math.min(best, dt[i - 1] + 1);
      if (y > 0) best = Math.min(best, dt[i - size] + 1);
      if (x > 0 && y > 0) best = Math.min(best, dt[i - size - 1] + sqrt2);
      if (x < size - 1 && y > 0) best = Math.min(best, dt[i - size + 1] + sqrt2);
      dt[i] = best;
    }
  }

  for (let y = size - 1; y >= 0; y -= 1) {
    for (let x = size - 1; x >= 0; x -= 1) {
      const i = y * size + x;
      let best = dt[i];
      if (x < size - 1) best = Math.min(best, dt[i + 1] + 1);
      if (y < size - 1) best = Math.min(best, dt[i + size] + 1);
      if (x < size - 1 && y < size - 1) best = Math.min(best, dt[i + size + 1] + sqrt2);
      if (x > 0 && y < size - 1) best = Math.min(best, dt[i + size - 1] + sqrt2);
      dt[i] = best;
    }
  }

  return dt;
}

function chamferDistance(edgeA, dtB, edgeB, dtA) {
  let sumA = 0;
  let countA = 0;
  for (let i = 0; i < edgeA.length; i += 1) {
    if (edgeA[i] > 0) {
      sumA += dtB[i];
      countA += 1;
    }
  }

  let sumB = 0;
  let countB = 0;
  for (let i = 0; i < edgeB.length; i += 1) {
    if (edgeB[i] > 0) {
      sumB += dtA[i];
      countB += 1;
    }
  }

  const aTerm = countA > 0 ? sumA / countA : 0;
  const bTerm = countB > 0 ? sumB / countB : 0;
  return aTerm + bTerm;
}

function hogLite(vector, size, bins = 8) {
  const hist = new Array(bins).fill(0);
  for (let y = 1; y < size - 1; y += 1) {
    for (let x = 1; x < size - 1; x += 1) {
      const gx = (vector[y * size + (x + 1)] || 0) - (vector[y * size + (x - 1)] || 0);
      const gy = (vector[(y + 1) * size + x] || 0) - (vector[(y - 1) * size + x] || 0);
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag < 1e-6) continue;
      const angle = Math.atan2(gy, gx);
      const t = ((angle + Math.PI) / (2 * Math.PI)) * bins;
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(t)));
      hist[idx] += mag;
    }
  }

  const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
  return hist.map((value) => value / (norm + 1e-6));
}

function distanceAlgo28(inputFeatures, sampleFeatures, options = {}) {
  const { alpha = 0.35, beta = 0.45, gamma = 0.2 } = options;
  const n = Math.max(1, inputFeatures.blurred.length);
  const k = Math.max(1, inputFeatures.hog.length);

  let pixSum = 0;
  for (let i = 0; i < inputFeatures.blurred.length; i += 1) {
    const d = inputFeatures.blurred[i] - sampleFeatures.blurred[i];
    pixSum += d * d;
  }
  const dPix = Math.sqrt(pixSum / n);

  const dCh = chamferDistance(inputFeatures.edge, sampleFeatures.dt, sampleFeatures.edge, inputFeatures.dt) / Math.max(1, inputFeatures.size);

  let hogSum = 0;
  for (let i = 0; i < inputFeatures.hog.length; i += 1) {
    const d = inputFeatures.hog[i] - sampleFeatures.hog[i];
    hogSum += d * d;
  }
  const dHog = Math.sqrt(hogSum / k);

  return alpha * dPix + beta * dCh + gamma * dHog;
}

function buildAlgo28VariantFeatures(baseCanonical, size) {
  return generateTransformVariantsForSize(baseCanonical, size).map((variant) => {
    const blurred = blur3x3(variant, size);
    const edge = sobelEdges(variant, size, 0.14);
    const dt = distanceTransformChamfer(edge, size);
    const hog = hogLite(variant, size, 8);
    return { size, blurred, edge, dt, hog };
  });
}

function scoreAlgo28(input, dataset16, options = {}) {
  const { k = 21, distanceFloor = 0.01, targetRadius = 9 } = options;
  const size = GRID_SIZE;
  const inputCanonical = canonicalizeByMoments(input, size, targetRadius);
  const inputVariants = buildAlgo28VariantFeatures(inputCanonical, size);

  const scored = dataset16.map((item) => {
    const sampleCanonical = canonicalizeByMoments(item.vector, size, targetRadius);
    const sampleVariants = buildAlgo28VariantFeatures(sampleCanonical, size);
    let bestDistance = Number.POSITIVE_INFINITY;

    inputVariants.forEach((inVariant) => {
      sampleVariants.forEach((sampleVariant) => {
        bestDistance = Math.min(bestDistance, distanceAlgo28(inVariant, sampleVariant));
      });
    });

    return { label: item.label, distance: bestDistance };
  }).sort((a, b) => a.distance - b.distance);

  return voteByInverseDistance(
    scored.map((item) => ({
      label: item.label,
      distance: Math.max(distanceFloor, item.distance),
    })),
    k
  );
}

function extractInvariantShapeDescriptor(vector, size) {
  const normalized = normalizeVectorForSize(vector, size);
  const binary = normalized.map((value) => (value >= 0.2 ? 1 : 0));
  const radialBins = new Array(12).fill(0);
  let m00 = 0;
  let m10 = 0;
  let m01 = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = binary[y * size + x];
      m00 += value;
      m10 += value * x;
      m01 += value * y;
    }
  }

  if (m00 === 0) {
    return [0, ...radialBins, 0, 0, 0, 0, 0, 0, 0];
  }

  const cx = m10 / m00;
  const cy = m01 / m00;
  const safeScale = Math.max(1, size - 1);
  const moments = { "20": 0, "02": 0, "11": 0, "30": 0, "03": 0, "21": 0, "12": 0 };

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = binary[y * size + x];
      if (!value) continue;

      const dx = x - cx;
      const dy = y - cy;
      const radius = Math.sqrt(dx * dx + dy * dy) / safeScale;
      const bin = Math.min(radialBins.length - 1, Math.floor(radius * radialBins.length));
      radialBins[bin] += 1;

      moments["20"] += dx * dx;
      moments["02"] += dy * dy;
      moments["11"] += dx * dy;
      moments["30"] += dx * dx * dx;
      moments["03"] += dy * dy * dy;
      moments["21"] += dx * dx * dy;
      moments["12"] += dx * dy * dy;
    }
  }

  for (let i = 0; i < radialBins.length; i += 1) {
    radialBins[i] /= m00;
  }

  const eta = (p, q) => {
    const key = `${p}${q}`;
    const gamma = (p + q) / 2 + 1;
    return moments[key] / Math.pow(m00, gamma);
  };

  const n20 = eta(2, 0);
  const n02 = eta(0, 2);
  const n11 = eta(1, 1);
  const n30 = eta(3, 0);
  const n03 = eta(0, 3);
  const n21 = eta(2, 1);
  const n12 = eta(1, 2);

  const hu = [
    n20 + n02,
    (n20 - n02) ** 2 + 4 * n11 ** 2,
    (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
    (n30 + n12) ** 2 + (n21 + n03) ** 2,
    (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
      (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
    (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
      (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
  ].map((value) => {
    const absValue = Math.abs(value);
    if (absValue < 1e-12) return 0;
    return Math.log10(absValue + 1e-12);
  });

  const occupancy = m00 / (size * size);
  return [occupancy, ...radialBins, ...hu];
}

function scoreAlgo29(input, dataset16, options = {}) {
  const { k = 27, distanceFloor = 0.008 } = options;
  const inputDescriptor = extractInvariantShapeDescriptor(input, GRID_SIZE);

  const scored = dataset16
    .map((item) => ({
      label: item.label,
      distance: featureDistance(inputDescriptor, extractInvariantShapeDescriptor(item.vector, GRID_SIZE)),
    }))
    .sort((a, b) => a.distance - b.distance);

  return voteByInverseDistance(
    scored.map((entry) => ({
      label: entry.label,
      distance: Math.max(distanceFloor, entry.distance),
    })),
    k
  );
}

function scoreAlgo30(input16, dataset16) {
  const candidates = [
    {
      ...scoreTransformInvariantModelForSize(input16, dataset16, GRID_SIZE, {
        k: 35,
        distanceFloor: 0.005,
        featureWeight: 0.34,
        centerWeightPower: 1.5,
      }),
      weight: 1.35,
    },
    {
      ...scoreAlgo28(input16, dataset16, {
        k: 33,
        distanceFloor: 0.007,
        targetRadius: 9,
      }),
      weight: 1.2,
    },
    {
      ...scoreAlgo29(input16, dataset16, {
        k: 35,
        distanceFloor: 0.007,
      }),
      weight: 1.15,
    },
    {
      ...scoreTransformInvariantModelForSize(input16, dataset16, GRID_SIZE, {
        k: 29,
        distanceFloor: 0.01,
        featureWeight: 0.36,
        centerWeightPower: 1.1,
      }),
      weight: 1,
    },
  ];

  const labelScores = candidates.reduce((acc, model) => {
    const confidenceWeight = 0.3 + (Math.max(0, model.confidence || 0) / 100);
    const vote = model.weight * confidenceWeight;
    acc[model.label] = (acc[model.label] || 0) + vote;
    return acc;
  }, {});

  const ranked = Object.entries(labelScores).sort((a, b) => b[1] - a[1]);
  const probabilities = softmax(ranked.map(([, score]) => score));
  return {
    label: ranked[0]?.[0] || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((probabilities[0] || 0) * 100))),
  };
}

function scoreAlgo31(input16, dataset16) {
  const experts = [
    {
      ...scoreAlgo30(input16, dataset16),
      weight: 1.55,
    },
    {
      ...scoreTransformInvariantModelForSize(input16, dataset16, GRID_SIZE, {
        k: 39,
        distanceFloor: 0.004,
        featureWeight: 0.38,
        centerWeightPower: 1.9,
      }),
      weight: 1.35,
    },
    {
      ...scoreAlgo28(input16, dataset16, {
        k: 37,
        distanceFloor: 0.006,
        targetRadius: 9,
      }),
      weight: 1.2,
    },
    {
      ...scoreAlgo29(input16, dataset16, {
        k: 39,
        distanceFloor: 0.006,
      }),
      weight: 1.15,
    },
    {
      ...scoreTransformInvariantModel(input16, dataset16, {
        k: 27,
        distanceFloor: 0.01,
        featureWeight: 0.42,
        centerWeightPower: 2,
      }),
      weight: 1,
    },
  ];

  const voteCountByLabel = experts.reduce((acc, expert) => {
    acc[expert.label] = (acc[expert.label] || 0) + 1;
    return acc;
  }, {});

  const labelScores = experts.reduce((acc, expert) => {
    const confidence = Math.max(0, expert.confidence || 0) / 100;
    const agreementBoost = 1 + ((voteCountByLabel[expert.label] || 1) - 1) * 0.18;
    const vote = expert.weight * (0.45 + confidence) * agreementBoost;
    acc[expert.label] = (acc[expert.label] || 0) + vote;
    return acc;
  }, {});

  const ranked = Object.entries(labelScores).sort((a, b) => b[1] - a[1]);
  const probabilities = softmax(ranked.map(([, score]) => score));
  const winnerVotes = voteCountByLabel[ranked[0]?.[0]] || 1;
  const margin = Math.max(0, (probabilities[0] || 0) - (probabilities[1] || 0));

  return {
    label: ranked[0]?.[0] || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((probabilities[0] || 0) * 100 + winnerVotes * 2 + margin * 18))),
  };
}

function buildRaesDescriptor(vector, size = GRID_SIZE, radialBins = 8, angleBins = 16) {
  const normalized = normalizeVectorForSize(vector, size);
  const center = centroidForSize(normalized, size);

  let totalInk = 0;
  let radiusMoment = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = Math.max(0, normalized[y * size + x] || 0);
      if (value <= 0.01) continue;
      const dx = x - center.x;
      const dy = y - center.y;
      totalInk += value;
      radiusMoment += (dx * dx + dy * dy) * value;
    }
  }

  const effectiveRadius = Math.sqrt(radiusMoment / Math.max(totalInk, 1e-6)) + 1e-6;
  const hist = new Array(radialBins * angleBins).fill(0);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = Math.max(0, normalized[y * size + x] || 0);
      if (value <= 0.01) continue;

      const dx = x - center.x;
      const dy = y - center.y;
      const normalizedRadius = Math.min(0.9999, Math.sqrt(dx * dx + dy * dy) / effectiveRadius);
      const angle = Math.atan2(dy, dx);
      const angleUnit = (angle + Math.PI) / (2 * Math.PI);

      const radialIndex = Math.min(radialBins - 1, Math.floor(normalizedRadius * radialBins));
      const angleIndex = Math.min(angleBins - 1, Math.floor(angleUnit * angleBins));
      hist[radialIndex * angleBins + angleIndex] += value;
    }
  }

  const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
  const normalizedHist = hist.map((value) => value / Math.max(norm, 1e-6));

  return {
    hist: normalizedHist,
    angleBins,
    radialBins,
    inkDensity: totalInk / Math.max(1, size * size),
  };
}

function flipRaesAngles(hist, radialBins, angleBins) {
  const flipped = new Array(hist.length).fill(0);
  for (let r = 0; r < radialBins; r += 1) {
    for (let a = 0; a < angleBins; a += 1) {
      const targetA = (angleBins - a) % angleBins;
      flipped[r * angleBins + targetA] = hist[r * angleBins + a];
    }
  }
  return flipped;
}

function raesRotationalDistance(histA, histB, radialBins, angleBins) {
  const featureCount = radialBins * angleBins;
  let best = Number.POSITIVE_INFINITY;

  for (let shift = 0; shift < angleBins; shift += 1) {
    let sum = 0;
    for (let r = 0; r < radialBins; r += 1) {
      const base = r * angleBins;
      for (let a = 0; a < angleBins; a += 1) {
        const shiftedIndex = base + ((a + shift) % angleBins);
        const d = histA[base + a] - histB[shiftedIndex];
        sum += d * d;
      }
    }
    best = Math.min(best, Math.sqrt(sum / Math.max(1, featureCount)));
  }

  return best;
}

function raesInvariantDistance(descA, descB) {
  const direct = raesRotationalDistance(descA.hist, descB.hist, descA.radialBins, descA.angleBins);
  const flippedHist = flipRaesAngles(descB.hist, descA.radialBins, descA.angleBins);
  const flipped = raesRotationalDistance(descA.hist, flippedHist, descA.radialBins, descA.angleBins);
  const shapeDistance = Math.min(direct, flipped);
  const densityDistance = Math.abs(descA.inkDensity - descB.inkDensity);
  return shapeDistance * 0.9 + densityDistance * 0.1;
}

function scoreAlgo64(input16, dataset16, options = {}) {
  const {
    radialBins = 8,
    angleBins = 16,
    topLabels = 8,
    k = 19,
    distanceFloor = 0.01,
  } = options;

  const inputTransforms = generateTransformVariantsForSize(input16, GRID_SIZE);
  const inputDescriptors = inputTransforms.map((variant) => buildRaesDescriptor(variant, GRID_SIZE, radialBins, angleBins));
  const descriptors = dataset16.map((item) => ({
    label: item.label,
    desc: buildRaesDescriptor(item.vector, GRID_SIZE, radialBins, angleBins),
  }));

  const grouped = descriptors.reduce((acc, item) => {
    if (!acc[item.label]) {
      acc[item.label] = {
        count: 0,
        hist: new Array(radialBins * angleBins).fill(0),
        inkDensity: 0,
      };
    }
    acc[item.label].count += 1;
    acc[item.label].inkDensity += item.desc.inkDensity;
    for (let i = 0; i < acc[item.label].hist.length; i += 1) {
      acc[item.label].hist[i] += item.desc.hist[i];
    }
    return acc;
  }, {});

  const prototypeRanked = Object.entries(grouped)
    .map(([label, proto]) => {
      const count = Math.max(1, proto.count);
      const hist = proto.hist.map((value) => value / count);
      const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
      const prototypeDesc = {
        hist: hist.map((value) => value / Math.max(norm, 1e-6)),
        radialBins,
        angleBins,
        inkDensity: proto.inkDensity / count,
      };

      const bestDistance = inputDescriptors.reduce(
        (best, inputDesc) => Math.min(best, raesInvariantDistance(inputDesc, prototypeDesc)),
        Number.POSITIVE_INFINITY
      );

      return {
        label,
        distance: bestDistance,
      };
    })
    .sort((a, b) => a.distance - b.distance);

  const candidateLabels = new Set(prototypeRanked.slice(0, Math.min(topLabels, prototypeRanked.length)).map((entry) => entry.label));
  const scored = descriptors
    .filter((item) => candidateLabels.has(item.label))
    .map((item) => {
      const bestDistance = inputDescriptors.reduce(
        (best, inputDesc) => Math.min(best, raesInvariantDistance(inputDesc, item.desc)),
        Number.POSITIVE_INFINITY
      );
      return {
        label: item.label,
        distance: Math.max(distanceFloor, bestDistance),
      };
    })
    .sort((a, b) => a.distance - b.distance);

  return voteByInverseDistance(scored, k);
}

function scoreAlgo65(input16, dataset16, options = {}) {
  const {
    radialBins = 8,
    angleBins = 16,
    topLabels = 8,
    k = 17,
    distanceFloor = 0.01,
    logRadiusPower = 15,
    lineBlend = 0.08,
    densityWeight = 0.03,
    temperature = 2.3,
  } = options;

  const inputDesc = buildRaesDescriptor(input16, GRID_SIZE, radialBins, angleBins);
  const inputFeatures = extractLineFeaturesForSize(input16, GRID_SIZE);
  const descriptors = dataset16.map((item) => ({
    label: item.label,
    desc: buildRaesDescriptor(item.vector, GRID_SIZE, radialBins, angleBins),
    features: extractLineFeaturesForSize(item.vector, GRID_SIZE),
  }));

  const remapLogPolar = (hist) => {
    if (logRadiusPower <= 0) return hist;
    const output = new Array(hist.length).fill(0);
    for (let r = 0; r < radialBins; r += 1) {
      const radialUnit = (r + 0.5) / radialBins;
      const mappedUnit = Math.log1p(radialUnit * logRadiusPower) / Math.log1p(logRadiusPower);
      const targetR = Math.min(radialBins - 1, Math.floor(mappedUnit * radialBins));
      for (let a = 0; a < angleBins; a += 1) {
        output[targetR * angleBins + a] += hist[r * angleBins + a];
      }
    }
    const norm = Math.sqrt(output.reduce((sum, value) => sum + value * value, 0));
    return output.map((value) => value / Math.max(norm, 1e-6));
  };

  const grouped = descriptors.reduce((acc, item) => {
    if (!acc[item.label]) {
      acc[item.label] = {
        count: 0,
        hist: new Array(radialBins * angleBins).fill(0),
        inkDensity: 0,
        features: new Array(inputFeatures.length).fill(0),
      };
    }

    const logHist = remapLogPolar(item.desc.hist);
    acc[item.label].count += 1;
    acc[item.label].inkDensity += item.desc.inkDensity;
    for (let i = 0; i < acc[item.label].hist.length; i += 1) {
      acc[item.label].hist[i] += logHist[i];
    }
    for (let i = 0; i < acc[item.label].features.length; i += 1) {
      acc[item.label].features[i] += item.features[i];
    }
    return acc;
  }, {});

  const inputLogDesc = {
    ...inputDesc,
    hist: remapLogPolar(inputDesc.hist),
  };

  const prototypeRanked = Object.entries(grouped)
    .map(([label, proto]) => {
      const count = Math.max(1, proto.count);
      const hist = proto.hist.map((value) => value / count);
      const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
      const prototypeFeatures = proto.features.map((value) => value / count);
      const prototypeDesc = {
        hist: hist.map((value) => value / Math.max(norm, 1e-6)),
        radialBins,
        angleBins,
        inkDensity: proto.inkDensity / count,
      };

      const raesDistance = raesInvariantDistance(inputLogDesc, prototypeDesc);
      const featureGap = featureDistance(inputFeatures, prototypeFeatures);
      const densityGap = Math.abs(inputDesc.inkDensity - prototypeDesc.inkDensity);
      const distance = raesDistance * (1 - lineBlend) + featureGap * lineBlend + densityGap * densityWeight;

      return { label, distance };
    })
    .sort((a, b) => a.distance - b.distance);

  const candidateLabels = new Set(prototypeRanked.slice(0, Math.min(topLabels, prototypeRanked.length)).map((entry) => entry.label));
  const scored = descriptors
    .filter((item) => candidateLabels.has(item.label))
    .map((item) => {
      const logDesc = {
        ...item.desc,
        hist: remapLogPolar(item.desc.hist),
      };
      const raesDistance = raesInvariantDistance(inputLogDesc, logDesc);
      const featureGap = featureDistance(inputFeatures, item.features);
      const densityGap = Math.abs(inputDesc.inkDensity - item.desc.inkDensity);
      return {
        label: item.label,
        distance: Math.max(distanceFloor, raesDistance * (1 - lineBlend) + featureGap * lineBlend + densityGap * densityWeight),
      };
    })
    .sort((a, b) => a.distance - b.distance);

  const vote = voteByInverseDistance(scored, k);
  const prototypeScores = prototypeRanked.map((entry) => 1 / Math.max(entry.distance, distanceFloor));
  const probabilities = softmax(prototypeScores.map((score) => score * temperature));
  const rankedLabels = prototypeRanked.map((entry) => entry.label);
  const winnerIndex = rankedLabels.indexOf(vote.label);
  const calibrated = Math.round((probabilities[winnerIndex >= 0 ? winnerIndex : 0] || 0) * 100);

  return {
    label: vote.label,
    confidence: Math.max(vote.confidence, calibrated),
  };
}

function scoreAlgo66(input16, dataset16, options = {}) {
  const {
    rotationSteps = 24,
    k = 25,
    topLabels = 10,
    distanceFloor = 0.008,
    lineBlend = 0.2,
    descriptorBlend = 0.28,
    neighborBonus = 0.08,
    temperature = 2.65,
  } = options;

  const inputNormalized = normalizeVector(input16);
  const generateDenseCandidates = (vector) => {
    const horizontalFlip = flipVectorHorizontal(vector);
    const verticalFlip = rotateVector90(rotateVector90(horizontalFlip));
    const flipModes = [vector, horizontalFlip, verticalFlip];
    const candidates = [];

    flipModes.forEach((base) => {
      for (let step = 0; step < rotationSteps; step += 1) {
        const angle = (2 * Math.PI * step) / Math.max(1, rotationSteps);
        candidates.push(transformVector(base, { angle }));
      }
    });

    return candidates;
  };

  const inputCandidates = generateDenseCandidates(inputNormalized);

  const grouped = dataset16.reduce((acc, item) => {
    if (!acc[item.label]) acc[item.label] = [];
    acc[item.label].push(item.vector);
    return acc;
  }, {});

  const labelPrototypes = Object.entries(grouped).map(([label, vectors]) => {
    const proto = new Array(GRID_SIZE * GRID_SIZE).fill(0);
    vectors.forEach((vector) => {
      const norm = normalizeVector(vector);
      for (let i = 0; i < proto.length; i += 1) {
        proto[i] += norm[i] || 0;
      }
    });

    for (let i = 0; i < proto.length; i += 1) {
      proto[i] /= Math.max(1, vectors.length);
    }

    return {
      label,
      vector: proto,
      features: extractLineFeaturesForSize(proto, GRID_SIZE),
      descriptor: extractInvariantShapeDescriptor(proto, GRID_SIZE),
    };
  });

  const labelRanked = labelPrototypes
    .map((prototype) => {
      const bestDistance = inputCandidates.reduce((best, candidate) => {
        const pixelDistance = distance(candidate, prototype.vector) / Math.sqrt(candidate.length);
        const lineDistance = featureDistance(
          extractLineFeaturesForSize(candidate, GRID_SIZE),
          prototype.features
        );
        const descriptorDistance = featureDistance(
          extractInvariantShapeDescriptor(candidate, GRID_SIZE),
          prototype.descriptor
        );

        const blendedDistance =
          pixelDistance * (1 - lineBlend - descriptorBlend) +
          lineDistance * lineBlend +
          descriptorDistance * descriptorBlend;

        return Math.min(best, blendedDistance);
      }, Number.POSITIVE_INFINITY);

      return {
        label: prototype.label,
        distance: Math.max(distanceFloor, bestDistance),
      };
    })
    .sort((a, b) => a.distance - b.distance);

  const candidateLabels = new Set(labelRanked.slice(0, Math.min(topLabels, labelRanked.length)).map((item) => item.label));

  const scoredSamples = dataset16
    .filter((item) => candidateLabels.has(item.label))
    .map((item) => {
      const sample = normalizeVector(item.vector);
      const sampleFeatures = extractLineFeaturesForSize(sample, GRID_SIZE);
      const sampleDescriptor = extractInvariantShapeDescriptor(sample, GRID_SIZE);

      const bestDistance = inputCandidates.reduce((best, candidate) => {
        const pixelDistance = distance(candidate, sample) / Math.sqrt(candidate.length);
        const lineDistance = featureDistance(extractLineFeaturesForSize(candidate, GRID_SIZE), sampleFeatures);
        const descriptorDistance = featureDistance(
          extractInvariantShapeDescriptor(candidate, GRID_SIZE),
          sampleDescriptor
        );
        const blendedDistance =
          pixelDistance * (1 - lineBlend - descriptorBlend) +
          lineDistance * lineBlend +
          descriptorDistance * descriptorBlend;

        return Math.min(best, blendedDistance);
      }, Number.POSITIVE_INFINITY);

      return {
        label: item.label,
        distance: Math.max(distanceFloor, bestDistance),
      };
    })
    .sort((a, b) => a.distance - b.distance);

  const withPrototypeSupport = scoredSamples.map((entry) => {
    const prototypeIndex = labelRanked.findIndex((item) => item.label === entry.label);
    const prototypeBoost = prototypeIndex >= 0 ? neighborBonus / (prototypeIndex + 1) : 0;
    return {
      ...entry,
      distance: Math.max(distanceFloor, entry.distance - prototypeBoost),
    };
  });

  const vote = voteByInverseDistance(withPrototypeSupport, k);
  const scoreByLabel = withPrototypeSupport.reduce((acc, item) => {
    acc[item.label] = (acc[item.label] || 0) + 1 / Math.max(item.distance, distanceFloor);
    return acc;
  }, {});
  const rankedScores = Object.entries(scoreByLabel).sort((a, b) => b[1] - a[1]);
  const probabilities = softmax(rankedScores.map(([, score]) => score * temperature));
  const winnerIndex = rankedScores.findIndex(([label]) => label === vote.label);
  const calibratedConfidence = Math.round((probabilities[winnerIndex >= 0 ? winnerIndex : 0] || 0) * 100);

  return {
    label: vote.label,
    confidence: Math.max(vote.confidence, calibratedConfidence),
  };
}

function buildEdgeMapFromBinary(binary, size) {
  const edge = new Array(size * size).fill(0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = y * size + x;
      if (!binary[i]) continue;
      const left = x > 0 ? binary[i - 1] : 0;
      const right = x < size - 1 ? binary[i + 1] : 0;
      const up = y > 0 ? binary[i - size] : 0;
      const down = y < size - 1 ? binary[i + size] : 0;
      if (!left || !right || !up || !down) edge[i] = 1;
    }
  }
  return edge;
}

function estimateStrokeWidth(binary, edge, size) {
  let area = 0;
  let perimeter = 0;
  for (let i = 0; i < size * size; i += 1) {
    area += binary[i] ? 1 : 0;
    perimeter += edge[i] ? 1 : 0;
  }
  if (!area || !perimeter) return 1;
  return Math.max(0.75, Math.min(4, (2 * area) / perimeter));
}

function extractThicknessCompensatedFeatures(vector, size = GRID_SIZE) {
  const normalized = normalizeVectorForSize(vector, size);
  const binary = normalized.map((value) => (value >= 0.2 ? 1 : 0));
  const edge = buildEdgeMapFromBinary(binary, size);
  const edgeDt = distanceTransformChamfer(edge, size);
  const strokeWidth = estimateStrokeWidth(binary, edge, size);
  const compensated = new Array(size * size).fill(0);

  for (let i = 0; i < compensated.length; i += 1) {
    if (!binary[i]) continue;
    const depth = Math.max(0, edgeDt[i]);
    compensated[i] = 1 / (1 + depth / Math.max(0.6, strokeWidth));
  }

  return { normalized, binary, edge, edgeDt, strokeWidth, compensated };
}

function rotateArray(values, shift) {
  const n = values.length;
  if (!n) return [];
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) out[i] = values[(i + shift + n) % n];
  return out;
}

function minCyclicL2(a, b) {
  if (!a.length || !b.length || a.length !== b.length) return Number.POSITIVE_INFINITY;
  let best = Number.POSITIVE_INFINITY;
  for (let shift = 0; shift < a.length; shift += 1) {
    const shifted = rotateArray(b, shift);
    best = Math.min(best, featureDistance(a, shifted));
    best = Math.min(best, featureDistance(a, [...shifted].reverse()));
  }
  return best;
}

function classifyFromDistances(scored, distanceFloor = 0.01, temperature = 2) {
  const safe = scored.map((item) => ({ label: item.label, distance: Math.max(distanceFloor, item.distance) }));
  const vote = voteByInverseDistance(safe, Math.min(21, safe.length));
  const scores = safe.reduce((acc, item) => {
    acc[item.label] = (acc[item.label] || 0) + 1 / item.distance;
    return acc;
  }, {});
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const probs = softmax(ranked.map(([, score]) => score * temperature));
  const winnerIndex = ranked.findIndex(([label]) => label === vote.label);
  return {
    label: vote.label,
    confidence: Math.max(vote.confidence, Math.round((probs[Math.max(0, winnerIndex)] || 0) * 100)),
  };
}

function scoreAlgo67(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputVariants = generateTransformVariantsForSize(input.compensated, GRID_SIZE);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleVariants = generateTransformVariantsForSize(sample.compensated, GRID_SIZE);
    let best = Number.POSITIVE_INFINITY;
    inputVariants.forEach((a) => {
      sampleVariants.forEach((b) => {
        best = Math.min(best, distance(a, b) / Math.sqrt(a.length));
      });
    });
    return { label: item.label, distance: best + Math.abs(input.strokeWidth - sample.strokeWidth) * 0.02 };
  });
  return classifyFromDistances(scored, 0.01, 2.3);
}

function radialSignatureFromCompensated(comp, size, bins = 18) {
  let cx = 0;
  let cy = 0;
  let mass = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v = comp[y * size + x];
      mass += v;
      cx += x * v;
      cy += y * v;
    }
  }
  if (mass < 1e-6) return new Array(bins).fill(0);
  cx /= mass;
  cy /= mass;
  const sig = new Array(bins).fill(0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v = comp[y * size + x];
      if (v <= 0) continue;
      const dx = x - cx;
      const dy = y - cy;
      const ang = Math.atan2(dy, dx);
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(((ang + Math.PI) / (2 * Math.PI)) * bins)));
      sig[idx] += Math.sqrt(dx * dx + dy * dy) * v;
    }
  }
  const norm = Math.sqrt(sig.reduce((sum, v) => sum + v * v, 0));
  return sig.map((v) => v / Math.max(norm, 1e-6));
}

function scoreAlgo68(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputSig = radialSignatureFromCompensated(input.compensated, GRID_SIZE, 24);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleSig = radialSignatureFromCompensated(sample.compensated, GRID_SIZE, 24);
    const sigDistance = minCyclicL2(inputSig, sampleSig);
    const widthDistance = Math.abs(Math.log((input.strokeWidth + 1e-3) / (sample.strokeWidth + 1e-3))) * 0.08;
    return { label: item.label, distance: sigDistance + widthDistance };
  });
  return classifyFromDistances(scored, 0.01, 2.4);
}

function huLikeMomentsFromCompensated(comp, size) {
  let m00 = 0, m10 = 0, m01 = 0;
  for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) { const w = comp[y * size + x]; m00 += w; m10 += x * w; m01 += y * w; }
  if (m00 < 1e-6) return new Array(7).fill(0);
  const cx = m10 / m00, cy = m01 / m00;
  const mu = {20:0,02:0,11:0,30:0,03:0,21:0,12:0};
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const w = comp[y * size + x]; if (!w) continue;
      const dx = x - cx, dy = y - cy;
      mu[20] += dx*dx*w; mu[02] += dy*dy*w; mu[11] += dx*dy*w;
      mu[30] += dx*dx*dx*w; mu[03] += dy*dy*dy*w; mu[21] += dx*dx*dy*w; mu[12] += dx*dy*dy*w;
    }
  }
  const eta = (p,q,v) => v / Math.pow(m00, 1 + (p+q)/2);
  const n20 = eta(2,0,mu[20]), n02 = eta(0,2,mu[02]), n11 = eta(1,1,mu[11]);
  const n30 = eta(3,0,mu[30]), n03 = eta(0,3,mu[03]), n21 = eta(2,1,mu[21]), n12 = eta(1,2,mu[12]);
  const h1 = n20 + n02;
  const h2 = (n20 - n02) ** 2 + 4 * (n11 ** 2);
  const h3 = (n30 - 3*n12) ** 2 + (3*n21 - n03) ** 2;
  const h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2;
  const h5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2);
  const h6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03);
  const h7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2);
  return [h1,h2,h3,h4,Math.abs(h5),Math.abs(h6),Math.abs(h7)].map((v) => Math.sign(v) * Math.log10(1 + Math.abs(v)));
}

function scoreAlgo69(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputHu = huLikeMomentsFromCompensated(input.compensated, GRID_SIZE);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleHu = huLikeMomentsFromCompensated(sample.compensated, GRID_SIZE);
    return { label: item.label, distance: featureDistance(inputHu, sampleHu) + Math.abs(input.strokeWidth - sample.strokeWidth) * 0.03 };
  });
  return classifyFromDistances(scored, 0.01, 2.2);
}

function ringMassDescriptor(comp, size, rings = 8, sectors = 12) {
  const center = (size - 1) / 2;
  const maxR = Math.sqrt(2) * center;
  const out = new Array(rings * sectors).fill(0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v = comp[y * size + x];
      if (v <= 0) continue;
      const dx = x - center;
      const dy = y - center;
      const rBin = Math.min(rings - 1, Math.floor((Math.sqrt(dx * dx + dy * dy) / maxR) * rings));
      const aBin = Math.min(sectors - 1, Math.max(0, Math.floor(((Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI)) * sectors)));
      out[rBin * sectors + aBin] += v;
    }
  }
  const norm = Math.sqrt(out.reduce((sum, v) => sum + v * v, 0));
  return out.map((v) => v / Math.max(norm, 1e-6));
}

function compareRingDescriptors(a, b, sectors = 12) {
  const rings = Math.floor(a.length / sectors);
  let best = Number.POSITIVE_INFINITY;
  for (let shift = 0; shift < sectors; shift += 1) {
    const shifted = new Array(b.length).fill(0);
    for (let r = 0; r < rings; r += 1) {
      for (let s = 0; s < sectors; s += 1) {
        shifted[r * sectors + s] = b[r * sectors + ((s + shift) % sectors)];
      }
    }
    best = Math.min(best, featureDistance(a, shifted));
    const flipped = new Array(shifted.length).fill(0);
    for (let r = 0; r < rings; r += 1) {
      for (let s = 0; s < sectors; s += 1) {
        flipped[r * sectors + s] = shifted[r * sectors + ((sectors - s) % sectors)];
      }
    }
    best = Math.min(best, featureDistance(a, flipped));
  }
  return best;
}

function scoreAlgo70(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputDesc = ringMassDescriptor(input.compensated, GRID_SIZE, 8, 16);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleDesc = ringMassDescriptor(sample.compensated, GRID_SIZE, 8, 16);
    return { label: item.label, distance: compareRingDescriptors(inputDesc, sampleDesc, 16) };
  });
  return classifyFromDistances(scored, 0.01, 2.4);
}

function projectionSpectrum(comp, size, bins = 16) {
  const rows = new Array(size).fill(0);
  const cols = new Array(size).fill(0);
  for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) { const v = comp[y * size + x]; rows[y] += v; cols[x] += v; }
  const fold = (arr) => {
    const out = new Array(bins).fill(0);
    for (let i = 0; i < arr.length; i += 1) out[Math.min(bins - 1, Math.floor((i / arr.length) * bins))] += arr[i];
    const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0));
    return out.map((v) => v / Math.max(norm, 1e-6));
  };
  return [...fold(rows), ...fold(cols)].sort((a, b) => b - a);
}

function scoreAlgo71(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputSpec = projectionSpectrum(input.compensated, GRID_SIZE, 16);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleSpec = projectionSpectrum(sample.compensated, GRID_SIZE, 16);
    const edgeDt = chamferDistance(input.edge, sample.edgeDt, sample.edge, input.edgeDt) / GRID_SIZE;
    return { label: item.label, distance: featureDistance(inputSpec, sampleSpec) * 0.65 + edgeDt * 0.35 };
  });
  return classifyFromDistances(scored, 0.01, 2.1);
}

function contourPointCloud(edge, size) {
  const points = [];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (!edge[y * size + x]) continue;
      points.push([x / Math.max(1, size - 1), y / Math.max(1, size - 1)]);
    }
  }
  return points;
}

function directedChamfer(pointsA, pointsB) {
  if (!pointsA.length || !pointsB.length) return 1;
  let total = 0;
  pointsA.forEach(([ax, ay]) => {
    let best = Number.POSITIVE_INFINITY;
    pointsB.forEach(([bx, by]) => {
      const dx = ax - bx;
      const dy = ay - by;
      best = Math.min(best, Math.sqrt(dx * dx + dy * dy));
    });
    total += best;
  });
  return total / Math.max(1, pointsA.length);
}

function scoreAlgo72(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputPts = contourPointCloud(input.edge, GRID_SIZE);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const variants = generateTransformVariantsForSize(sample.edge, GRID_SIZE).map((variant) => contourPointCloud(variant.map((v) => (v >= 0.5 ? 1 : 0)), GRID_SIZE));
    let best = Number.POSITIVE_INFINITY;
    variants.forEach((pts) => {
      const d = directedChamfer(inputPts, pts) + directedChamfer(pts, inputPts);
      best = Math.min(best, d);
    });
    return { label: item.label, distance: best };
  });
  return classifyFromDistances(scored, 0.01, 2.3);
}

function frequencyMagnitudeDescriptor(comp, size, low = 5) {
  const mags = [];
  for (let u = 0; u < low; u += 1) {
    for (let v = 0; v < low; v += 1) {
      if (u === 0 && v === 0) continue;
      let re = 0;
      let im = 0;
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const val = comp[y * size + x] || 0;
          const angle = (-2 * Math.PI * ((u * x) / size + (v * y) / size));
          re += val * Math.cos(angle);
          im += val * Math.sin(angle);
        }
      }
      mags.push(Math.sqrt(re * re + im * im));
    }
  }
  mags.sort((a, b) => b - a);
  const norm = Math.sqrt(mags.reduce((sum, value) => sum + value * value, 0));
  return mags.map((v) => v / Math.max(norm, 1e-6));
}

function scoreAlgo73(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputFreq = frequencyMagnitudeDescriptor(input.compensated, GRID_SIZE, 5);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleFreq = frequencyMagnitudeDescriptor(sample.compensated, GRID_SIZE, 5);
    const lineGap = featureDistance(extractLineFeaturesForSize(input.compensated, GRID_SIZE), extractLineFeaturesForSize(sample.compensated, GRID_SIZE));
    return { label: item.label, distance: featureDistance(inputFreq, sampleFreq) * 0.72 + lineGap * 0.28 };
  });
  return classifyFromDistances(scored, 0.01, 2.25);
}

function skeletonProxy(comp, edgeDt, size) {
  const out = new Array(size * size).fill(0);
  for (let y = 1; y < size - 1; y += 1) {
    for (let x = 1; x < size - 1; x += 1) {
      const i = y * size + x;
      if (comp[i] <= 0) continue;
      const d = edgeDt[i];
      if (d <= 0) continue;
      let isPeak = true;
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (!ox && !oy) continue;
          if (edgeDt[(y + oy) * size + (x + ox)] > d) isPeak = false;
        }
      }
      if (isPeak) out[i] = 1;
    }
  }
  return out;
}

function scoreAlgo74(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputSkeleton = skeletonProxy(input.compensated, input.edgeDt, GRID_SIZE);
  const inputDesc = ringMassDescriptor(inputSkeleton, GRID_SIZE, 7, 14);
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const sampleSkeleton = skeletonProxy(sample.compensated, sample.edgeDt, GRID_SIZE);
    const sampleDesc = ringMassDescriptor(sampleSkeleton, GRID_SIZE, 7, 14);
    return { label: item.label, distance: compareRingDescriptors(inputDesc, sampleDesc, 14) };
  });
  return classifyFromDistances(scored, 0.01, 2.2);
}

function scoreAlgo75(input16, dataset16) {
  const input = extractThicknessCompensatedFeatures(input16);
  const inputInv = extractInvariantShapeDescriptor(input.compensated, GRID_SIZE);
  const inputRaes = extractRAESDescriptorForSize(input.compensated, GRID_SIZE, { radialBins: 7, angleBins: 20 });
  const scored = dataset16.map((item) => {
    const sample = extractThicknessCompensatedFeatures(item.vector);
    const dInv = featureDistance(inputInv, extractInvariantShapeDescriptor(sample.compensated, GRID_SIZE));
    const dRaes = raesInvariantDistance(inputRaes, extractRAESDescriptorForSize(sample.compensated, GRID_SIZE, { radialBins: 7, angleBins: 20 }));
    const widthGap = Math.abs(Math.log((input.strokeWidth + 0.1) / (sample.strokeWidth + 0.1)));
    return { label: item.label, distance: dInv * 0.42 + dRaes * 0.5 + widthGap * 0.08 };
  });
  return classifyFromDistances(scored, 0.01, 2.55);
}

function scoreAlgo76(input16, dataset16) {
  const models = [
    { model: scoreAlgo67(input16, dataset16), weight: 1.05 },
    { model: scoreAlgo68(input16, dataset16), weight: 1.0 },
    { model: scoreAlgo69(input16, dataset16), weight: 0.95 },
    { model: scoreAlgo70(input16, dataset16), weight: 1.0 },
    { model: scoreAlgo71(input16, dataset16), weight: 0.9 },
    { model: scoreAlgo72(input16, dataset16), weight: 1.0 },
    { model: scoreAlgo73(input16, dataset16), weight: 0.95 },
    { model: scoreAlgo74(input16, dataset16), weight: 0.9 },
    { model: scoreAlgo75(input16, dataset16), weight: 1.05 },
  ];

  const scores = {};
  models.forEach(({ model, weight }) => {
    const confidenceWeight = Math.max(0.2, model.confidence / 100);
    scores[model.label] = (scores[model.label] || 0) + weight * confidenceWeight;
  });

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const probs = softmax(ranked.map(([, score]) => score * 2.4));
  return {
    label: ranked[0]?.[0] || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((probs[0] || 0) * 100))),
  };
}


function extractLineFeaturesForSize(vector, size) {
  const norm = normalizeVectorForSize(vector, size);
  const binary = norm.map((value) => (value >= 0.25 ? 1 : 0));
  const rowSums = new Array(size).fill(0);
  const colSums = new Array(size).fill(0);
  let hTransitions = 0;
  let vTransitions = 0;
  let d1Transitions = 0;
  let d2Transitions = 0;
  let active = 0;
  let cx = 0;
  let cy = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = y * size + x;
      const value = binary[index];
      rowSums[y] += value;
      colSums[x] += value;
      active += value;
      cx += value * x;
      cy += value * y;

      if (x < size - 1 && value !== binary[index + 1]) hTransitions += 1;
      if (y < size - 1 && value !== binary[index + size]) vTransitions += 1;
      if (x < size - 1 && y < size - 1 && value !== binary[index + size + 1]) d1Transitions += 1;
      if (x > 0 && y < size - 1 && value !== binary[index + size - 1]) d2Transitions += 1;
    }
  }

  const safeActive = Math.max(active, 1);
  return [
    hTransitions / (size * size),
    vTransitions / (size * size),
    d1Transitions / (size * size),
    d2Transitions / (size * size),
    active / (size * size),
    cx / safeActive / size,
    cy / safeActive / size,
  ];
}

function scoreTransformInvariantModelForSize(inputVector, dataset, size, options = {}) {
  const { k = 17, distanceFloor = 0.02, featureWeight = 0.35, centerWeightPower = 0 } = options;
  const inputNorm = normalizeVectorForSize(inputVector, size);
  const inputFeatures = extractLineFeaturesForSize(inputNorm, size);
  const inputCandidates = generateTransformVariantsForSize(inputNorm, size);

  const scored = dataset.map((item) => {
    const base = normalizeVectorForSize(item.vector, size);
    const candidates = generateTransformVariantsForSize(base, size);

    const bestDistance = inputCandidates.reduce((bestInput, inputCandidate) => {
      const bestForInput = candidates.reduce((bestCandidate, candidate) => {
        let d = distance(inputCandidate, candidate) / Math.sqrt(size * size);
        if (centerWeightPower > 0) {
          const box = boundingBoxForSize(candidate, size);
          if (box) {
            const cx = (box.minX + box.maxX) / 2;
            const cy = (box.minY + box.maxY) / 2;
            const centerDx = Math.abs(cx - (size - 1) / 2) / (size / 2);
            const centerDy = Math.abs(cy - (size - 1) / 2) / (size / 2);
            d *= 1 + Math.pow((centerDx + centerDy) / 2, centerWeightPower) * 0.2;
          }
        }
        return Math.min(bestCandidate, d);
      }, Number.POSITIVE_INFINITY);
      return Math.min(bestInput, bestForInput);
    }, Number.POSITIVE_INFINITY);

    const lineDistance = featureDistance(inputFeatures, extractLineFeaturesForSize(base, size));
    return {
      label: item.label,
      distance: bestDistance * (1 - featureWeight) + lineDistance * featureWeight,
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

function runLiveAlgorithms(vector, dataset) {
  if (!dataset.length) {
    return {
      hyperDraw: { label: "Need training data first", confidence: 0 },
      hyperDrawV2: { label: "Need training data first", confidence: 0 },
    };
  }

  const prepared = prepareLiveDataset(dataset);
  return runLiveAlgorithmsPrepared(vector, prepared);
}

function prepareLiveDataset(dataset) {
  const normalizedDataset = dataset.map((item) => ({
    label: item.label,
    normalizedVector: normalizeVector(item.vector),
  }));

  const prototypesNormalized = buildLabelPrototypes(
    normalizedDataset.map((item) => ({ label: item.label, vector: item.normalizedVector }))
  );

  return {
    normalizedDataset,
    prototypesNormalized,
  };
}

function runLiveAlgorithmsPrepared(vector, prepared) {
  const { normalizedDataset, prototypesNormalized } = prepared;

  if (!normalizedDataset.length) {
    return {
      hyperDraw: { label: "Need training data first", confidence: 0 },
      hyperDrawV2: { label: "Need training data first", confidence: 0 },
    };
  }

  const normalizedInput = normalizeVector(vector);
  const normalizedDistances = normalizedDataset
    .map((item) => ({
      label: item.label,
      distance: distance(normalizedInput, item.normalizedVector) / Math.sqrt(vector.length),
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
  const hyperDraw = {
    label: algo1Ranked[0]?.[0] || "unknown",
    confidence: Math.round((algo1Probs[0] || 0) * 100),
  };

  const prototypeNorm = Object.entries(prototypesNormalized)
    .map(([label, proto]) => ({ label, distance: distance(normalizedInput, proto) / Math.sqrt(vector.length) }))
    .sort((a, b) => a.distance - b.distance)[0];

  const hyperDrawV2 = {
    label: prototypeNorm?.label || "unknown",
    confidence: Math.round((1 - Math.min(1, prototypeNorm?.distance || 1)) * 100),
  };

  return { hyperDraw, hyperDrawV2 };
}

function runAlgorithms(vector, dataset) {
  if (!dataset.length) {
    return [
      { id: 1, name: "Algorithm 1 (Current)", label: "Need training data first", confidence: 0 },
      { id: 7, name: "Algorithm 7 (Prototype Normalized)", label: "Need training data first", confidence: 0 },
      { id: 45, name: "Algorithm 45 (Dev: Alg7 + 4NN support)", label: "Need training data first", confidence: 0 },
      { id: 57, name: "Algorithm 57 (Dev: Alg45 + confidence heat)", label: "Need training data first", confidence: 0 },
      { id: 64, name: "Algorithm 64 (Dev: Alg63 + explicit transform parity)", label: "Need training data first", confidence: 0 },
      { id: 65, name: "Algorithm 65 (Dev: Alg57 + log-polar RAES v2)", label: "Need training data first", confidence: 0 },
      { id: 66, name: "Algorithm 66 (Dev: Alg57 + omni-rotation parity)", label: "Need training data first", confidence: 0 },
      { id: 67, name: "Algorithm 67 (Edge-compensated transform lattice)", label: "Need training data first", confidence: 0 },
      { id: 68, name: "Algorithm 68 (Centroid radial signature matcher)", label: "Need training data first", confidence: 0 },
      { id: 69, name: "Algorithm 69 (Hu-like compensated moment invariants)", label: "Need training data first", confidence: 0 },
      { id: 70, name: "Algorithm 70 (Ring-sector mass alignment)", label: "Need training data first", confidence: 0 },
      { id: 71, name: "Algorithm 71 (Projection-spectrum + edge chamfer)", label: "Need training data first", confidence: 0 },
      { id: 72, name: "Algorithm 72 (Contour cloud mirrored chamfer)", label: "Need training data first", confidence: 0 },
      { id: 73, name: "Algorithm 73 (Low-frequency magnitude signature)", label: "Need training data first", confidence: 0 },
      { id: 74, name: "Algorithm 74 (Skeleton proxy ring context)", label: "Need training data first", confidence: 0 },
      { id: 75, name: "Algorithm 75 (Invariant descriptor fusion)", label: "Need training data first", confidence: 0 },
      { id: 76, name: "Algorithm 76 (Consensus of 67-75)", label: "Need training data first", confidence: 0 },
    ];
  }

  const normalizedInput = normalizeVector(vector);
  const prototypesNormalized = buildLabelPrototypes(dataset.map((item) => ({ ...item, vector: normalizeVector(item.vector) })));

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

  const prototypeNorm = Object.entries(prototypesNormalized)
    .map(([label, proto]) => ({ label, distance: distance(normalizedInput, proto) / Math.sqrt(vector.length) }))
    .sort((a, b) => a.distance - b.distance)[0];

  const inputFeatures = extractLineFeaturesForSize(normalizedInput, GRID_SIZE);
  const prototypeFeaturesByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = extractLineFeaturesForSize(prototype, GRID_SIZE);
    return acc;
  }, {});

  const datasetLabelCounts = dataset.reduce((acc, item) => {
    acc[item.label] = (acc[item.label] || 0) + 1;
    return acc;
  }, {});
  const datasetSize = Math.max(1, dataset.length);

  const scoreAlgo7Variant = ({
    lineBlend = 0,
    densityWeight = 0,
    centerWeight = 0,
    neighborDepth = 0,
    balancePenalty = 0,
    temperature = 2,
  }) => {
    const topNeighbors = normalizedDistances.slice(0, Math.min(neighborDepth, normalizedDistances.length));
    const ranked = Object.entries(prototypesNormalized)
      .map(([label, prototype]) => {
        const prototypeDistance = distance(normalizedInput, prototype) / Math.sqrt(vector.length);
        const prototypeFeatures = prototypeFeaturesByLabel[label];
        const lineDistance = featureDistance(inputFeatures, prototypeFeatures);
        const densityGap = Math.abs((inputFeatures[4] || 0) - (prototypeFeatures[4] || 0));
        const centerGap =
          Math.abs((inputFeatures[5] || 0.5) - (prototypeFeatures[5] || 0.5)) +
          Math.abs((inputFeatures[6] || 0.5) - (prototypeFeatures[6] || 0.5));

        const blendedDistance =
          prototypeDistance * (1 - lineBlend) +
          lineDistance * lineBlend +
          densityGap * densityWeight +
          centerGap * centerWeight;

        const baseScore = 1 / Math.max(0.001, blendedDistance + 0.05);
        const neighborScore = topNeighbors.reduce((bonus, neighbor, index) => {
          if (neighbor.label !== label) return bonus;
          return bonus + 0.05 / (index + 1);
        }, 0);
        const priorPenalty = ((datasetLabelCounts[label] || 0) / datasetSize) * balancePenalty;

        return {
          label,
          score: baseScore + neighborScore - priorPenalty,
        };
      })
      .sort((a, b) => b.score - a.score);

    const probabilities = softmax(ranked.map((entry) => entry.score * temperature));
    return {
      label: ranked[0]?.label || "unknown",
      confidence: Math.max(1, Math.min(99, Math.round((probabilities[0] || 0) * 100))),
    };
  };

  const algorithm45 = scoreAlgo7Variant({ neighborDepth: 4 });
  const algorithm57 = scoreAlgo7Variant({ neighborDepth: 4, lineBlend: 0.06, densityWeight: 0.04, centerWeight: 0.03, temperature: 2.35 });
  const algorithm64 = scoreAlgo64(normalizedInput, dataset);
  const algorithm65 = scoreAlgo65(normalizedInput, dataset);
  const algorithm66 = scoreAlgo66(normalizedInput, dataset);
  const algorithm67 = scoreAlgo67(normalizedInput, dataset);
  const algorithm68 = scoreAlgo68(normalizedInput, dataset);
  const algorithm69 = scoreAlgo69(normalizedInput, dataset);
  const algorithm70 = scoreAlgo70(normalizedInput, dataset);
  const algorithm71 = scoreAlgo71(normalizedInput, dataset);
  const algorithm72 = scoreAlgo72(normalizedInput, dataset);
  const algorithm73 = scoreAlgo73(normalizedInput, dataset);
  const algorithm74 = scoreAlgo74(normalizedInput, dataset);
  const algorithm75 = scoreAlgo75(normalizedInput, dataset);
  const algorithm76 = scoreAlgo76(normalizedInput, dataset);

  return [
    { id: 1, name: "Algorithm 1 (Current)", label: algo1Guess, confidence: algo1Confidence },
    { id: 7, name: "Algorithm 7 (Prototype Normalized)", label: prototypeNorm?.label || "unknown", confidence: Math.round((1 - Math.min(1, prototypeNorm?.distance || 1)) * 100) },
    { id: 45, name: "Algorithm 45 (Dev: Alg7 + 4NN support)", label: algorithm45.label, confidence: algorithm45.confidence },
    { id: 57, name: "Algorithm 57 (Dev: Alg45 + confidence heat)", label: algorithm57.label, confidence: algorithm57.confidence },
    { id: 64, name: "Algorithm 64 (Dev: Alg63 + explicit transform parity)", label: algorithm64.label, confidence: algorithm64.confidence },
    { id: 65, name: "Algorithm 65 (Dev: Alg57 + log-polar RAES v2)", label: algorithm65.label, confidence: algorithm65.confidence },
    { id: 66, name: "Algorithm 66 (Dev: Alg57 + omni-rotation parity)", label: algorithm66.label, confidence: algorithm66.confidence },
    { id: 67, name: "Algorithm 67 (Edge-compensated transform lattice)", label: algorithm67.label, confidence: algorithm67.confidence },
    { id: 68, name: "Algorithm 68 (Centroid radial signature matcher)", label: algorithm68.label, confidence: algorithm68.confidence },
    { id: 69, name: "Algorithm 69 (Hu-like compensated moment invariants)", label: algorithm69.label, confidence: algorithm69.confidence },
    { id: 70, name: "Algorithm 70 (Ring-sector mass alignment)", label: algorithm70.label, confidence: algorithm70.confidence },
    { id: 71, name: "Algorithm 71 (Projection-spectrum + edge chamfer)", label: algorithm71.label, confidence: algorithm71.confidence },
    { id: 72, name: "Algorithm 72 (Contour cloud mirrored chamfer)", label: algorithm72.label, confidence: algorithm72.confidence },
    { id: 73, name: "Algorithm 73 (Low-frequency magnitude signature)", label: algorithm73.label, confidence: algorithm73.confidence },
    { id: 74, name: "Algorithm 74 (Skeleton proxy ring context)", label: algorithm74.label, confidence: algorithm74.confidence },
    { id: 75, name: "Algorithm 75 (Invariant descriptor fusion)", label: algorithm75.label, confidence: algorithm75.confidence },
    { id: 76, name: "Algorithm 76 (Consensus of 67-75)", label: algorithm76.label, confidence: algorithm76.confidence },
  ];
}



function App() {
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const strokesRef = useRef([]);
  const activeStrokeRef = useRef(null);
  const lastLiveGuessAtRef = useRef(0);
  const drawingRevisionRef = useRef(0);
  const lastGuessedRevisionRef = useRef(-1);
  const guessTimeoutRef = useRef(null);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => randomPrompt());
  const [selectedModel, setSelectedModel] = useState("hyperdraw_v2");
  const [compareMode, setCompareMode] = useState(false);
  const [guess, setGuess] = useState("start drawing");
  const [compareResults, setCompareResults] = useState({
    hyperDraw: { label: "start drawing" },
    hyperDrawV2: { label: "start drawing" },
  });
  const [compareStats, setCompareStats] = useState(() => loadCompareStats());
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [activeTab, setActiveTab] = useState("draw");
  const [algorithmStats, setAlgorithmStats] = useState(() => loadAlgorithmStats());
  const [sessionAlgorithmStats, setSessionAlgorithmStats] = useState(() => createDefaultAlgorithmStats());
  const [devStatsView, setDevStatsView] = useState("session");
  const [lastDoneResults, setLastDoneResults] = useState([]);
  const preparedLiveDataset = useMemo(() => prepareLiveDataset(dataset), [dataset]);

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
  }, []);

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

    const now = Date.now();
    if (now - lastLiveGuessAtRef.current >= 80) {
      lastLiveGuessAtRef.current = now;
      scheduleGuess();
    }
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
    setCompareResults({
      hyperDraw: { label: "start drawing" },
      hyperDrawV2: { label: "start drawing" },
    });
    setStatusMessage("");
    if (guessTimeoutRef.current) {
      clearTimeout(guessTimeoutRef.current);
      guessTimeoutRef.current = null;
    }
  };

  const vectorizeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return new Array(GRID_SIZE * GRID_SIZE).fill(0);

    if (!offscreenCanvasRef.current) {
      const offscreen = document.createElement("canvas");
      offscreen.width = GRID_SIZE;
      offscreen.height = GRID_SIZE;
      offscreenCanvasRef.current = offscreen;
    }

    const octx = offscreenCanvasRef.current.getContext("2d");
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

    const shouldGuessV2Early = selectedModel === "hyperdraw_v2" || compareMode || devMode;

    if (!drawingStats.hasAnyInk) {
      setStatusMessage("Draw something first — erased/blank canvas cannot be guessed.");
      return;
    }

    if (!drawingStats.hasMeaningfulDrawing && !shouldGuessV2Early) {
      setStatusMessage("Draw a little more for HyperDraw to start guessing.");
      return;
    }

    if (dataset.length === 0) {
      setGuess("Need training data first");
      setStatusMessage("Train me with a few drawings before guessing.");
      setLastDoneResults([]);
      return;
    }

    const { hyperDraw, hyperDrawV2 } = runLiveAlgorithmsPrepared(drawingStats.vec, preparedLiveDataset);
    const selected = selectedModel === "hyperdraw_v2" ? hyperDrawV2 : hyperDraw;

    setGuess(selected.label);
    setCompareResults({
      hyperDraw: { label: hyperDraw.label },
      hyperDrawV2: { label: hyperDrawV2.label },
    });
    if (devMode) {
      setLastDoneResults(runAlgorithms(drawingStats.vec, dataset));
    }
    setStatusMessage("");
  };

  const scheduleGuess = (immediate = false) => {
    if (drawingRevisionRef.current === lastGuessedRevisionRef.current && !immediate) return;

    if (guessTimeoutRef.current) {
      clearTimeout(guessTimeoutRef.current);
      guessTimeoutRef.current = null;
    }

    const delay = immediate ? 0 : 180;
    guessTimeoutRef.current = setTimeout(() => {
      if (drawingRevisionRef.current === lastGuessedRevisionRef.current && !immediate) return;
      lastGuessedRevisionRef.current = drawingRevisionRef.current;
      guessDrawing();
      guessTimeoutRef.current = null;
    }, delay);
  };

  const stopDrawingAndGuess = () => {
    stopDrawing();
    scheduleGuess(true);
  };

  useEffect(
    () => () => {
      if (guessTimeoutRef.current) {
        clearTimeout(guessTimeoutRef.current);
      }
    },
    []
  );

  const saveDrawing = () => {
    const drawingStats = getDrawingStats();

    if (!drawingStats.hasMeaningfulDrawing) {
      setStatusMessage("Nope — draw first. Blank/erased canvas won't be saved to training.");
      return;
    }

    const { vec } = drawingStats;
    const { hyperDraw, hyperDrawV2 } = runLiveAlgorithmsPrepared(vec, preparedLiveDataset);
    const results = runAlgorithms(vec, dataset);

    setCompareResults({
      hyperDraw: { label: hyperDraw.label },
      hyperDrawV2: { label: hyperDrawV2.label },
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
    setSessionAlgorithmStats((previous) =>
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
                <option value="hyperdraw_v2">HyperDraw_v2 (default)</option>
                <option value="hyperdraw">HyperDraw</option>
              </select>
            </label>
            <button className={`secondary ${compareMode ? "active" : ""}`} onClick={() => setCompareMode((on) => !on)}>
              {compareMode ? "Hide Compare" : "Compare"}
            </button>
          </div>
          {!compareMode ? (
            <>
              <p className="big">{guess}</p>
            </>
          ) : (
            <div className="compare-grid">
              <div className="stat">
                <div><strong>HyperDraw</strong></div>
                <div>Guess: {compareResults.hyperDraw.label}</div>
              </div>
              <div className="stat">
                <div><strong>HyperDraw_v2</strong></div>
                <div>Guess: {compareResults.hyperDrawV2.label}</div>
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
              <p>Click <strong>Done</strong> to log correctness rates for all active algorithms (1, 7, 45, 57, 64, 65, 66, and 67-76).</p>
              <div className="row">
                <button
                  className={`secondary ${devStatsView === "session" ? "active" : ""}`}
                  onClick={() => setDevStatsView("session")}
                >
                  Session checks
                </button>
                <button
                  className={`secondary ${devStatsView === "lifetime" ? "active" : ""}`}
                  onClick={() => setDevStatsView("lifetime")}
                >
                  Lifetime checks
                </button>
              </div>
              <p>
                {devStatsView === "session"
                  ? "Session checks reset on reload."
                  : "Lifetime checks are saved in your browser."}
              </p>
              <div className="algo-grid">
                {[...(devStatsView === "session" ? sessionAlgorithmStats : algorithmStats)]
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
                      <div className="stat" key={`${devStatsView}-${algo.id}`}>
                        <div><strong>Algorithm {algo.id}</strong>{algo.id === 1 ? " (live model)" : ""}</div>
                        <div>Guess: {latest?.label || "-"}</div>
                        <div>
                          {devStatsView === "session" ? "Session rate" : "Correctness rate"}: {accuracy}% ({algo.correct}/{algo.attempts})
                        </div>
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
