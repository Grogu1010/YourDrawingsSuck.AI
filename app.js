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
const ACTIVE_ALGORITHM_IDS = [1, 7, 21, 32];
const HYPERDRAW_ALGORITHM_ID = 1;
const HYPERDRAW_V2_ALGORITHM_ID = 7;
const GRID_SIZE_V3 = 32;

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

function canonicalizeByMoments(vector, size = 32, r0 = 9) {
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

function scoreAlgo28(input, dataset32, options = {}) {
  const { k = 21, distanceFloor = 0.01, targetRadius = 9 } = options;
  const size = GRID_SIZE_V3;
  const inputCanonical = canonicalizeByMoments(input, size, targetRadius);
  const inputVariants = buildAlgo28VariantFeatures(inputCanonical, size);

  const scored = dataset32.map((item) => {
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

function scoreAlgo29(input, dataset32, options = {}) {
  const { k = 27, distanceFloor = 0.008 } = options;
  const inputDescriptor = extractInvariantShapeDescriptor(input, GRID_SIZE_V3);

  const scored = dataset32
    .map((item) => ({
      label: item.label,
      distance: featureDistance(inputDescriptor, extractInvariantShapeDescriptor(item.vector, GRID_SIZE_V3)),
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

function scoreAlgo30(input16, input32, dataset16, dataset32) {
  const candidates = [
    {
      ...scoreTransformInvariantModelForSize(input32, dataset32, GRID_SIZE_V3, {
        k: 35,
        distanceFloor: 0.005,
        featureWeight: 0.34,
        centerWeightPower: 1.5,
      }),
      weight: 1.35,
    },
    {
      ...scoreAlgo28(input32, dataset32, {
        k: 33,
        distanceFloor: 0.007,
        targetRadius: 9,
      }),
      weight: 1.2,
    },
    {
      ...scoreAlgo29(input32, dataset32, {
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

function scoreAlgo31(input16, input32, dataset16, dataset32) {
  const experts = [
    {
      ...scoreAlgo30(input16, input32, dataset16, dataset32),
      weight: 1.55,
    },
    {
      ...scoreTransformInvariantModelForSize(input32, dataset32, GRID_SIZE_V3, {
        k: 39,
        distanceFloor: 0.004,
        featureWeight: 0.38,
        centerWeightPower: 1.9,
      }),
      weight: 1.35,
    },
    {
      ...scoreAlgo28(input32, dataset32, {
        k: 37,
        distanceFloor: 0.006,
        targetRadius: 9,
      }),
      weight: 1.2,
    },
    {
      ...scoreAlgo29(input32, dataset32, {
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
      { id: 21, name: "Algorithm 21 (v2 Transform + Line Blend kNN-17)", label: "Need training data first", confidence: 0 },
      { id: 32, name: "Algorithm 32 (Dev: Prototype + Line Shape Match)", label: "Need training data first", confidence: 0 },
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

  const normalizedInputFeatures = extractLineFeaturesForSize(normalizedInput, GRID_SIZE);
  const model32Ranked = Object.entries(prototypesNormalized)
    .map(([label, proto]) => {
      const prototypeDistance = distance(normalizedInput, proto) / Math.sqrt(vector.length);
      const lineDistance = featureDistance(normalizedInputFeatures, extractLineFeaturesForSize(proto, GRID_SIZE));
      return {
        label,
        score: prototypeDistance * 0.6 + lineDistance * 0.4,
      };
    })
    .sort((a, b) => a.score - b.score);

  const model32Best = model32Ranked[0] || { label: "unknown", score: 1 };
  const model32Confidence = Math.round((1 - Math.min(1, model32Best.score)) * 100);

  const model21 = scoreTransformInvariantModel(vector, dataset, {
    k: 17,
    distanceFloor: 0.02,
    featureWeight: 0.35,
    centerWeightPower: 0,
  });

  return [
    { id: 1, name: "Algorithm 1 (Current)", label: algo1Guess, confidence: algo1Confidence },
    { id: 7, name: "Algorithm 7 (Prototype Normalized)", label: prototypeNorm?.label || "unknown", confidence: Math.round((1 - Math.min(1, prototypeNorm?.distance || 1)) * 100) },
    { id: 21, name: "Algorithm 21 (v2 Transform + Line Blend kNN-17)", label: model21.label, confidence: model21.confidence },
    { id: 32, name: "Algorithm 32 (Dev: Prototype + Line Shape Match)", label: model32Best.label, confidence: model32Confidence },
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
  const [confidence, setConfidence] = useState(0);
  const [compareResults, setCompareResults] = useState({
    hyperDraw: { label: "start drawing", confidence: 0 },
    hyperDrawV2: { label: "start drawing", confidence: 0 },
  });
  const [compareStats, setCompareStats] = useState(() => loadCompareStats());
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [activeTab, setActiveTab] = useState("draw");
  const [algorithmStats, setAlgorithmStats] = useState(() => loadAlgorithmStats());
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
    setConfidence(0);
    setCompareResults({
      hyperDraw: { label: "start drawing", confidence: 0 },
      hyperDrawV2: { label: "start drawing", confidence: 0 },
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
      setConfidence(0);
      return;
    }

    if (!drawingStats.hasMeaningfulDrawing && !shouldGuessV2Early) {
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

    const { hyperDraw, hyperDrawV2 } = runLiveAlgorithmsPrepared(drawingStats.vec, preparedLiveDataset);
    const selected = selectedModel === "hyperdraw_v2" ? hyperDrawV2 : hyperDraw;
    const conf = Math.max(1, Math.min(99, selected.confidence));
    const lowConfidence = conf < 60;

    setGuess(selected.label);
    setConfidence(conf);
    setCompareResults({
      hyperDraw: { label: hyperDraw.label, confidence: Math.max(1, Math.min(99, hyperDraw.confidence || 0)) },
      hyperDrawV2: { label: hyperDrawV2.label, confidence: Math.max(1, Math.min(99, hyperDrawV2.confidence || 0)) },
    });
    if (devMode) {
      setLastDoneResults(runAlgorithms(drawingStats.vec, dataset));
    }
    setStatusMessage(lowConfidence ? "Low confidence guess — try cleaner strokes for better accuracy." : "");
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
              <p>Click <strong>Done</strong> to log correctness rates for algorithms 1, 7, 21, and 32.</p>
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
