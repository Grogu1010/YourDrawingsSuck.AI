const { useEffect, useMemo, useRef, useState } = React;

const OBJECTS = [
  "cat", "dog", "tree", "house", "car", "bicycle", "apple", "banana", "fish", "bird",
  "cloud", "sun", "moon", "star", "flower", "cup", "book", "chair", "table", "phone",
  "clock", "pizza", "burger", "ice cream", "guitar", "drum", "camera", "airplane", "rocket", "boat",
  "train", "bus", "robot", "monster", "dragon", "castle", "crown", "shoe", "hat", "glasses",
  "toothbrush", "key", "lock", "lamp", "cookie", "donut", "snail", "frog", "whale", "tennis ball",
  "mountains", "cube", "lightbulb", "shirt", "laptop", "nail", "bed", "alien", "cactus", "mushroom",
  "ghost", "envelope", "balloon", "candle", "arrow", "flames", "watermelon", "panda", "wizard staff", "astronaut helmet",
  "snowflake", "rainbow", "rock", "hourglass", "magnet", "stop sign", "dice", "headphones", "trophie", "palm trees"
];

const STORAGE_KEY = "yourdrawingssuckai.dataset.v1";
const ALGO_STATS_STORAGE_KEY = "yourdrawingssuckai.algorithmStats.v1";
const USER_PROFILE_STORAGE_KEY = "yourdrawingssuckai.userProfile.v1";
const SERVER_URL_STORAGE_KEY = "yourdrawingssuckai.serverUrl.v1";
const SERVER_SYNC_REV_STORAGE_KEY = "yourdrawingssuckai.serverSyncRevision.v1";
const DRAWING_CRYPTO_CONFIG_STORAGE_KEY = "yourdrawingssuckai.cryptoConfig.v1";
const DEV_TRAINING_MODE_STORAGE_KEY = "yourdrawingssuckai.devTrainingMode.v1";

const COMPARE_STATS_STORAGE_KEY = "yourdrawingssuckai.modelCompareStats.v1";
const GRID_SIZE = 16;
const MIN_POINT_DISTANCE_SQ = 9;

const DEV_TEST_SAMPLE_SIZE = 25;

const ACTIVE_ALGORITHM_IDS = [1, 7, 72, 77, 78, 79, 80];
const HYPERDRAW_ALGORITHM_ID = 1;
const HYPERDRAW_V2_ALGORITHM_ID = 7;

const V2X_ARTICLE_PARAGRAPHS = [
  "We are launching v2X as a meaningful upgrade over v2, focused on understanding what you meant to draw instead of matching only perfect pixel placement.",
  "Compared with v2, this release leans harder on scale-aware and rotation-aware matching, so guesses stay stable whether your sketch is tiny, huge, off-center, or turned in a different direction.",
  "It is also more forgiving when drawings are rougher, so quick doodles with shakier lines should still be interpreted correctly more often.",
  "In our internal checks, we expect this upgrade to answer correctly about 25% more often than v2 while also locking onto the right object faster during live drawing.",
  "Beyond raw accuracy, we are more confident in how quickly this model can learn new objects from incoming examples, which is why we are expanding the object set by 30 additions.",
  "New objects now include mountains, cube, lightbulb, shirt, laptop, nail, bed, alien, cactus, mushroom, ghost, envelope, balloon, candle, arrow, flames, watermelon, panda, wizard staff, astronaut helmet, snowflake, rainbow, rock, hourglass, magnet, stop sign, dice, headphones, trophie, and palm trees.",
  "We are also shipping quality-of-life upgrades: a skip object button, an undo button, and a revamped layout that fits everything on screen more clearly and aesthetically.",
  "Soon, everyone’s drawings will be connected to the shared server with end-to-end encryption so the server owner cannot view the raw drawings, preserving player privacy.",
  "This is still not perfect, and v3 will refresh the algorithm end-to-end with these goals built in from the ground up, but v2X is a strong step in the right direction today.",
];

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

const ARTICLE_ENTRIES = [
  { id: "v2x", title: "HyperDraw v2X Update", subtitle: "A short breakdown of what changed, and what comes next.", paragraphs: V2X_ARTICLE_PARAGRAPHS },
  { id: "v2", title: "HyperDraw v2 Deep Dive", subtitle: "The full original v2 research write-up.", paragraphs: V2_ARTICLE_PARAGRAPHS },
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

function randomPromptWeighted(promptCounts = {}) {
  const weighted = OBJECTS.map((label) => {
    const count = promptCounts[label] || 0;
    const scarcityWeight = 1 / (1 + count);
    const jitter = 0.88 + Math.random() * 0.24;
    return { label, weight: scarcityWeight * jitter };
  });

  const totalWeight = weighted.reduce((sum, item) => sum + item.weight, 0);
  if (totalWeight <= 0) return randomPrompt();

  let roll = Math.random() * totalWeight;
  for (const item of weighted) {
    roll -= item.weight;
    if (roll <= 0) return item.label;
  }
  return weighted[weighted.length - 1].label;
}

function chooseNextPrompt({ trainingMode = false, promptCounts = {} } = {}) {
  return trainingMode ? randomPromptWeighted(promptCounts) : randomPrompt();
}

function applyTrainingNoise(vector, intensity = 0.04) {
  if (!Array.isArray(vector)) return [];
  return vector.map((value) => Math.max(0, Math.min(1, value + (Math.random() * 2 - 1) * intensity)));
}

function loadDevTrainingMode() {
  return getStorageItem(DEV_TRAINING_MODE_STORAGE_KEY) === "1";
}

function saveDevTrainingMode(value) {
  setStorageItem(DEV_TRAINING_MODE_STORAGE_KEY, value ? "1" : "0");
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

function randomId() {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function bytesToBase64(bytes) {
  let binary = "";
  bytes.forEach((value) => {
    binary += String.fromCharCode(value);
  });
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(value);
  const output = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    output[i] = binary.charCodeAt(i);
  }
  return output;
}

function loadCryptoConfig() {
  try {
    const raw = getStorageItem(DRAWING_CRYPTO_CONFIG_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed.salt !== "string" || typeof parsed.keyHint !== "string") return null;
    return parsed;
  } catch {
    return null;
  }
}

function saveCryptoConfig(config) {
  setStorageItem(DRAWING_CRYPTO_CONFIG_STORAGE_KEY, JSON.stringify(config));
}

async function getOrCreateCryptoContext() {
  if (!window.crypto?.subtle) throw new Error("This browser cannot encrypt drawings.");

  let config = loadCryptoConfig();
  let passphrase = "";

  while (!passphrase.trim()) {
    const entered = window.prompt("Create a drawing passphrase. The server will store encrypted blobs only.") || "";
    passphrase = entered.trim();
  }

  if (!config) {
    const saltBytes = crypto.getRandomValues(new Uint8Array(16));
    config = {
      salt: bytesToBase64(saltBytes),
      keyHint: bytesToBase64(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(passphrase))).slice(0, 22),
    };
    saveCryptoConfig(config);
  }

  const imported = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(passphrase),
    { name: "PBKDF2" },
    false,
    ["deriveKey"]
  );

  const key = await crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: base64ToBytes(config.salt),
      iterations: 210000,
      hash: "SHA-256",
    },
    imported,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );

  return { key, keyHint: config.keyHint };
}

async function encryptDrawingEntry(entry, cryptoContext) {
  return encryptPayload(
    {
      id: entry.id,
      label: entry.label,
      vector: entry.vector,
      ts: entry.ts,
      authorName: entry.authorName,
      clientId: entry.clientId,
    },
    cryptoContext,
    entry.id
  );
}

async function encryptPayload(payload, cryptoContext, id = randomId()) {
  const ivBytes = crypto.getRandomValues(new Uint8Array(12));
  const plaintext = JSON.stringify(payload);

  const encrypted = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: ivBytes },
    cryptoContext.key,
    new TextEncoder().encode(plaintext)
  );

  return {
    id,
    iv: bytesToBase64(ivBytes),
    enc: bytesToBase64(new Uint8Array(encrypted)),
    ver: 1,
    keyHint: cryptoContext.keyHint,
  };
}

async function decryptDrawingEntry(encryptedEntry, cryptoContext) {
  try {
    const plaintextBytes = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: base64ToBytes(encryptedEntry.iv) },
      cryptoContext.key,
      base64ToBytes(encryptedEntry.enc)
    );
    const parsed = JSON.parse(new TextDecoder().decode(plaintextBytes));
    if (!parsed || typeof parsed.label !== "string" || !Array.isArray(parsed.vector) || typeof parsed.ts !== "number") return null;
    return {
      id: typeof parsed.id === "string" ? parsed.id : encryptedEntry.id,
      label: parsed.label,
      vector: parsed.vector,
      ts: parsed.ts,
      authorName: typeof parsed.authorName === "string" ? parsed.authorName : "anonymous",
      clientId: typeof parsed.clientId === "string" ? parsed.clientId : "",
    };
  } catch {
    return null;
  }
}

async function fetchPublicIpAddress() {
  try {
    const response = await fetch("https://api.ipify.org?format=json", { cache: "no-store" });
    if (!response.ok) throw new Error("ip fetch failed");
    const data = await response.json();
    if (typeof data?.ip !== "string" || !data.ip.trim()) throw new Error("missing ip");
    return data.ip.trim();
  } catch {
    return "unknown";
  }
}

function getServerBaseUrl() {
  const configured =
    (typeof window !== "undefined" && window.__YDS_SERVER_URL__) ||
    getStorageItem(SERVER_URL_STORAGE_KEY) ||
    "";
  return configured.trim().replace(/\/$/, "");
}

function loadUserProfile() {
  try {
    const raw = getStorageItem(USER_PROFILE_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.clientId === "string" && typeof parsed.name === "string" && parsed.name.trim()) {
        return parsed;
      }
    }
  } catch {
    // Use fallback profile creation.
  }

  let nextName = "";
  while (!nextName.trim()) {
    const entered = window.prompt("Please enter a name") || "";
    nextName = entered.trim();
  }

  const profile = { clientId: randomId(), name: nextName };
  setStorageItem(USER_PROFILE_STORAGE_KEY, JSON.stringify(profile));
  return profile;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) throw new Error(`Server error (${response.status})`);
  return response.json();
}

async function syncWithServer({ profile, drawings, cryptoContext, forceFullSync = false }) {
  const baseUrl = getServerBaseUrl();
  if (!baseUrl) return null;

  const revision = forceFullSync
    ? null
    : getStorageItem(SERVER_SYNC_REV_STORAGE_KEY);

  const encryptedDrawings = await Promise.all(drawings.map((item) => encryptDrawingEntry(item, cryptoContext)));
  const encryptedIp = await encryptPayload(
    { ip: await fetchPublicIpAddress(), clientId: profile.clientId, ts: Date.now() },
    cryptoContext,
    `${profile.clientId}_ip`
  );

  const data = await postJson(`${baseUrl}/api/sync`, {
    profile,
    encryptedIp,
    drawings: encryptedDrawings,
    revision,
  });

  if (data?.revision) setStorageItem(SERVER_SYNC_REV_STORAGE_KEY, data.revision);
  return data;
}


function loadCompareStats() {
  try {
    const raw = getStorageItem(COMPARE_STATS_STORAGE_KEY);
    if (!raw) return { attempts: 0, hyperDrawWins: 0, hyperDrawV2Wins: 0, hyperDrawV2XWins: 0, ties: 0 };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") throw new Error("invalid stats");
    return {
      attempts: Math.max(0, Math.floor(parsed.attempts || 0)),
      hyperDrawWins: Math.max(0, Math.floor(parsed.hyperDrawWins || 0)),
      hyperDrawV2Wins: Math.max(0, Math.floor(parsed.hyperDrawV2Wins || 0)),
      hyperDrawV2XWins: Math.max(0, Math.floor(parsed.hyperDrawV2XWins || 0)),
      ties: Math.max(0, Math.floor(parsed.ties || 0)),
    };
  } catch {
    return { attempts: 0, hyperDrawWins: 0, hyperDrawV2Wins: 0, hyperDrawV2XWins: 0, ties: 0 };
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

function extractScaleAwareLineLengthFeatures(vector, size = GRID_SIZE) {
  const normalized = normalizeVectorForSize(vector, size);
  const { edge, strokeWidth } = extractThicknessCompensatedFeatures(normalized, size);
  const rowLengths = new Array(size).fill(0);
  const colLengths = new Array(size).fill(0);
  let edgePixels = 0;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = edge[y * size + x] ? 1 : 0;
      edgePixels += value;
      rowLengths[y] += value;
      colLengths[x] += value;
    }
  }

  const safeEdgePixels = Math.max(1, edgePixels);
  const rowProfile = rowLengths.map((value) => value / safeEdgePixels);
  const colProfile = colLengths.map((value) => value / safeEdgePixels);
  const shapeFeatures = extractLineFeaturesForSize(normalized, size);

  return {
    normalized,
    profile: [...rowProfile, ...colProfile, ...shapeFeatures, Math.min(1, strokeWidth / 4)],
    density: edgePixels / (size * size),
  };
}

function extractAlgorithm77Architecture(vector, size = GRID_SIZE) {
  const normalized = normalizeVectorForSize(vector, size);
  const binary = normalized.map((value) => (value >= 0.22 ? 1 : 0));
  const occupied = [];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (!binary[y * size + x]) continue;
      occupied.push({ x, y });
    }
  }

  if (!occupied.length) {
    return {
      lines: [],
      relations: [],
      density: 0,
      symmetryX: 0,
      symmetryY: 0,
      lineCountNorm: 0,
    };
  }

  const xs = occupied.map((point) => point.x);
  const ys = occupied.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = Math.max(1, maxX - minX + 1);
  const height = Math.max(1, maxY - minY + 1);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const scale = Math.max(width, height, 1);
  const keySet = new Set(occupied.map((point) => `${point.x},${point.y}`));

  const normalizePoint = (x, y) => ({
    x: (x - centerX) / scale,
    y: (y - centerY) / scale,
  });

  const normalizedPoints = occupied.map((point) => normalizePoint(point.x, point.y));
  const tolerance = 1 / Math.max(2, scale * 0.9);
  const symmetryXHits = normalizedPoints.reduce((hits, point) => {
    const mirroredKey = `${Math.round(centerX - (point.x * scale))},${Math.round(centerY + (point.y * scale))}`;
    return hits + (keySet.has(mirroredKey) ? 1 : 0);
  }, 0);
  const symmetryYHits = normalizedPoints.reduce((hits, point) => {
    const mirroredKey = `${Math.round(centerX + (point.x * scale))},${Math.round(centerY - (point.y * scale))}`;
    return hits + (keySet.has(mirroredKey) ? 1 : 0);
  }, 0);

  const directions = [
    { dx: 1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: 1, dy: 1 },
    { dx: 1, dy: -1 },
  ];
  const segments = [];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (!keySet.has(`${x},${y}`)) continue;
      directions.forEach(({ dx, dy }) => {
        const prevX = x - dx;
        const prevY = y - dy;
        if (keySet.has(`${prevX},${prevY}`)) return;
        let run = 0;
        let curX = x;
        let curY = y;
        while (keySet.has(`${curX},${curY}`)) {
          run += 1;
          curX += dx;
          curY += dy;
        }
        if (run < 2) return;
        const endX = x + dx * (run - 1);
        const endY = y + dy * (run - 1);
        const nx1 = (x - centerX) / scale;
        const ny1 = (y - centerY) / scale;
        const nx2 = (endX - centerX) / scale;
        const ny2 = (endY - centerY) / scale;
        const lineLength = Math.hypot(nx2 - nx1, ny2 - ny1);
        const midpoint = { x: (nx1 + nx2) / 2, y: (ny1 + ny2) / 2 };
        const angle = Math.atan2(ny2 - ny1, nx2 - nx1);
        const orientation = ((angle % Math.PI) + Math.PI) % Math.PI;
        const straightness = Math.max(0.6, Math.min(1, run / (run + 1)));
        segments.push({ length: lineLength, orientation, midpoint, straightness });
      });
    }
  }

  const rankedSegments = segments
    .sort((a, b) => b.length - a.length)
    .slice(0, 24);

  const relations = [];
  for (let i = 0; i < rankedSegments.length; i += 1) {
    for (let j = i + 1; j < rankedSegments.length; j += 1) {
      const a = rankedSegments[i];
      const b = rankedSegments[j];
      relations.push({
        lengthRatio: Math.min(a.length, b.length) / Math.max(0.001, Math.max(a.length, b.length)),
        angleDelta: Math.min(
          Math.abs(a.orientation - b.orientation),
          Math.PI - Math.abs(a.orientation - b.orientation)
        ) / Math.PI,
        lineDistance: Math.min(2, Math.hypot(a.midpoint.x - b.midpoint.x, a.midpoint.y - b.midpoint.y)),
      });
    }
  }

  return {
    lines: rankedSegments,
    relations: relations
      .sort((a, b) => a.lineDistance - b.lineDistance)
      .slice(0, 60),
    density: occupied.length / (size * size),
    symmetryX: symmetryXHits / Math.max(1, occupied.length),
    symmetryY: symmetryYHits / Math.max(1, occupied.length),
    lineCountNorm: Math.min(1, rankedSegments.length / 24),
    tolerance,
  };
}

function algorithm77ArchitectureDistance(a, b) {
  const lineMatchCost = a.lines.reduce((sum, line) => {
    const best = b.lines.reduce((bestDistance, candidate) => {
      const lengthGap = Math.abs(line.length - candidate.length);
      const straightGap = Math.abs(line.straightness - candidate.straightness);
      return Math.min(bestDistance, lengthGap * 0.58 + straightGap * 0.42);
    }, 1.2);
    return sum + best;
  }, 0) / Math.max(1, a.lines.length);

  const relationMatchCost = a.relations.reduce((sum, relation) => {
    const best = b.relations.reduce((bestDistance, candidate) => {
      const ratioGap = Math.abs(relation.lengthRatio - candidate.lengthRatio);
      const angleGap = Math.abs(relation.angleDelta - candidate.angleDelta);
      const distanceGap = Math.abs(relation.lineDistance - candidate.lineDistance);
      return Math.min(bestDistance, ratioGap * 0.36 + angleGap * 0.34 + distanceGap * 0.3);
    }, 1.2);
    return sum + best;
  }, 0) / Math.max(1, a.relations.length);

  const symmetryGap =
    Math.abs(a.symmetryX - b.symmetryX) * 0.5 +
    Math.abs(a.symmetryY - b.symmetryY) * 0.5;
  const densityGap = Math.abs(a.density - b.density);
  const lineCountGap = Math.abs(a.lineCountNorm - b.lineCountNorm);

  return lineMatchCost * 0.46 + relationMatchCost * 0.39 + symmetryGap * 0.08 + densityGap * 0.04 + lineCountGap * 0.03;
}

function buildAlgorithm78RaySignature(vector, size = GRID_SIZE, angleBins = 36) {
  const normalized = normalizeVectorForSize(vector, size);
  const center = centroidForSize(normalized, size);
  const maxRadius = Math.sqrt(2) * (size / 2);
  const signature = new Array(angleBins).fill(0);

  for (let angleIndex = 0; angleIndex < angleBins; angleIndex += 1) {
    const angle = (angleIndex / angleBins) * 2 * Math.PI;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let firstHit = 1;
    let lastHit = 0;
    let accumulated = 0;
    let hits = 0;

    for (let step = 0; step <= size * 2; step += 1) {
      const radius = (step / (size * 2)) * maxRadius;
      const x = center.x + dx * radius;
      const y = center.y + dy * radius;
      const value = sampleBilinear(normalized, size, x, y);
      if (value <= 0.08) continue;
      const normalizedRadius = radius / Math.max(maxRadius, 1e-6);
      firstHit = Math.min(firstHit, normalizedRadius);
      lastHit = Math.max(lastHit, normalizedRadius);
      accumulated += value;
      hits += 1;
    }

    const occupancy = hits / Math.max(1, size * 2);
    signature[angleIndex] = (1 - firstHit) * 0.42 + lastHit * 0.36 + occupancy * 0.22 + accumulated * 0.04;
  }

  const norm = Math.sqrt(signature.reduce((sum, value) => sum + value * value, 0));
  return signature.map((value) => value / Math.max(norm, 1e-6));
}

function algorithm78SignatureDistance(a, b) {
  const count = Math.min(a.length, b.length);
  let best = Number.POSITIVE_INFINITY;

  for (let shift = 0; shift < count; shift += 1) {
    let directSum = 0;
    let flippedSum = 0;
    for (let i = 0; i < count; i += 1) {
      const shifted = b[(i + shift) % count];
      const flipped = b[(count - ((i + shift) % count)) % count];
      const directGap = a[i] - shifted;
      const flippedGap = a[i] - flipped;
      directSum += directGap * directGap;
      flippedSum += flippedGap * flippedGap;
    }
    best = Math.min(best, Math.sqrt(directSum / Math.max(1, count)), Math.sqrt(flippedSum / Math.max(1, count)));
  }

  return best;
}

function buildAlgorithm79PairGeometryDescriptor(vector, size = GRID_SIZE, bins = 18) {
  const normalized = normalizeVectorForSize(vector, size);
  const points = [];
  let minX = size;
  let minY = size;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = normalized[y * size + x] || 0;
      if (value <= 0.2) continue;
      points.push({ x, y, value });
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (!points.length) {
    return { distanceHist: new Array(bins).fill(0), areaHist: new Array(bins).fill(0), density: 0 };
  }

  const stride = Math.max(1, Math.floor(points.length / 96));
  const sampled = points.filter((_, index) => index % stride === 0).slice(0, 96);
  const width = Math.max(1, maxX - minX + 1);
  const height = Math.max(1, maxY - minY + 1);
  const scale = Math.max(width, height, 1);
  const distanceHist = new Array(bins).fill(0);
  const areaHist = new Array(bins).fill(0);

  for (let i = 0; i < sampled.length; i += 1) {
    const a = sampled[i];
    for (let j = i + 1; j < sampled.length; j += 1) {
      const b = sampled[j];
      const normalizedDistance = Math.min(0.9999, Math.hypot(a.x - b.x, a.y - b.y) / scale);
      distanceHist[Math.floor(normalizedDistance * bins)] += 1;

      const midX = (a.x + b.x) / 2;
      const midY = (a.y + b.y) / 2;
      const radial = Math.min(0.9999, Math.hypot(midX - (minX + maxX) / 2, midY - (minY + maxY) / 2) / scale);
      areaHist[Math.floor(radial * bins)] += 1;
    }
  }

  const normalizeHist = (hist) => {
    const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
    return hist.map((value) => value / Math.max(norm, 1e-6));
  };

  return {
    distanceHist: normalizeHist(distanceHist),
    areaHist: normalizeHist(areaHist),
    density: points.length / Math.max(1, size * size),
  };
}

function algorithm79DescriptorDistance(a, b) {
  let distanceSum = 0;
  for (let i = 0; i < a.distanceHist.length; i += 1) {
    const d = a.distanceHist[i] - b.distanceHist[i];
    distanceSum += d * d;
  }

  let areaSum = 0;
  for (let i = 0; i < a.areaHist.length; i += 1) {
    const d = a.areaHist[i] - b.areaHist[i];
    areaSum += d * d;
  }

  const densityGap = Math.abs(a.density - b.density);
  return Math.sqrt(distanceSum / Math.max(1, a.distanceHist.length)) * 0.62
    + Math.sqrt(areaSum / Math.max(1, a.areaHist.length)) * 0.3
    + densityGap * 0.08;
}

function thinBinaryMap(binary, size = GRID_SIZE, maxPasses = 12) {
  const output = [...binary];
  const neighborOffsets = [
    [0, -1], [1, -1], [1, 0], [1, 1],
    [0, 1], [-1, 1], [-1, 0], [-1, -1],
  ];
  const get = (x, y) => (x < 0 || y < 0 || x >= size || y >= size ? 0 : output[y * size + x]);

  for (let pass = 0; pass < maxPasses; pass += 1) {
    let changed = false;
    [0, 1].forEach((phase) => {
      const toRemove = [];
      for (let y = 1; y < size - 1; y += 1) {
        for (let x = 1; x < size - 1; x += 1) {
          if (get(x, y) === 0) continue;
          const neighbors = neighborOffsets.map(([dx, dy]) => get(x + dx, y + dy));
          const count = neighbors.reduce((sum, value) => sum + value, 0);
          if (count < 2 || count > 6) continue;

          let transitions = 0;
          for (let i = 0; i < neighbors.length; i += 1) {
            const cur = neighbors[i];
            const next = neighbors[(i + 1) % neighbors.length];
            if (cur === 0 && next === 1) transitions += 1;
          }
          if (transitions !== 1) continue;

          const p2 = neighbors[0];
          const p4 = neighbors[2];
          const p6 = neighbors[4];
          const p8 = neighbors[6];
          const phaseCondition = phase === 0
            ? (p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0)
            : (p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0);
          if (!phaseCondition) continue;
          toRemove.push(y * size + x);
        }
      }

      if (toRemove.length) {
        changed = true;
        toRemove.forEach((index) => {
          output[index] = 0;
        });
      }
    });

    if (!changed) break;
  }

  return output;
}

function buildAlgorithm80TopologyDescriptor(vector, size = GRID_SIZE) {
  const normalized = normalizeVectorForSize(vector, size);
  const binary = normalized.map((value) => (value > 0.2 ? 1 : 0));
  const thin = thinBinaryMap(binary, size);
  const points = [];
  const endpoints = [];
  const junctions = [];
  const branchHist = new Array(5).fill(0);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (thin[y * size + x] === 0) continue;
      points.push({ x, y });
      let neighbors = 0;
      for (let ny = -1; ny <= 1; ny += 1) {
        for (let nx = -1; nx <= 1; nx += 1) {
          if (nx === 0 && ny === 0) continue;
          const tx = x + nx;
          const ty = y + ny;
          if (tx < 0 || ty < 0 || tx >= size || ty >= size) continue;
          neighbors += thin[ty * size + tx];
        }
      }
      branchHist[Math.min(branchHist.length - 1, neighbors)] += 1;
      if (neighbors === 1) endpoints.push({ x, y });
      if (neighbors >= 3) junctions.push({ x, y });
    }
  }

  const keypoints = [...endpoints, ...junctions].slice(0, 20);
  const pairHist = new Array(12).fill(0);
  const scale = Math.max(1, size - 1);
  for (let i = 0; i < keypoints.length; i += 1) {
    for (let j = i + 1; j < keypoints.length; j += 1) {
      const d = Math.min(0.9999, Math.hypot(keypoints[i].x - keypoints[j].x, keypoints[i].y - keypoints[j].y) / scale);
      pairHist[Math.floor(d * pairHist.length)] += 1;
    }
  }

  const normalizeHist = (hist) => {
    const norm = Math.sqrt(hist.reduce((sum, value) => sum + value * value, 0));
    return hist.map((value) => value / Math.max(norm, 1e-6));
  };

  return {
    branchHist: normalizeHist(branchHist),
    pairHist: normalizeHist(pairHist),
    endpointRatio: endpoints.length / Math.max(1, points.length),
    junctionRatio: junctions.length / Math.max(1, points.length),
    strokeDensity: points.length / Math.max(1, size * size),
  };
}

function algorithm80TopologyDistance(a, b) {
  const histDistance = (histA, histB) => {
    let sum = 0;
    for (let i = 0; i < histA.length; i += 1) {
      const d = histA[i] - histB[i];
      sum += d * d;
    }
    return Math.sqrt(sum / Math.max(1, histA.length));
  };

  return histDistance(a.branchHist, b.branchHist) * 0.44
    + histDistance(a.pairHist, b.pairHist) * 0.38
    + Math.abs(a.endpointRatio - b.endpointRatio) * 0.1
    + Math.abs(a.junctionRatio - b.junctionRatio) * 0.06
    + Math.abs(a.strokeDensity - b.strokeDensity) * 0.02;
}

function minTransformDistanceForSize(a, b, size = GRID_SIZE) {
  const aVariants = generateTransformVariantsForSize(a, size);
  const bVariants = generateTransformVariantsForSize(b, size);
  let best = Number.POSITIVE_INFINITY;

  aVariants.forEach((aVariant) => {
    bVariants.forEach((bVariant) => {
      best = Math.min(best, distance(aVariant, bVariant) / Math.sqrt(size * size));
    });
  });

  return best;
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
      hyperDrawV2X: { label: "Need training data first", confidence: 0 },
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
      hyperDrawV2X: { label: "Need training data first", confidence: 0 },
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

  const hyperDrawV2X = runAlgorithms(vector, normalizedDataset.map((item) => ({ label: item.label, vector: item.normalizedVector }))).find((item) => item.id === 72) || { label: "unknown", confidence: 0 };

  return { hyperDraw, hyperDrawV2, hyperDrawV2X };
}

function runAlgorithms(vector, dataset) {
  if (!dataset.length) {
    return [
      { id: 1, name: "Algorithm 1 (Current)", label: "Need training data first", confidence: 0 },
      { id: 7, name: "Algorithm 7 (Prototype Normalized)", label: "Need training data first", confidence: 0 },
      { id: 72, name: "Algorithm 72 (v2X transform-aware)", label: "Need training data first", confidence: 0 },
      { id: 77, name: "Algorithm 77 (line-architecture matcher)", label: "Need training data first", confidence: 0 },
      { id: 78, name: "Algorithm 78 (polar ray signature)", label: "Need training data first", confidence: 0 },
      { id: 79, name: "Algorithm 79 (pair-geometry spectrum)", label: "Need training data first", confidence: 0 },
      { id: 80, name: "Algorithm 80 (skeleton topology graph)", label: "Need training data first", confidence: 0 },
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

  const algorithm77Input = extractAlgorithm77Architecture(normalizedInput, GRID_SIZE);
  const algorithm77PrototypeByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = extractAlgorithm77Architecture(prototype, GRID_SIZE);
    return acc;
  }, {});
  const algorithm78Input = buildAlgorithm78RaySignature(normalizedInput, GRID_SIZE);
  const algorithm78PrototypeByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = buildAlgorithm78RaySignature(prototype, GRID_SIZE);
    return acc;
  }, {});
  const algorithm79Input = buildAlgorithm79PairGeometryDescriptor(normalizedInput, GRID_SIZE);
  const algorithm79PrototypeByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = buildAlgorithm79PairGeometryDescriptor(prototype, GRID_SIZE);
    return acc;
  }, {});
  const algorithm80Input = buildAlgorithm80TopologyDescriptor(normalizedInput, GRID_SIZE);
  const algorithm80PrototypeByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = buildAlgorithm80TopologyDescriptor(prototype, GRID_SIZE);
    return acc;
  }, {});

  const invariantInput = extractScaleAwareLineLengthFeatures(normalizedInput, GRID_SIZE);
  const invariantPrototypeByLabel = Object.entries(prototypesNormalized).reduce((acc, [label, prototype]) => {
    acc[label] = extractScaleAwareLineLengthFeatures(prototype, GRID_SIZE);
    return acc;
  }, {});
  const invariantDistances = dataset
    .map((item) => {
      const sample = extractScaleAwareLineLengthFeatures(normalizeVector(item.vector), GRID_SIZE);
      const transformDistance = minTransformDistanceForSize(invariantInput.normalized, sample.normalized, GRID_SIZE);
      const lineLengthDistance = featureDistance(invariantInput.profile, sample.profile);
      const densityGap = Math.abs(invariantInput.density - sample.density);
      return {
        label: item.label,
        distance: transformDistance + lineLengthDistance * 0.12 + densityGap * 0.16,
      };
    })
    .sort((a, b) => a.distance - b.distance);

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

  const scoreAlgo45TransformVariant = ({
    neighborDepth = 6,
    lineBlend = 0.12,
    densityWeight = 0.04,
    balancePenalty = 0,
    temperature = 2.3,
    centerWeight = 0,
  }) => {
    const topNeighbors = invariantDistances.slice(0, Math.min(neighborDepth, invariantDistances.length));
    const ranked = Object.entries(invariantPrototypeByLabel)
      .map(([label, prototype]) => {
        const transformDistance = minTransformDistanceForSize(invariantInput.normalized, prototype.normalized, GRID_SIZE);
        const lineDistance = featureDistance(invariantInput.profile, prototype.profile);
        const densityGap = Math.abs(invariantInput.density - prototype.density);

        const centerGap =
          Math.abs((invariantInput.profile[2 * GRID_SIZE + 5] || 0.5) - (prototype.profile[2 * GRID_SIZE + 5] || 0.5)) +
          Math.abs((invariantInput.profile[2 * GRID_SIZE + 6] || 0.5) - (prototype.profile[2 * GRID_SIZE + 6] || 0.5));

        const blendedDistance =
          transformDistance * (1 - lineBlend) +
          lineDistance * lineBlend +
          densityGap * densityWeight +
          centerGap * centerWeight;

        const baseScore = 1 / Math.max(0.001, blendedDistance + 0.04);
        const neighborScore = topNeighbors.reduce((bonus, neighbor, index) => {
          if (neighbor.label !== label) return bonus;
          return bonus + 0.06 / (index + 1);
        }, 0);
        const priorPenalty = ((datasetLabelCounts[label] || 0) / datasetSize) * balancePenalty;

        return { label, score: baseScore + neighborScore - priorPenalty };
      })
      .sort((a, b) => b.score - a.score);

    const probabilities = softmax(ranked.map((entry) => entry.score * temperature));
    return {
      label: ranked[0]?.label || "unknown",
      confidence: Math.max(1, Math.min(99, Math.round((probabilities[0] || 0) * 100))),
    };
  };

  const algorithm67 = scoreAlgo45TransformVariant({ neighborDepth: 6, lineBlend: 0.1, densityWeight: 0.04, temperature: 2.35 });
  const algorithm68 = scoreAlgo45TransformVariant({ neighborDepth: 6, lineBlend: 0.16, densityWeight: 0.05, temperature: 2.3 });
  const algorithm69 = scoreAlgo45TransformVariant({ neighborDepth: 7, lineBlend: 0.14, densityWeight: 0.04, balancePenalty: 0.08, temperature: 2.4 });
  const algorithm70 = scoreAlgo45TransformVariant({ neighborDepth: 5, lineBlend: 0.11, densityWeight: 0.08, temperature: 2.3 });
  const algorithm71 = scoreAlgo45TransformVariant({ neighborDepth: 9, lineBlend: 0.13, densityWeight: 0.04, temperature: 2.2 });
  const algorithm72 = scoreAlgo45TransformVariant({ neighborDepth: 6, lineBlend: 0.13, densityWeight: 0.04, centerWeight: 0.05, temperature: 2.35 });
  const algorithm73 = scoreAlgo45TransformVariant({ neighborDepth: 6, lineBlend: 0.12, densityWeight: 0.04, centerWeight: 0.03, temperature: 2.55 });
  const algorithm74 = scoreAlgo45TransformVariant({ neighborDepth: 6, lineBlend: 0.22, densityWeight: 0.03, temperature: 2.3 });
  const algorithm75 = scoreAlgo45TransformVariant({ neighborDepth: 7, lineBlend: 0.16, densityWeight: 0.04, balancePenalty: 0.14, temperature: 2.35 });
  const algorithm76 = scoreAlgo45TransformVariant({ neighborDepth: 8, lineBlend: 0.18, densityWeight: 0.05, balancePenalty: 0.08, centerWeight: 0.03, temperature: 2.3 });
  const algorithm77Ranked = Object.entries(algorithm77PrototypeByLabel)
    .map(([label, architecture]) => {
      const architectureDistance = algorithm77ArchitectureDistance(algorithm77Input, architecture);
      const neighborBoost = normalizedDistances
        .slice(0, Math.min(8, normalizedDistances.length))
        .reduce((bonus, neighbor, index) => bonus + (neighbor.label === label ? 0.07 / (index + 1) : 0), 0);
      const score = 1 / Math.max(0.001, architectureDistance + 0.045) + neighborBoost;
      return { label, score };
    })
    .sort((a, b) => b.score - a.score);
  const algorithm77Probs = softmax(algorithm77Ranked.map((entry) => entry.score * 2.25));
  const algorithm77 = {
    label: algorithm77Ranked[0]?.label || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((algorithm77Probs[0] || 0) * 100))),
  };
  const algorithm78Ranked = Object.entries(algorithm78PrototypeByLabel)
    .map(([label, signature]) => {
      const signatureDistance = algorithm78SignatureDistance(algorithm78Input, signature);
      const score = 1 / Math.max(0.001, signatureDistance + 0.03);
      return { label, score };
    })
    .sort((a, b) => b.score - a.score);
  const algorithm78Probs = softmax(algorithm78Ranked.map((entry) => entry.score * 2.2));
  const algorithm78 = {
    label: algorithm78Ranked[0]?.label || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((algorithm78Probs[0] || 0) * 100))),
  };

  const algorithm79Ranked = Object.entries(algorithm79PrototypeByLabel)
    .map(([label, descriptor]) => {
      const descriptorDistance = algorithm79DescriptorDistance(algorithm79Input, descriptor);
      const score = 1 / Math.max(0.001, descriptorDistance + 0.035);
      return { label, score };
    })
    .sort((a, b) => b.score - a.score);
  const algorithm79Probs = softmax(algorithm79Ranked.map((entry) => entry.score * 2.15));
  const algorithm79 = {
    label: algorithm79Ranked[0]?.label || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((algorithm79Probs[0] || 0) * 100))),
  };

  const algorithm80Ranked = Object.entries(algorithm80PrototypeByLabel)
    .map(([label, descriptor]) => {
      const topologyDistance = algorithm80TopologyDistance(algorithm80Input, descriptor);
      const score = 1 / Math.max(0.001, topologyDistance + 0.04);
      return { label, score };
    })
    .sort((a, b) => b.score - a.score);
  const algorithm80Probs = softmax(algorithm80Ranked.map((entry) => entry.score * 2.1));
  const algorithm80 = {
    label: algorithm80Ranked[0]?.label || "unknown",
    confidence: Math.max(1, Math.min(99, Math.round((algorithm80Probs[0] || 0) * 100))),
  };

  return [
    { id: 1, name: "Algorithm 1 (v1 current)", label: algo1Guess, confidence: algo1Confidence },
    { id: 7, name: "Algorithm 7 (v2 normalized)", label: prototypeNorm?.label || "unknown", confidence: Math.round((1 - Math.min(1, prototypeNorm?.distance || 1)) * 100) },
    { id: 72, name: "Algorithm 72 (v2X transform-aware)", label: algorithm72.label, confidence: algorithm72.confidence },
    { id: 77, name: "Algorithm 77 (line-architecture matcher)", label: algorithm77.label, confidence: algorithm77.confidence },
    { id: 78, name: "Algorithm 78 (polar ray signature)", label: algorithm78.label, confidence: algorithm78.confidence },
    { id: 79, name: "Algorithm 79 (pair-geometry spectrum)", label: algorithm79.label, confidence: algorithm79.confidence },
    { id: 80, name: "Algorithm 80 (skeleton topology graph)", label: algorithm80.label, confidence: algorithm80.confidence },
  ];
}


function createEmptyPerformanceTelemetry() {
  return {
    runs: 0,
    averageComputeMs: 0,
    averageQueueDelayMs: 0,
    averageLagMs: 0,
    maxComputeMs: 0,
    maxQueueDelayMs: 0,
    slowFrameRatio: 0,
    performanceScore: 100,
    issues: ["No telemetry yet. Start drawing to collect metrics."],
  };
}

function derivePerformanceTelemetry(metrics) {
  if (!metrics.samples) return createEmptyPerformanceTelemetry();

  const averageComputeMs = metrics.computeMs / metrics.samples;
  const averageQueueDelayMs = metrics.queueDelayMs / metrics.samples;
  const averageLagMs = metrics.lagMs / metrics.samples;
  const slowFrameRatio = metrics.slowFrames / metrics.samples;

  const computePenalty = Math.max(0, (averageComputeMs - 8) / 20) * 38;
  const queuePenalty = Math.max(0, (averageQueueDelayMs - 14) / 70) * 26;
  const lagPenalty = Math.max(0, (averageLagMs - 3) / 14) * 24;
  const slowFramePenalty = Math.min(1, slowFrameRatio * 1.2) * 12;

  const rawScore = 100 - computePenalty - queuePenalty - lagPenalty - slowFramePenalty;
  const score = Math.round(Math.max(0, Math.min(100, rawScore)));

  const issues = [];
  if (averageComputeMs > 16) issues.push("Inference compute time is high. Consider reducing dataset size or algorithm complexity.");
  if (averageQueueDelayMs > 48) issues.push("Guess queue delay is high. Inputs are arriving faster than inference completes.");
  if (averageLagMs > 10) issues.push("Frame lag is noticeable. Rendering + prediction work is heavy while drawing.");
  if (slowFrameRatio > 0.25) issues.push("Over a quarter of checks are slow frames. Performance is inconsistent under live interaction.");
  if (!issues.length) issues.push("No major issues detected right now.");

  return {
    runs: metrics.samples,
    averageComputeMs: Number(averageComputeMs.toFixed(1)),
    averageQueueDelayMs: Number(averageQueueDelayMs.toFixed(1)),
    averageLagMs: Number(averageLagMs.toFixed(1)),
    maxComputeMs: Number((metrics.maxComputeMs || 0).toFixed(1)),
    maxQueueDelayMs: Number((metrics.maxQueueDelayMs || 0).toFixed(1)),
    slowFrameRatio: Number((slowFrameRatio * 100).toFixed(1)),
    performanceScore: score,
    issues,
  };
}

function pickRandomItems(items, count) {
  if (!Array.isArray(items) || count <= 0) return [];
  if (count >= items.length) return [...items];

  const pool = [...items];
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const randomIndex = Math.floor(Math.random() * (i + 1));
    const temp = pool[i];
    pool[i] = pool[randomIndex];
    pool[randomIndex] = temp;
  }
  return pool.slice(0, count);
}


function App() {
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const canvasRectRef = useRef(null);
  const canvasContextRef = useRef(null);
  const isDrawingRef = useRef(false);
  const strokesRef = useRef([]);
  const activeStrokeRef = useRef(null);
  const drawingRevisionRef = useRef(0);
  const lastGuessedRevisionRef = useRef(-1);
  const guessTimeoutRef = useRef(null);
  const profileRef = useRef(loadUserProfile());
  const cryptoContextRef = useRef(null);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => chooseNextPrompt({ trainingMode: loadDevTrainingMode(), promptCounts: {} }));
  const [selectedModel, setSelectedModel] = useState("hyperdraw_v2x");
  const [compareMode, setCompareMode] = useState(false);
  const [guess, setGuess] = useState("start drawing");
  const [compareResults, setCompareResults] = useState({
    hyperDraw: { label: "start drawing" },
    hyperDrawV2: { label: "start drawing" },
    hyperDrawV2X: { label: "start drawing" },
  });
  const [compareStats, setCompareStats] = useState(() => loadCompareStats());
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [trainingMode, setTrainingMode] = useState(() => loadDevTrainingMode());
  const [activeTab, setActiveTab] = useState("draw");
  const [expandedArticleId, setExpandedArticleId] = useState(null);
  const [algorithmStats, setAlgorithmStats] = useState(() => loadAlgorithmStats());
  const [sessionAlgorithmStats, setSessionAlgorithmStats] = useState(() => createDefaultAlgorithmStats());
  const [trainingSessionAlgorithmStats, setTrainingSessionAlgorithmStats] = useState(() => createDefaultAlgorithmStats());
  const [trainingLifetimeAlgorithmStats, setTrainingLifetimeAlgorithmStats] = useState(() => createDefaultAlgorithmStats());
  const [devStatsView, setDevStatsView] = useState("session");
  const [lastDoneResults, setLastDoneResults] = useState([]);
  const [onlinePlayers, setOnlinePlayers] = useState([]);
  const performanceMetricsRef = useRef({
    samples: 0,
    computeMs: 0,
    queueDelayMs: 0,
    lagMs: 0,
    slowFrames: 0,
    maxComputeMs: 0,
    maxQueueDelayMs: 0,
  });
  const [devPerformance, setDevPerformance] = useState(() => createEmptyPerformanceTelemetry());
  const [devTestRunning, setDevTestRunning] = useState(false);
  const [devTestProgress, setDevTestProgress] = useState({ processed: 0, total: 0 });
  const [devTestReport, setDevTestReport] = useState(null);
  const [devTestSampleSize, setDevTestSampleSize] = useState(25);
  const [devTestPopupOpen, setDevTestPopupOpen] = useState(false);
  const [devTestElapsedMs, setDevTestElapsedMs] = useState(0);
  const [devTestLiveSnapshot, setDevTestLiveSnapshot] = useState({
    currentLabel: "-",
    currentGuesses: [],
  });
  const devTestStartedAtRef = useRef(0);
  const devTestStopRequestedRef = useRef(false);
  const preparedLiveDataset = useMemo(() => prepareLiveDataset(dataset), [dataset]);

  useEffect(() => {
    const profile = profileRef.current;

    getOrCreateCryptoContext()
      .then((cryptoContext) => {
        cryptoContextRef.current = cryptoContext;
        const currentDatasetWithProfile = dataset.map((item) => ({ ...item, authorName: profile.name }));

        return syncWithServer({
          profile,
          drawings: currentDatasetWithProfile,
          cryptoContext,
          forceFullSync: true,
        });
      })
      .then((result) => {
        if (!result || !Array.isArray(result.drawings)) return;

        setOnlinePlayers(Array.isArray(result.online) ? result.online : []);

        return Promise.all(
          result.drawings
            .filter((item) => item && typeof item.enc === "string" && typeof item.iv === "string")
            .map((item) => decryptDrawingEntry(item, cryptoContextRef.current))
        ).then((decrypted) => {
          const merged = decrypted.filter(Boolean).slice(-2000);
          setDataset(merged);
          saveDataset(merged);
        });
      })
      .catch(() => {
        // Silent fallback: app keeps working fully offline/local-only.
      });

    window.changeDrawingPlayerName = async (nextNameRaw) => {
      const nextName = `${nextNameRaw || ""}`.trim();
      if (!nextName) return false;

      profileRef.current = { ...profileRef.current, name: nextName };
      setStorageItem(USER_PROFILE_STORAGE_KEY, JSON.stringify(profileRef.current));

      const renamedLocal = loadDataset().map((item) => ({ ...item, authorName: nextName }));
      saveDataset(renamedLocal);
      setDataset(renamedLocal);

      try {
        if (!cryptoContextRef.current) return true;
        await syncWithServer({
          profile: profileRef.current,
          drawings: renamedLocal,
          cryptoContext: cryptoContextRef.current,
          forceFullSync: true,
        });
      } catch {
        // Local rename still succeeds even if server is currently unavailable.
      }
      return true;
    };

    return () => {
      delete window.changeDrawingPlayerName;
    };
  }, []);

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

    canvasContextRef.current = ctx;

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
    const rect = canvasRectRef.current || canvas.getBoundingClientRect();

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
    const canvas = canvasRef.current;
    const ctx = canvasContextRef.current || canvas.getContext("2d");
    canvasRectRef.current = canvas.getBoundingClientRect();
    const point = getPoint(event);
    isDrawingRef.current = true;
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(2, ctx.lineWidth / 2), 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    activeStrokeRef.current = { points: [point], erase: isErasing };
    strokesRef.current.push(activeStrokeRef.current);
    drawingRevisionRef.current += 1;
  };

  const draw = (event) => {
    if (!isDrawingRef.current) return;
    event.preventDefault();
    const ctx = canvasContextRef.current || canvasRef.current.getContext("2d");
    const point = getPoint(event);
    const activeStroke = activeStrokeRef.current;
    const points = activeStroke?.points;
    const previousPoint = points?.[points.length - 1];

    if (previousPoint) {
      const dx = point.x - previousPoint.x;
      const dy = point.y - previousPoint.y;
      if (dx * dx + dy * dy < MIN_POINT_DISTANCE_SQ) return;
    }

    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    points?.push(point);
    drawingRevisionRef.current += 1;
  };

  const stopDrawing = () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    (canvasContextRef.current || canvasRef.current.getContext("2d")).closePath();
    activeStrokeRef.current = null;
    canvasRectRef.current = null;
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
      hyperDrawV2X: { label: "start drawing" },
    });
    setStatusMessage("");
    if (guessTimeoutRef.current) {
      clearTimeout(guessTimeoutRef.current);
      guessTimeoutRef.current = null;
    }
  };

  const redrawStrokes = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    strokesRef.current.forEach((stroke) => {
      if (!stroke || !Array.isArray(stroke.points) || stroke.points.length < 1) return;
      ctx.strokeStyle = stroke.erase ? "#ffffff" : "#111827";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = stroke.erase ? 32 : 20;

      if (stroke.points.length === 1) {
        const point = stroke.points[0];
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(2, ctx.lineWidth / 2), 0, Math.PI * 2);
        ctx.fill();
        ctx.closePath();
        return;
      }

      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      stroke.points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.stroke();
      ctx.closePath();
    });

    ctx.strokeStyle = isErasing ? "#ffffff" : "#111827";
    ctx.lineWidth = isErasing ? 32 : 20;
  };

  const undoLastStroke = () => {
    if (!strokesRef.current.length) {
      setStatusMessage("Nothing to undo yet.");
      return;
    }

    strokesRef.current.pop();
    activeStrokeRef.current = null;
    drawingRevisionRef.current += 1;
    redrawStrokes();
    scheduleGuess(true);
    setStatusMessage("Undid last stroke.");
  };

  const skipObject = () => {
    const currentCounts = dataset.reduce((acc, item) => {
      acc[item.label] = (acc[item.label] || 0) + 1;
      return acc;
    }, {});
    setPrompt(chooseNextPrompt({ trainingMode, promptCounts: currentCounts }));
    clearCanvas();
    setStatusMessage(trainingMode ? "Skipped. Training Mode prioritized a less-trained object." : "Skipped. New object loaded.");
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
    const drawnStrokeCount = strokesRef.current.filter((stroke) => stroke?.points?.length > 1).length;

    return {
      vec,
      totalInk,
      activePixels,
      drawnStrokeCount,
      hasAnyInk: totalInk > 0.03 || activePixels > 0,
      hasMeaningfulDrawing: totalInk > 5 && activePixels > 8 && drawnStrokeCount > 0,
    };
  };

  const guessDrawing = ({ scheduledAt = null } = {}) => {
    if (!canvasRef.current) return;

    const drawingStats = getDrawingStats();

    const shouldGuessV2Early = selectedModel !== "hyperdraw" || compareMode || devMode;

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

    const guessStart = performance.now();
    const { hyperDraw, hyperDrawV2, hyperDrawV2X } = runLiveAlgorithmsPrepared(drawingStats.vec, preparedLiveDataset);
    const selected = selectedModel === "hyperdraw" ? hyperDraw : (selectedModel === "hyperdraw_v2" ? hyperDrawV2 : hyperDrawV2X);

    setGuess(selected.label);
    setCompareResults({
      hyperDraw: { label: hyperDraw.label },
      hyperDrawV2: { label: hyperDrawV2.label },
      hyperDrawV2X: { label: hyperDrawV2X.label },
    });
    if (devMode) {
      setLastDoneResults(runAlgorithms(drawingStats.vec, dataset));

      const computeMs = performance.now() - guessStart;
      const queueDelayMs = scheduledAt ? Math.max(0, guessStart - scheduledAt) : 0;
      const lagMs = Math.max(0, computeMs - 16.7);
      const metrics = performanceMetricsRef.current;
      metrics.samples += 1;
      metrics.computeMs += computeMs;
      metrics.queueDelayMs += queueDelayMs;
      metrics.lagMs += lagMs;
      metrics.maxComputeMs = Math.max(metrics.maxComputeMs, computeMs);
      metrics.maxQueueDelayMs = Math.max(metrics.maxQueueDelayMs, queueDelayMs);
      if (computeMs > 24 || queueDelayMs > 42) metrics.slowFrames += 1;
      setDevPerformance(derivePerformanceTelemetry(metrics));
    }
    setStatusMessage("");
  };

  const scheduleGuess = (immediate = false) => {
    if (drawingRevisionRef.current === lastGuessedRevisionRef.current && !immediate) return;

    if (!immediate && guessTimeoutRef.current) return;

    if (guessTimeoutRef.current) {
      clearTimeout(guessTimeoutRef.current);
      guessTimeoutRef.current = null;
    }

    const delay = immediate ? 0 : 180;
    const scheduledAt = performance.now() + delay;
    guessTimeoutRef.current = setTimeout(() => {
      if (drawingRevisionRef.current === lastGuessedRevisionRef.current && !immediate) return;
      lastGuessedRevisionRef.current = drawingRevisionRef.current;
      guessDrawing({ scheduledAt });
      guessTimeoutRef.current = null;
    }, delay);
  };

  const stopDrawingAndGuess = () => {
    stopDrawing();
    scheduleGuess(true);
  };


  const runDevModelSweep = async ({ sampleSize = null } = {}) => {
    if (devTestRunning) return;

    if (dataset.length < 2) {
      setStatusMessage("Need at least 2 drawings before running dev tests.");
      return;
    }

    const indexedDataset = dataset.map((item, index) => ({ ...item, sourceIndex: index }));
    const selectedDrawings = sampleSize ? pickRandomItems(indexedDataset, sampleSize) : indexedDataset;

    if (!selectedDrawings.length) {
      setStatusMessage("No drawings selected for dev test run.");
      return;
    }

    const byAlgorithm = ACTIVE_ALGORITHM_IDS.reduce((acc, id) => {
      acc[id] = { id, attempts: 0, correct: 0 };
      return acc;
    }, {});

    devTestStopRequestedRef.current = false;
    devTestStartedAtRef.current = Date.now();
    setDevTestElapsedMs(0);
    setDevTestPopupOpen(true);
    setDevTestRunning(true);
    setDevTestProgress({ processed: 0, total: selectedDrawings.length });
    setDevTestLiveSnapshot({ currentLabel: "-", currentGuesses: [] });
    setStatusMessage(`Running dev test over ${selectedDrawings.length} drawing${selectedDrawings.length === 1 ? "" : "s"}...`);

    for (let i = 0; i < selectedDrawings.length; i += 1) {
      if (devTestStopRequestedRef.current) break;

    setDevTestRunning(true);
    setDevTestProgress({ processed: 0, total: selectedDrawings.length });
    setStatusMessage(`Running dev test over ${selectedDrawings.length} drawing${selectedDrawings.length === 1 ? "" : "s"}...`);

    for (let i = 0; i < selectedDrawings.length; i += 1) {
      const drawing = selectedDrawings[i];
      const trainingDataset = indexedDataset.filter((item) => item.sourceIndex !== drawing.sourceIndex);
      if (!trainingDataset.length) continue;

      const results = runAlgorithms(drawing.vector, trainingDataset);
      setDevTestLiveSnapshot({
        currentLabel: drawing.label,
        currentGuesses: results.map((result) => ({
          id: result.id,
          label: result.label,
          confidence: result.confidence,
        })),
      });
      ACTIVE_ALGORITHM_IDS.forEach((algorithmId) => {
        const entry = byAlgorithm[algorithmId];
        const result = results.find((candidate) => candidate.id === algorithmId);
        entry.attempts += 1;
        if (result?.label === drawing.label) entry.correct += 1;
      });

      const rollingSummary = ACTIVE_ALGORITHM_IDS.map((algorithmId) => {
        const entry = byAlgorithm[algorithmId];
        const winRate = entry.attempts ? Number(((entry.correct / entry.attempts) * 100).toFixed(1)) : 0;
        return { ...entry, winRate };
      }).sort((a, b) => {
        if (b.winRate !== a.winRate) return b.winRate - a.winRate;
        if (b.correct !== a.correct) return b.correct - a.correct;
        return a.id - b.id;
      });

      setDevTestReport({
        generatedAt: Date.now(),
        sampleLabel: sampleSize ? `Dev Test ${sampleSize}` : "Dev Test All",
        totalDrawings: selectedDrawings.length,
        summary: rollingSummary,
      });

      if ((i + 1) % 5 === 0 || i === selectedDrawings.length - 1) {
        setDevTestProgress({ processed: i + 1, total: selectedDrawings.length });
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    const summary = ACTIVE_ALGORITHM_IDS.map((algorithmId) => {
      const entry = byAlgorithm[algorithmId];
      const winRate = entry.attempts ? Number(((entry.correct / entry.attempts) * 100).toFixed(1)) : 0;
      return { ...entry, winRate };
    }).sort((a, b) => {
      if (b.winRate !== a.winRate) return b.winRate - a.winRate;
      if (b.correct !== a.correct) return b.correct - a.correct;
      return a.id - b.id;
    });

    setDevTestReport({
      generatedAt: Date.now(),
      sampleLabel: sampleSize ? `Dev Test ${sampleSize}` : "Dev Test All",
      totalDrawings: selectedDrawings.length,
      summary,
    });
    setDevTestRunning(false);
    const finalProcessed = Object.values(byAlgorithm)[0]?.attempts || 0;
    setDevTestProgress({ processed: finalProcessed, total: selectedDrawings.length });
    if (devTestStopRequestedRef.current) {
      setStatusMessage(`Dev test stopped early at ${finalProcessed}/${selectedDrawings.length} drawings.`);
    } else {
      setStatusMessage(`Dev test complete. Evaluated ${selectedDrawings.length} drawing${selectedDrawings.length === 1 ? "" : "s"}.`);
    }
  };

  const startDevTestWithPromptedSample = () => {
    if (devTestRunning) return;

    const suggested = Number.isFinite(devTestSampleSize) && devTestSampleSize > 0
      ? Math.floor(devTestSampleSize)
      : 25;
    const valueRaw = window.prompt("How many random drawings should Dev Test run?", String(suggested));
    if (valueRaw === null) return;

    const parsed = Number.parseInt(valueRaw, 10);
    if (!Number.isFinite(parsed) || parsed < 1) {
      setStatusMessage("Please enter a whole number greater than 0 for Dev Test sample size.");
      return;
    }

    const clamped = Math.min(parsed, Math.max(1, dataset.length));
    setDevTestSampleSize(clamped);
    runDevModelSweep({ sampleSize: clamped });
  };

  const stopDevTestNow = () => {
    if (!devTestRunning) return;
    devTestStopRequestedRef.current = true;
    setStatusMessage("Stopping dev test after current drawing...");
  };

  useEffect(() => {
    if (!devTestRunning) return undefined;

    const timer = setInterval(() => {
      setDevTestElapsedMs(Date.now() - devTestStartedAtRef.current);
    }, 150);

    return () => clearInterval(timer);
  }, [devTestRunning]);

    setStatusMessage(`Dev test complete. Evaluated ${selectedDrawings.length} drawing${selectedDrawings.length === 1 ? "" : "s"}.`);
  };

  useEffect(() => {
    saveDevTrainingMode(trainingMode);
  }, [trainingMode]);

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
    const trainingVec = trainingMode ? applyTrainingNoise(vec) : vec;
    const { hyperDraw, hyperDrawV2, hyperDrawV2X } = runLiveAlgorithmsPrepared(vec, preparedLiveDataset);
    const results = devMode ? runAlgorithms(vec, dataset) : [];

    setCompareResults({
      hyperDraw: { label: hyperDraw.label },
      hyperDrawV2: { label: hyperDrawV2.label },
      hyperDrawV2X: { label: hyperDrawV2X.label },
    });
    setLastDoneResults(results);
    if (devMode) {
      if (trainingMode) {
        setTrainingSessionAlgorithmStats((previous) =>
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

        setTrainingLifetimeAlgorithmStats((previous) =>
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
      } else {
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
      }
    }

    setCompareStats((previous) => {
      const next = { ...previous, attempts: previous.attempts + 1 };
      const hyperDrawCorrect = hyperDraw.label === prompt;
      const hyperDrawV2Correct = hyperDrawV2.label === prompt;
      const hyperDrawV2XCorrect = hyperDrawV2X.label === prompt;
      const maxCorrect = Math.max(hyperDrawCorrect ? 1 : 0, hyperDrawV2Correct ? 1 : 0, hyperDrawV2XCorrect ? 1 : 0);
      if (maxCorrect === 0 || [hyperDrawCorrect, hyperDrawV2Correct, hyperDrawV2XCorrect].filter(Boolean).length > 1) next.ties += 1;
      else if (hyperDrawCorrect) next.hyperDrawWins += 1;
      else if (hyperDrawV2Correct) next.hyperDrawV2Wins += 1;
      else next.hyperDrawV2XWins += 1;
      return next;
    });

    const profile = profileRef.current;
    const updated = [
      ...dataset,
      { id: randomId(), label: prompt, vector: trainingVec, ts: Date.now(), authorName: profile.name, clientId: profile.clientId },
    ].slice(-2000);
    setDataset(updated);
    saveDataset(updated);

    if (cryptoContextRef.current) {
      syncWithServer({ profile, drawings: updated, cryptoContext: cryptoContextRef.current }).catch(() => {
        // Keep local save successful even if server is unavailable.
      });
    } else {
      // Keep local save successful even if server is unavailable.
    }

    const nextPromptCounts = updated.reduce((acc, item) => {
      acc[item.label] = (acc[item.label] || 0) + 1;
      return acc;
    }, {});
    setPrompt(chooseNextPrompt({ trainingMode, promptCounts: nextPromptCounts }));
    clearCanvas();
    setStatusMessage(trainingMode
      ? "Done! Saved with training noise and queued a less-trained prompt."
      : "Done! Added to dataset and moved to the next prompt.");
  };

  useEffect(() => {
    const onKeyDown = (event) => {
      if (activeTab !== "draw") return;
      if (event.isComposing) return;

      const target = event.target;
      const tagName = target?.tagName;
      const isEditable =
        target?.isContentEditable ||
        tagName === "INPUT" ||
        tagName === "TEXTAREA" ||
        tagName === "SELECT";
      if (isEditable) return;

      const isUndoShortcut = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z";
      if (isUndoShortcut) {
        event.preventDefault();
        stopDrawing();
        undoLastStroke();
        return;
      }

      const isFinishShortcut = event.key === "Enter" || event.key === "NumpadEnter";
      if (isFinishShortcut && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        stopDrawing();
        saveDrawing();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTab, saveDrawing]);

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
      <p className="subtitle">Get a random object, draw it, and let our hilariously judgy AI guess from community sketches. Server sync is end-to-end encrypted in your browser.</p>

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
            <button className="secondary" onClick={undoLastStroke}>Undo</button>
            <button className="secondary" onClick={skipObject}>Skip object</button>
            <button className={`secondary ${devMode ? "active" : ""}`} onClick={() => setDevMode((on) => !on)}>
              {devMode ? "Dev Mode: ON" : "Dev Mode"}
            </button>
            {devMode && (
              <button className={`secondary ${trainingMode ? "active" : ""}`} onClick={() => setTrainingMode((on) => !on)}>
                {trainingMode ? "Training Mode: ON" : "Training Mode"}
              </button>
            )}
          </div>
          <p className="subtitle">Shortcuts: Ctrl/Cmd + Z = Undo, Enter = Done</p>
          {statusMessage && <p className="status-msg">{statusMessage}</p>}
        </section>

        <aside className="card">
          <h2>AI Guess Console</h2>
          <div className="row controls-row">
            <label>
              Model:&nbsp;
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                <option value="hyperdraw_v2x">HyperDraw_v2X (default)</option>
                <option value="hyperdraw_v2">HyperDraw_v2</option>
                <option value="hyperdraw">HyperDraw_v1</option>
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
                <div><strong>HyperDraw_v1</strong></div>
                <div>Guess: {compareResults.hyperDraw.label}</div>
              </div>
              <div className="stat">
                <div><strong>HyperDraw_v2</strong></div>
                <div>Guess: {compareResults.hyperDrawV2.label}</div>
              </div>
              <div className="stat">
                <div><strong>HyperDraw_v2X</strong></div>
                <div>Guess: {compareResults.hyperDrawV2X.label}</div>
              </div>
            </div>
          )}

          <div className="stats">
            <div className="stat"><div>Total drawings</div><div className="big">{dataset.length}</div></div>
            <div className="stat"><div>Objects learned</div><div className="big">{Object.keys(promptCounts).length}</div></div>
          </div>

          <h3>Online now ({onlinePlayers.length})</h3>
          <ul>
            {onlinePlayers.length === 0 ? <li>No active players right now.</li> : onlinePlayers.map((player) => <li key={player.clientId}>{player.name}</li>)}
          </ul>

          <div className="stats">
            <div className="stat"><div>Compare rounds</div><div className="big">{compareStats.attempts}</div></div>
            <div className="stat"><div>HyperDraw_v1 wins</div><div className="big">{compareStats.hyperDrawWins}</div></div>
            <div className="stat"><div>HyperDraw_v2 wins</div><div className="big">{compareStats.hyperDrawV2Wins}</div></div>
            <div className="stat"><div>HyperDraw_v2X wins</div><div className="big">{compareStats.hyperDrawV2XWins}</div></div>
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
              <p>Click <strong>Done</strong> to log correctness rates for the active algorithms (1, 7, 72, 77, 78, 79, and 80).</p>
              <div className="stats dev-performance-grid">
                <div className="stat">
                  <div>Performance score</div>
                  <div className="big">{devPerformance.performanceScore}/100</div>
                </div>
                <div className="stat">
                  <div>Avg speed</div>
                  <div>{devPerformance.averageComputeMs}ms compute</div>
                  <div>{devPerformance.averageQueueDelayMs}ms queue delay</div>
                  <div>{devPerformance.averageLagMs}ms lag estimate</div>
                  <div>{devPerformance.maxComputeMs}ms worst compute</div>
                  <div>{devPerformance.maxQueueDelayMs}ms worst queue delay</div>
                  <div>{devPerformance.slowFrameRatio}% slow-frame rate</div>
                </div>
              </div>
              <div className="stat">
                <div><strong>Key issues</strong></div>
                <ul className="issues-list">
                  {devPerformance.issues.map((issue) => <li key={issue}>{issue}</li>)}
                </ul>
              </div>
              <p>Runs tracked: {devPerformance.runs}</p>

              <div className="row">
                <button className="secondary" onClick={startDevTestWithPromptedSample} disabled={devTestRunning}>
                  Dev Test {Math.max(1, Math.floor(devTestSampleSize || 1))}
                <button className="secondary" onClick={() => runDevModelSweep({ sampleSize: DEV_TEST_SAMPLE_SIZE })} disabled={devTestRunning}>
                  Dev Test {DEV_TEST_SAMPLE_SIZE}
                </button>
                <button className="secondary" onClick={() => runDevModelSweep()} disabled={devTestRunning}>
                  Dev Test All
                </button>
              </div>
              {devTestRunning && (
                <p>Running benchmark: {devTestProgress.processed}/{devTestProgress.total} drawings tested.</p>
              )}
              {devTestReport && (
                <>
                  <p>
                    <strong>{devTestReport.sampleLabel}</strong> · {devTestReport.totalDrawings} drawings · {new Date(devTestReport.generatedAt).toLocaleString()}
                  </p>
                  <div className="algo-grid">
                    {devTestReport.summary.map((algo) => (
                      <div className="stat" key={`dev-test-${algo.id}`}>
                        <div><strong>Algorithm {algo.id}</strong></div>
                        <div>Win rate: {algo.winRate}%</div>
                        <div>Correct: {algo.correct}/{algo.attempts}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}

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
                {!trainingMode && devStatsView === "session" && "Session checks reset on reload."}
                {!trainingMode && devStatsView === "lifetime" && "Lifetime checks are saved in your browser."}
                {trainingMode && devStatsView === "session" && "Training-session checks are separate and reset on reload."}
                {trainingMode && devStatsView === "lifetime" && "Training-lifetime checks are separate and never saved to storage."}
              </p>
              <div className="algo-grid">
                {[...(
                  trainingMode
                    ? (devStatsView === "session" ? trainingSessionAlgorithmStats : trainingLifetimeAlgorithmStats)
                    : (devStatsView === "session" ? sessionAlgorithmStats : algorithmStats)
                )]
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
          <h2>HyperDraw Articles</h2>
          <p className="subtitle">Preview each update below, then click to expand the full article.</p>
          <div className="article-list">
            {ARTICLE_ENTRIES.map((entry) => {
              const isExpanded = expandedArticleId === entry.id;
              const previewParagraphs = entry.paragraphs.slice(0, 2);
              const remainingParagraphs = entry.paragraphs.slice(2);
              return (
                <article
                  key={entry.id}
                  className={`article-preview ${isExpanded ? "expanded" : ""}`}
                  onClick={() => setExpandedArticleId((current) => (current === entry.id ? null : entry.id))}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setExpandedArticleId((current) => (current === entry.id ? null : entry.id));
                    }
                  }}
                  aria-expanded={isExpanded}
                >
                  <div className="article-preview-header">
                    <div>
                      <h3>{entry.title}</h3>
                      <p className="subtitle">{entry.subtitle}</p>
                    </div>
                    <span className="expand-pill">{isExpanded ? "Hide" : "Read more"}</span>
                  </div>
                  {previewParagraphs.map((paragraph, index) => (
                    <p key={`${entry.id}-preview-${index}`}>{paragraph}</p>
                  ))}
                  {!isExpanded && <p className="expand-hint">Click anywhere on this card to expand.</p>}
                  {isExpanded && remainingParagraphs.map((paragraph, index) => (
                    <p key={`${entry.id}-full-${index}`}>{paragraph}</p>
                  ))}
                </article>
              );
            })}
          </div>
          <h2>HyperDraw v2X Update</h2>
          <p className="subtitle">A short breakdown of what changed, and what comes next.</p>
          <article>
            {V2X_ARTICLE_PARAGRAPHS.map((paragraph, index) => (
              <p key={`v2x-article-${index}`}>{paragraph}</p>
            ))}
          </article>
        </section>
      )}

      {devTestPopupOpen && (
        <div className="dev-test-overlay" role="dialog" aria-modal="true" aria-label="Dev test benchmark">
          <section className="card dev-test-popup">
            <div className="dev-test-popup-header">
              <h3>Dev Test Live</h3>
              <div className="row">
                {devTestRunning ? (
                  <button className="warn" onClick={stopDevTestNow}>Stop now</button>
                ) : (
                  <button className="secondary" onClick={() => setDevTestPopupOpen(false)}>Close</button>
                )}
              </div>
            </div>
            <p>
              Time: {(devTestElapsedMs / 1000).toFixed(1)}s · Progress: {devTestProgress.processed}/{devTestProgress.total}
            </p>
            <p>Current drawing: <strong>{devTestLiveSnapshot.currentLabel}</strong></p>
            <div className="algo-grid">
              {devTestLiveSnapshot.currentGuesses.map((algo) => (
                <div className="stat" key={`live-guess-${algo.id}`}>
                  <div><strong>Algorithm {algo.id}</strong></div>
                  <div>Guess: {algo.label}</div>
                  <div>Confidence: {algo.confidence}%</div>
                </div>
              ))}
            </div>
            {devTestReport && (
              <>
                <h4>Live win rates</h4>
                <div className="algo-grid">
                  {devTestReport.summary.map((algo) => (
                    <div className="stat" key={`live-summary-${algo.id}`}>
                      <div><strong>Algorithm {algo.id}</strong></div>
                      <div>Win rate: {algo.winRate}%</div>
                      <div>Correct: {algo.correct}/{algo.attempts}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>
        </div>
      )}
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
