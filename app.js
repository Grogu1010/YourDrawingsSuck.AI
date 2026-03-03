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
const GRID_SIZE = 16;
const ALGORITHM_COUNT = 13;

function randomPrompt() {
  return OBJECTS[Math.floor(Math.random() * OBJECTS.length)];
}

function loadDataset() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
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
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
}

function createDefaultAlgorithmStats() {
  return Array.from({ length: ALGORITHM_COUNT }, (_, index) => ({ id: index + 1, attempts: 0, correct: 0 }));
}

function loadAlgorithmStats() {
  try {
    const raw = localStorage.getItem(ALGO_STATS_STORAGE_KEY);
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
  localStorage.setItem(ALGO_STATS_STORAGE_KEY, JSON.stringify(stats));
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
  const [guess, setGuess] = useState("start drawing");
  const [confidence, setConfidence] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [algorithmStats, setAlgorithmStats] = useState(() => loadAlgorithmStats());
  const [lastDoneResults, setLastDoneResults] = useState([]);

  useEffect(() => {
    saveAlgorithmStats(algorithmStats);
  }, [algorithmStats]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 14;
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = isErasing ? "#ffffff" : "#111827";
    ctx.lineWidth = isErasing ? 26 : 14;
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
    setStatusMessage("");
  };

  const vectorizeCanvas = () => {
    const canvas = canvasRef.current;
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
      hasMeaningfulDrawing: totalInk > 5 && activePixels > 8 && drawnStrokeCount > 0,
    };
  };

  const guessDrawing = () => {
    const drawingStats = getDrawingStats();

    if (!drawingStats.hasMeaningfulDrawing) {
      setStatusMessage("Draw something first — erased/blank canvas cannot be guessed.");
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
    const primary = results[0];
    const conf = Math.max(1, Math.min(99, primary.confidence));
    const lowConfidence = conf < 60;

    setGuess(primary.label);
    setConfidence(conf);
    setLastDoneResults(results);
    setStatusMessage(lowConfidence ? "Low confidence guess — try cleaner strokes for better accuracy." : "");
  };

  const stopDrawingAndGuess = () => {
    stopDrawing();
  };

  useEffect(() => {
    const intervalId = setInterval(() => {
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
          <p className="big">{guess}</p>
          <p>Confidence: {confidence}%</p>
          <div className="stats">
            <div className="stat"><div>Total drawings</div><div className="big">{dataset.length}</div></div>
            <div className="stat"><div>Objects learned</div><div className="big">{Object.keys(promptCounts).length}</div></div>
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
              <p>Click <strong>Done</strong> to log correctness rates for all 13 algorithms.</p>
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
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
