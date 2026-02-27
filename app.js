const { useEffect, useMemo, useRef, useState } = React;

const OBJECTS = [
  "cat", "dog", "tree", "house", "car", "bicycle", "apple", "banana", "fish", "bird",
  "cloud", "sun", "moon", "star", "flower", "cup", "book", "chair", "table", "phone",
  "clock", "pizza", "burger", "ice cream", "guitar", "drum", "camera", "airplane", "rocket", "boat",
  "train", "bus", "robot", "monster", "dragon", "castle", "crown", "shoe", "hat", "glasses",
  "toothbrush", "key", "lock", "lamp", "cookie", "donut", "snail", "frog", "whale", "tennis ball"
];

const STORAGE_KEY = "yourdrawingssuckai.dataset.v1";
const GRID_SIZE = 16;

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
        item.vector.length === GRID_SIZE * GRID_SIZE
    );
  } catch {
    return [];
  }
}

function extractShapeFeatures(vector) {
  const threshold = 0.2;
  const active = [];
  let ink = 0;

  for (let i = 0; i < vector.length; i += 1) {
    const value = vector[i];
    ink += value;
    if (value > threshold) {
      active.push(i);
    }
  }

  if (active.length === 0) {
    return {
      fillRatio: 0,
      aspectRatio: 1,
      compactness: 0,
      symmetryX: 1,
      symmetryY: 1,
      edgeDensity: 0,
      lengthNorm: 0,
      straightness: 0,
    };
  }

  let minX = GRID_SIZE;
  let minY = GRID_SIZE;
  let maxX = 0;
  let maxY = 0;

  for (const index of active) {
    const x = index % GRID_SIZE;
    const y = Math.floor(index / GRID_SIZE);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const boxArea = width * height;

  let edgeCount = 0;
  let symmetryMatchesX = 0;
  let symmetryMatchesY = 0;
  let symmetryChecksX = 0;
  let symmetryChecksY = 0;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const idx = y * GRID_SIZE + x;
      const on = vector[idx] > threshold;
      if (!on) continue;

      const neighbors = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ];

      if (
        neighbors.some(([nx, ny]) => {
          if (nx < minX || ny < minY || nx > maxX || ny > maxY) return true;
          return vector[ny * GRID_SIZE + nx] <= threshold;
        })
      ) {
        edgeCount += 1;
      }

      const mirrorX = maxX - (x - minX);
      const mirrorY = maxY - (y - minY);

      symmetryChecksX += 1;
      symmetryChecksY += 1;
      if (vector[y * GRID_SIZE + mirrorX] > threshold) symmetryMatchesX += 1;
      if (vector[mirrorY * GRID_SIZE + x] > threshold) symmetryMatchesY += 1;
    }
  }

  return {
    fillRatio: ink / (GRID_SIZE * GRID_SIZE),
    aspectRatio: width / Math.max(height, 1),
    compactness: active.length / Math.max(boxArea, 1),
    symmetryX: symmetryMatchesX / Math.max(symmetryChecksX, 1),
    symmetryY: symmetryMatchesY / Math.max(symmetryChecksY, 1),
    edgeDensity: edgeCount / Math.max(active.length, 1),
    lengthNorm: 0,
    straightness: 0,
  };
}

function extractStrokeFeatures(strokes) {
  if (!strokes.length) {
    return { lengthNorm: 0, straightness: 0, strokeCount: 0 };
  }

  const diagonal = Math.sqrt(500 * 500 + 500 * 500);
  let totalLength = 0;
  let weightedStraightness = 0;

  for (const stroke of strokes) {
    if (stroke.length < 2) continue;

    let length = 0;
    for (let i = 1; i < stroke.length; i += 1) {
      const dx = stroke[i].x - stroke[i - 1].x;
      const dy = stroke[i].y - stroke[i - 1].y;
      length += Math.sqrt(dx * dx + dy * dy);
    }

    const dx = stroke[stroke.length - 1].x - stroke[0].x;
    const dy = stroke[stroke.length - 1].y - stroke[0].y;
    const directDistance = Math.sqrt(dx * dx + dy * dy);
    const straightness = directDistance / Math.max(length, 1);

    totalLength += length;
    weightedStraightness += straightness * length;
  }

  return {
    lengthNorm: Math.min(totalLength / diagonal, 1.5),
    straightness: weightedStraightness / Math.max(totalLength, 1),
    strokeCount: strokes.length,
  };
}

function combineFeatures(vector, strokes = []) {
  return {
    ...extractShapeFeatures(vector),
    ...extractStrokeFeatures(strokes),
  };
}

function featureDistance(a, b) {
  const aspectA = Math.log(a.aspectRatio + 1e-6);
  const aspectB = Math.log(b.aspectRatio + 1e-6);
  const strokeCountA = Math.min(a.strokeCount || 0, 8) / 8;
  const strokeCountB = Math.min(b.strokeCount || 0, 8) / 8;

  const terms = [
    [a.fillRatio, b.fillRatio, 1.4],
    [aspectA, aspectB, 1.1],
    [a.compactness, b.compactness, 1.3],
    [a.symmetryX, b.symmetryX, 0.8],
    [a.symmetryY, b.symmetryY, 0.8],
    [a.edgeDensity, b.edgeDensity, 1.1],
    [a.lengthNorm, b.lengthNorm, 1.3],
    [a.straightness, b.straightness, 1.4],
    [strokeCountA, strokeCountB, 0.7],
  ];

  let total = 0;
  for (const [x, y, weight] of terms) {
    const delta = x - y;
    total += weight * delta * delta;
  }
  return Math.sqrt(total);
}

function saveDataset(dataset) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
}

function distance(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    total += d * d;
  }
  return Math.sqrt(total);
}

function cosineDistance(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 1;
  return 1 - dot / denom;
}

function buildPrototypes(dataset) {
  const byLabel = new Map();

  for (const item of dataset) {
    if (!byLabel.has(item.label)) {
      byLabel.set(item.label, {
        count: 0,
        vectorSum: new Array(GRID_SIZE * GRID_SIZE).fill(0),
        featureSum: {
          fillRatio: 0,
          aspectRatio: 0,
          compactness: 0,
          symmetryX: 0,
          symmetryY: 0,
          edgeDensity: 0,
          lengthNorm: 0,
          straightness: 0,
          strokeCount: 0,
        },
      });
    }

    const bucket = byLabel.get(item.label);
    const features = item.features || combineFeatures(item.vector, []);
    bucket.count += 1;

    for (let i = 0; i < item.vector.length; i += 1) {
      bucket.vectorSum[i] += item.vector[i];
    }

    for (const key of Object.keys(bucket.featureSum)) {
      bucket.featureSum[key] += features[key] || 0;
    }
  }

  return [...byLabel.entries()].map(([label, bucket]) => ({
    label,
    count: bucket.count,
    vector: bucket.vectorSum.map((value) => value / bucket.count),
    features: Object.fromEntries(
      Object.entries(bucket.featureSum).map(([key, value]) => [key, value / bucket.count])
    ),
  }));
}

function App() {
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const strokesRef = useRef([]);
  const activeStrokeRef = useRef(null);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => randomPrompt());
  const [guess, setGuess] = useState("unknown");
  const [confidence, setConfidence] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);

  const prototypes = useMemo(() => buildPrototypes(dataset), [dataset]);

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
    setGuess("unknown");
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
      setGuess("unknown");
      setConfidence(0);
      return;
    }

    if (dataset.length === 0) {
      setGuess("Need training data first");
      setConfidence(0);
      setStatusMessage("Train me with a few drawings before guessing.");
      return;
    }

    const { vec } = drawingStats;
    const features = combineFeatures(vec, strokesRef.current);
    const scoredExamples = dataset.map((item) => {
      const itemFeatures = item.features || combineFeatures(item.vector, []);
      const pixelDist = distance(vec, item.vector) / Math.sqrt(vec.length);
      const angularDist = cosineDistance(vec, item.vector);
      const shapeDist = featureDistance(features, itemFeatures);
      const dist = pixelDist * 1.9 + angularDist * 1.4 + shapeDist * 1.7;
      return { label: item.label, dist };
    });

    const scoredPrototypes = prototypes.map((proto) => {
      const pixelDist = distance(vec, proto.vector) / Math.sqrt(vec.length);
      const angularDist = cosineDistance(vec, proto.vector);
      const shapeDist = featureDistance(features, proto.features);
      const sampleBonus = Math.min(proto.count, 12) / 120;
      const dist = pixelDist * 1.6 + angularDist * 1.2 + shapeDist * 1.5 - sampleBonus;
      return { label: proto.label, dist };
    });

    const votes = new Map();
    const nearest = [...scoredExamples].sort((a, b) => a.dist - b.dist).slice(0, 9);

    for (const neighbor of nearest) {
      const weight = 1 / Math.max(neighbor.dist, 0.05);
      votes.set(neighbor.label, (votes.get(neighbor.label) || 0) + weight);
    }

    for (const proto of scoredPrototypes) {
      const weight = 0.9 / Math.max(proto.dist, 0.05);
      votes.set(proto.label, (votes.get(proto.label) || 0) + weight);
    }

    const ranked = [...votes.entries()].sort((a, b) => b[1] - a[1]);
    const [bestLabel = "unknown", bestScore = 0] = ranked[0] || [];
    const secondScore = ranked[1]?.[1] || 0;
    const voteTotal = ranked.reduce((sum, [, score]) => sum + score, 0);
    const margin = bestScore - secondScore;
    const certainty = voteTotal > 0 ? bestScore / voteTotal : 0;
    const prototypeWeight = Math.min(prototypes.length / 10, 1);
    const conf = Math.max(
      1,
      Math.min(99, Math.round((certainty * 0.65 + margin * 0.25 + prototypeWeight * 0.1) * 100))
    );
    const lowConfidence = conf < 60 || margin < 0.12;

    setGuess(lowConfidence ? "unknown" : bestLabel);
    setConfidence(conf);
    setStatusMessage(lowConfidence ? "Not confident enough to guess yet — try cleaner strokes." : "");
  };

  const stopDrawingAndGuess = () => {
    stopDrawing();
    guessDrawing();
  };

  const saveDrawing = () => {
    const drawingStats = getDrawingStats();

    if (!drawingStats.hasMeaningfulDrawing) {
      setStatusMessage("Nope — draw first. Blank/erased canvas won't be saved to training.");
      return;
    }

    const { vec } = drawingStats;
    const features = combineFeatures(vec, strokesRef.current);
    const updated = [...dataset, { label: prompt, vector: vec, features, ts: Date.now() }].slice(-2000);
    setDataset(updated);
    saveDataset(updated);
    setPrompt(randomPrompt());
    clearCanvas();
    setStatusMessage("Saved to training set. Nice.");
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
            <button className="secondary" onClick={saveDrawing}>Save to training set + Next prompt</button>
            <button className="warn" onClick={clearCanvas}>Clear</button>
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
        </aside>
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
