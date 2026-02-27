const { useEffect, useMemo, useRef, useState } = React;

const OBJECTS = [
  "cat", "dog", "tree", "house", "car", "bicycle", "apple", "banana", "fish", "bird",
  "cloud", "sun", "moon", "star", "flower", "cup", "book", "chair", "table", "phone",
  "clock", "pizza", "burger", "ice cream", "guitar", "drum", "camera", "airplane", "rocket", "boat",
  "train", "bus", "robot", "monster", "dragon", "castle", "crown", "shoe", "hat", "glasses",
  "toothbrush", "key", "lock", "lamp", "cookie", "donut", "snail", "frog", "whale", "snack"
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

function App() {
  const canvasRef = useRef(null);
  const isDrawingRef = useRef(false);
  const strokesRef = useRef([]);
  const activeStrokeRef = useRef(null);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => randomPrompt());
  const [guess, setGuess] = useState("unknown");
  const [confidence, setConfidence] = useState(0);

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

  const guessDrawing = () => {
    if (dataset.length === 0) {
      setGuess("Need training data first");
      setConfidence(0);
      return;
    }

    const vec = vectorizeCanvas();
    const features = combineFeatures(vec, strokesRef.current);
    let best = { label: "unknown", dist: Infinity };

    for (const item of dataset) {
      const pixelDist = distance(vec, item.vector);
      const itemFeatures = item.features || combineFeatures(item.vector, []);
      const shapeDist = featureDistance(features, itemFeatures);
      const dist = pixelDist * 0.7 + shapeDist * 2.2;
      if (dist < best.dist) best = { label: item.label, dist };
    }

    const conf = Math.max(1, Math.round((1 - best.dist / 11) * 100));
    setGuess(best.label);
    setConfidence(conf);
  };

  const saveDrawing = () => {
    const vec = vectorizeCanvas();
    const features = combineFeatures(vec, strokesRef.current);
    const updated = [...dataset, { label: prompt, vector: vec, features, ts: Date.now() }].slice(-2000);
    setDataset(updated);
    saveDataset(updated);
    setPrompt(randomPrompt());
    clearCanvas();
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
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          ></canvas>
          <div className="row">
            <button className="primary" onClick={guessDrawing}>Guess my drawing</button>
            <button className="secondary" onClick={saveDrawing}>Save to training set + Next prompt</button>
            <button className="warn" onClick={clearCanvas}>Clear</button>
          </div>
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
