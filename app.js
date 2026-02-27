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
  const drawingRevisionRef = useRef(0);
  const lastGuessedRevisionRef = useRef(-1);

  const [dataset, setDataset] = useState(() => loadDataset());
  const [prompt, setPrompt] = useState(() => randomPrompt());
  const [guess, setGuess] = useState("start drawing");
  const [confidence, setConfidence] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [isErasing, setIsErasing] = useState(false);

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
      return;
    }

    const { vec } = drawingStats;
    const newestTimestamp = Math.max(...dataset.map((item) => item.ts));
    const scoredExamples = dataset.map((item) => {
      const pixelDist = distance(vec, item.vector) / Math.sqrt(vec.length);
      const recencyBonus = Math.max(0, (item.ts - newestTimestamp) / (1000 * 60 * 60 * 24 * 14));
      return {
        label: item.label,
        score: pixelDist - recencyBonus,
      };
    });

    const ranked = scoredExamples.sort((a, b) => a.score - b.score);
    const [best = null, second = null] = ranked;

    if (!best) {
      setGuess("unknown");
      setConfidence(0);
      return;
    }

    const margin = Math.max(0, (second?.score ?? best.score + 1) - best.score);
    const conf = Math.max(1, Math.min(99, Math.round((1 - Math.min(best.score, 1)) * 100 + margin * 20)));
    const lowConfidence = conf < 60;

    setGuess(best.label);
    setConfidence(conf);
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
    const updated = [...dataset, { label: prompt, vector: vec, ts: Date.now() }].slice(-2000);
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
