const { useEffect, useRef, useState } = React;

const OBJECTS = [
  "cat", "dog", "tree", "house", "car", "bicycle", "apple", "banana", "fish", "bird",
  "cloud", "sun", "moon", "star", "flower", "cup", "book", "chair", "table", "phone",
  "clock", "pizza", "burger", "ice cream", "guitar", "drum", "camera", "airplane", "rocket", "boat",
  "train", "bus", "robot", "monster", "dragon", "castle", "crown", "shoe", "hat", "glasses",
  "toothbrush", "key", "lock", "lamp", "cookie", "donut", "snail", "frog", "whale", "snack"
];

const STORAGE_KEY = "yourdrawingssuckai.dataset.v1";
const GRID_SIZE = 16;
const K_NEIGHBORS = 9;

function randomPrompt() {
  return OBJECTS[Math.floor(Math.random() * OBJECTS.length)];
}

function loadDataset() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch {
    return [];
  }
}

function saveDataset(dataset) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
}

function App() {
  const canvasRef = useRef(null);
  const [prompt, setPrompt] = useState(randomPrompt());
  const [dataset, setDataset] = useState([]);
  const [guess, setGuess] = useState("-");
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    setDataset(loadDataset());
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "black";
    ctx.lineWidth = 9;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    let drawing = false;

    const getPos = (e) => {
      const rect = canvas.getBoundingClientRect();
      const touch = e.touches?.[0];
      const x = ((touch?.clientX ?? e.clientX) - rect.left) * (canvas.width / rect.width);
      const y = ((touch?.clientY ?? e.clientY) - rect.top) * (canvas.height / rect.height);
      return { x, y };
    };

    const start = (e) => {
      drawing = true;
      const { x, y } = getPos(e);
      ctx.beginPath();
      ctx.moveTo(x, y);
      e.preventDefault();
    };

    const draw = (e) => {
      if (!drawing) return;
      const { x, y } = getPos(e);
      ctx.lineTo(x, y);
      ctx.stroke();
      e.preventDefault();
    };

    const end = () => {
      drawing = false;
    };

    canvas.addEventListener("mousedown", start);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", end);

    canvas.addEventListener("touchstart", start, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", end);

    return () => {
      canvas.removeEventListener("mousedown", start);
      canvas.removeEventListener("mousemove", draw);
      window.removeEventListener("mouseup", end);
      canvas.removeEventListener("touchstart", start);
      canvas.removeEventListener("touchmove", draw);
      canvas.removeEventListener("touchend", end);
    };
  }, []);

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setGuess("-");
    setConfidence(0);
  };

  const vectorizeCanvas = () => {
    const offscreen = document.createElement("canvas");
    offscreen.width = GRID_SIZE;
    offscreen.height = GRID_SIZE;
    const octx = offscreen.getContext("2d");
    octx.drawImage(canvasRef.current, 0, 0, GRID_SIZE, GRID_SIZE);
    const data = octx.getImageData(0, 0, GRID_SIZE, GRID_SIZE).data;
    const vec = [];
    for (let i = 0; i < data.length; i += 4) {
      const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
      vec.push(1 - lum / 255);
    }
    return vec;
  };

  const distance = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  };

  const guessDrawing = () => {
    const vec = vectorizeCanvas();
    if (!dataset.length) {
      setGuess("No clue yet. Draw & save more first!");
      setConfidence(0);
      return;
    }

    const ranked = [];
    for (const item of dataset) {
      const dist = distance(vec, item.vector);
      ranked.push({ label: item.label, dist });
    }

    ranked.sort((a, b) => a.dist - b.dist);
    const neighbors = ranked.slice(0, Math.min(K_NEIGHBORS, ranked.length));

    const voteByLabel = {};
    for (const n of neighbors) {
      const weight = 1 / (n.dist + 0.001);
      voteByLabel[n.label] = (voteByLabel[n.label] || 0) + weight;
    }

    const sortedVotes = Object.entries(voteByLabel)
      .sort((a, b) => b[1] - a[1]);

    const [bestLabel, bestVote] = sortedVotes[0];
    const secondVote = sortedVotes[1]?.[1] || 0;
    const voteTotal = sortedVotes.reduce((sum, [, vote]) => sum + vote, 0);
    const voteShare = voteTotal ? bestVote / voteTotal : 0;
    const separation = bestVote ? (bestVote - secondVote) / bestVote : 0;
    const conf = Math.round(Math.min(99, Math.max(1, (voteShare * 70 + separation * 30) * 100)));

    setGuess(bestLabel);
    setConfidence(conf);
  };

  const saveDrawing = () => {
    const vec = vectorizeCanvas();
    const updated = [...dataset, { label: prompt, vector: vec, ts: Date.now() }];
    setDataset(updated);
    saveDataset(updated);
    setPrompt(randomPrompt());
    clearCanvas();
  };

  const promptCounts = dataset.reduce((acc, item) => {
    acc[item.label] = (acc[item.label] || 0) + 1;
    return acc;
  }, {});

  return (
    <main className="app">
      <h1>YourDrawingsSuck.AI</h1>
      <p className="subtitle">Get a random object, draw it, and let our hilariously judgy AI guess from community sketches.</p>

      <div className="grid">
        <section className="card">
          <h2>Draw this: <span style={{ color: "#6ee7b7" }}>{prompt}</span></h2>
          <canvas ref={canvasRef} width="500" height="500" aria-label="drawing area"></canvas>
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
