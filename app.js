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
    let best = { label: "unknown", dist: Infinity };
    for (const item of dataset) {
      const dist = distance(vec, item.vector);
      if (dist < best.dist) best = { label: item.label, dist };
    }

    const conf = Math.max(1, Math.round((1 - best.dist / 8) * 100));
    setGuess(best.label);
    setConfidence(conf);
  };

  const saveDrawing = () => {
    const vec = vectorizeCanvas();
    const updated = [...dataset, { label: prompt, vector: vec, ts: Date.now() }].slice(-2000);
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
