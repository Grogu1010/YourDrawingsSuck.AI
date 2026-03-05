const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = Number(process.env.PORT || 8787);
const DB_PATH = process.env.DRAWINGS_DB_PATH || path.join(__dirname, 'drawings-db.json');

function readDb() {
  try {
    const raw = fs.readFileSync(DB_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') throw new Error('invalid db');
    return {
      profiles: parsed.profiles && typeof parsed.profiles === 'object' ? parsed.profiles : {},
      drawings: Array.isArray(parsed.drawings) ? parsed.drawings : [],
      revision: typeof parsed.revision === 'number' ? parsed.revision : Date.now(),
    };
  } catch {
    return { profiles: {}, drawings: [], revision: Date.now() };
  }
}

function writeDb(db) {
  fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
}

function getClientIp(req) {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string' && forwarded.trim()) {
    return forwarded.split(',')[0].trim();
  }
  return req.socket.remoteAddress || 'unknown';
}

function sendJson(res, code, payload) {
  res.writeHead(code, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST,OPTIONS',
  });
  res.end(JSON.stringify(payload));
}

function sanitizeDrawing(item, profile, ip) {
  if (!item || typeof item.label !== 'string' || !Array.isArray(item.vector) || typeof item.ts !== 'number') return null;
  return {
    id: typeof item.id === 'string' && item.id ? item.id : `${item.ts}_${Math.random().toString(36).slice(2, 9)}`,
    label: item.label,
    vector: item.vector.slice(0, 256),
    ts: item.ts,
    clientId: profile.clientId,
    authorName: profile.name,
    ip,
  };
}

function handleSync(req, res, body) {
  const db = readDb();
  const ip = getClientIp(req);
  const profile = body?.profile;

  if (!profile || typeof profile.clientId !== 'string' || typeof profile.name !== 'string' || !profile.name.trim()) {
    sendJson(res, 400, { error: 'Invalid profile' });
    return;
  }

  const normalizedProfile = { clientId: profile.clientId, name: profile.name.trim(), ip };
  db.profiles[profile.clientId] = normalizedProfile;

  const incomingDrawings = Array.isArray(body?.drawings) ? body.drawings : [];
  const existingIds = new Set(db.drawings.map((drawing) => drawing.id));
  incomingDrawings.forEach((item) => {
    const sanitized = sanitizeDrawing(item, normalizedProfile, ip);
    if (!sanitized || existingIds.has(sanitized.id)) return;
    existingIds.add(sanitized.id);
    db.drawings.push(sanitized);
  });

  db.drawings = db.drawings.map((drawing) => {
    if (drawing.clientId !== normalizedProfile.clientId) return drawing;
    return { ...drawing, authorName: normalizedProfile.name, ip };
  }).slice(-50000);

  db.revision = Date.now();
  writeDb(db);

  sendJson(res, 200, {
    ok: true,
    revision: String(db.revision),
    drawings: db.drawings,
  });
}

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') {
    sendJson(res, 200, { ok: true });
    return;
  }

  if (req.method !== 'POST' || req.url !== '/api/sync') {
    sendJson(res, 404, { error: 'Not found' });
    return;
  }

  let raw = '';
  req.on('data', (chunk) => {
    raw += chunk;
    if (raw.length > 20 * 1024 * 1024) {
      req.destroy();
    }
  });

  req.on('end', () => {
    try {
      const body = raw ? JSON.parse(raw) : {};
      handleSync(req, res, body);
    } catch {
      sendJson(res, 400, { error: 'Invalid JSON' });
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`YourDrawingsSuck.AI sync server listening on http://0.0.0.0:${PORT}`);
});
