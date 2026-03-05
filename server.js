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
      encryptedIps: parsed.encryptedIps && typeof parsed.encryptedIps === 'object' ? parsed.encryptedIps : {},
      activity: parsed.activity && typeof parsed.activity === 'object' ? parsed.activity : {},
      revision: typeof parsed.revision === 'number' ? parsed.revision : Date.now(),
    };
  } catch {
    return { profiles: {}, drawings: [], encryptedIps: {}, activity: {}, revision: Date.now() };
  }
}

function writeDb(db) {
  fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
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

function sanitizeDrawing(item, profile) {
  if (
    !item ||
    typeof item.id !== 'string' ||
    typeof item.enc !== 'string' ||
    typeof item.iv !== 'string' ||
    typeof item.ver !== 'number'
  ) {
    return null;
  }
  return {
    id: item.id,
    enc: item.enc,
    iv: item.iv,
    ver: item.ver,
    keyHint: typeof item.keyHint === 'string' ? item.keyHint : '',
    clientId: profile.clientId,
    authorName: profile.name,
  };
}

function handleSync(req, res, body) {
  const db = readDb();
  const profile = body?.profile;

  if (!profile || typeof profile.clientId !== 'string' || typeof profile.name !== 'string' || !profile.name.trim()) {
    sendJson(res, 400, { error: 'Invalid profile' });
    return;
  }

  const normalizedProfile = { clientId: profile.clientId, name: profile.name.trim() };
  db.profiles[profile.clientId] = normalizedProfile;
  db.activity[profile.clientId] = Date.now();

  const encryptedIp = body?.encryptedIp;
  if (
    encryptedIp &&
    typeof encryptedIp.id === 'string' &&
    typeof encryptedIp.enc === 'string' &&
    typeof encryptedIp.iv === 'string' &&
    typeof encryptedIp.ver === 'number'
  ) {
    db.encryptedIps[profile.clientId] = {
      id: encryptedIp.id,
      enc: encryptedIp.enc,
      iv: encryptedIp.iv,
      ver: encryptedIp.ver,
      keyHint: typeof encryptedIp.keyHint === 'string' ? encryptedIp.keyHint : '',
    };
  }

  const incomingDrawings = Array.isArray(body?.drawings) ? body.drawings : [];
  const existingIds = new Set(db.drawings.map((drawing) => drawing.id));
  incomingDrawings.forEach((item) => {
    const sanitized = sanitizeDrawing(item, normalizedProfile);
    if (!sanitized || existingIds.has(sanitized.id)) return;
    existingIds.add(sanitized.id);
    db.drawings.push(sanitized);
  });

  db.drawings = db.drawings.map((drawing) => {
    if (drawing.clientId !== normalizedProfile.clientId) return drawing;
    return { ...drawing, authorName: normalizedProfile.name };
  }).slice(-50000);

  db.revision = Date.now();
  writeDb(db);

  const cutoff = Date.now() - 2 * 60 * 1000;
  const online = Object.entries(db.activity)
    .filter(([, ts]) => typeof ts === 'number' && ts >= cutoff)
    .map(([clientId]) => ({
      clientId,
      name: db.profiles[clientId]?.name || 'anonymous',
    }));

  sendJson(res, 200, {
    ok: true,
    revision: String(db.revision),
    drawings: db.drawings,
    online,
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
