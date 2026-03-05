# YourDrawingsSuck.AI
YOUR DRAWINGS SUCK, YOU A**HOLE!!

## Optional shared sync server

The app still works fully offline/local by default.

If you want everybody's drawings synced to one server:

1. Start the included server on your host machine:

```bash
node server.js
```

2. Point clients to that server by setting one of these:
   - `window.__YDS_SERVER_URL = "http://YOUR_SERVER_IP:8787"` before `app.js` loads, or
   - localStorage key `yourdrawingssuckai.serverUrl.v1` to that URL.

When a user first opens the app, they get a `Please enter a name` prompt. Drawings are always saved locally and, when a server URL is configured, they are also synced to the server with:

- `authorName` (their chosen name)
- `clientId` (browser identity)
- server-observed IP address

Existing local drawings are bulk uploaded on first successful sync.

### Rename existing drawings for one user

The app exposes a helper so Settings can rename a user and backfill old entries:

```js
window.changeDrawingPlayerName("New Name")
```

That updates local drawings and triggers a full server sync so previous drawings also get the new name.
