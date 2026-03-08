# YourDrawingsSuck.AI
YOUR DRAWINGS SUCK, YOU A**HOLE!!

## Set up the optional sync server (easy version)

> You only need this if you want multiple people/devices sharing drawings.
> If you skip this, the app still works fully offline and saves drawings locally.

### 1) Start the server

From the project folder, run:

```bash
node server.js
```

By default, the server runs at:

- `http://localhost:8787` (on your own machine)
- `http://YOUR_SERVER_IP:8787` (from other devices on your network)

### 2) Tell the app where the server is

Choose **one** method:

#### Option A (best for developers): set `window.__YDS_SERVER_URL`

Set this **before** `app.js` loads:

```html
<script>
  window.__YDS_SERVER_URL = "http://YOUR_SERVER_IP:8787";
</script>
<script src="app.js"></script>
```

#### Option B: save the URL in localStorage

Open DevTools Console and run:

```js
localStorage.setItem("yourdrawingssuckai.serverUrl.v1", "http://YOUR_SERVER_IP:8787");
```

Then refresh the page.

### 3) Confirm it worked

When a user opens the app, they are prompted for a name (`Please enter a name`).

Drawings are always saved locally. If a server URL is configured, drawings are also synced to the server with:

- `authorName` (chosen name)
- `clientId` (browser identity)
- server-observed IP address

If there are existing local drawings, they are bulk uploaded on the first successful sync.

---

## Rename existing drawings for one user

You can rename a user and backfill previous entries by running:

```js
window.changeDrawingPlayerName("New Name")
```

This updates local drawings and triggers a full sync so older server entries also get the new name.
