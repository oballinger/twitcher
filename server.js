const express = require("express");
const cors    = require("cors");
const fetch   = (...args) => import("node-fetch").then(({ default: f }) => f(...args));

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));
app.get("/", (_req, res) => res.sendFile(__dirname + "/frontend.html"));

const TFL_KEY = process.env.TFL_KEY;
if (!TFL_KEY) console.warn("[warn] TFL_KEY not set — /api/cameras will fail");

// ── Cameras list ──────────────────────────────────────────────────────────────
app.get("/api/cameras", async (req, res) => {
  try {
    const response = await fetch(
      `https://api.tfl.gov.uk/Place/Type/JamCam?app_key=${TFL_KEY}`
    );
    if (!response.ok) {
      const text = await response.text();
      console.error(`TfL API error ${response.status}: ${text.slice(0, 200)}`);
      return res.status(502).json({ error: `TfL API returned ${response.status}` });
    }
    const rawData = await response.json();

    const cleaned = rawData.map(cam => {
      const props = Object.fromEntries(
        cam.additionalProperties.map(p => [p.key, p.value])
      );
      return {
        id:        cam.id,
        name:      cam.commonName,
        lat:       cam.lat,
        lon:       cam.lon,
        available: props.available === "true",
        imageUrl:  props.imageUrl,
        videoUrl:  props.videoUrl,
        view:      props.view
      };
    });

    res.json(cleaned);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch TfL data" });
  }
});

// ── MJPEG stream proxy ────────────────────────────────────────────────────────
// Proxies /api/stream?videoUrl=... → Python /stream?videoUrl=...
// so the browser doesn't have to hit port 5000 directly (avoids CORS).
app.get("/api/stream", async (req, res) => {
  const { videoUrl, fps } = req.query;
  if (!videoUrl) return res.status(400).json({ error: "missing videoUrl" });

  const params = new URLSearchParams({ videoUrl, fps: fps || "10" });
  const pyRes  = await fetch(`http://localhost:5000/stream?${params}`);

  if (!pyRes.ok) {
    return res.status(502).json({ error: "Detection server unavailable" });
  }

  // Forward the MJPEG headers and pipe the body straight through
  res.setHeader("Content-Type",  pyRes.headers.get("content-type"));
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Access-Control-Allow-Origin", "*");

  pyRes.body.pipe(res);

  req.on("close", () => pyRes.body.destroy());
});

// ── Single-frame JSON detection ───────────────────────────────────────────────
app.post("/api/detect", async (req, res) => {
  try {
    const { videoUrl, imageUrl } = req.body;
    if (!videoUrl && !imageUrl) return res.status(400).json({ error: "missing videoUrl" });

    const pyRes = await fetch("http://localhost:5000/detect", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ videoUrl: videoUrl || imageUrl })
    });

    if (!pyRes.ok) {
      const text = await pyRes.text();
      return res.status(502).json({ error: text });
    }

    res.json(await pyRes.json());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Batch detection ───────────────────────────────────────────────────────────
app.post("/api/detect-batch", async (req, res) => {
  try {
    const pyRes = await fetch("http://localhost:5000/detect-batch", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(req.body)
    });
    if (!pyRes.ok) {
      const text = await pyRes.text();
      return res.status(502).json({ error: text });
    }
    res.json(await pyRes.json());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Client config (non-secret keys for browser use) ──────────────────────────
app.get("/api/config", (_req, res) => {
  res.json({ googleKey: process.env.GOOGLE_KEY || null });
});

// ── Police stations (static JSON file) ───────────────────────────────────────
const stations = require("./police_stations.json");
app.get("/api/police-stations", (_req, res) => res.json(stations));

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
