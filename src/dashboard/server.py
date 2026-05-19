from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify


@dataclass
class SharedState:
    frame: np.ndarray | None = None
    status: dict[str, Any] = field(default_factory=dict)
    jpeg_quality: int = 80
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, frame: np.ndarray, status: dict[str, Any]) -> None:
        with self.lock:
            self.frame = frame.copy()
            self.status = dict(status)


def create_app(state: SharedState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return """
        <!doctype html>
        <html>
          <head>
            <title>Drone Tracker</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
            <style>
              body { margin: 0; background: #101418; color: #eef2f6; font-family: system-ui, sans-serif; }
              header { padding: 14px 18px; background: #171d24; display: flex; justify-content: space-between; }
              main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; }
              img { width: 100%; height: auto; background: #050608; display: block; }
              aside { display: grid; gap: 16px; align-content: start; }
              #map { height: 300px; background: #171d24; border-radius: 6px; overflow: hidden; }
              pre { white-space: pre-wrap; background: #171d24; padding: 12px; border-radius: 6px; }
              @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
            </style>
          </head>
          <body>
            <header><strong>Drone Tracker</strong><span>Local onboard dashboard</span></header>
            <main>
              <img src="/video.mjpg" alt="Live tracking stream">
              <aside>
                <div id="map"></div>
                <pre id="status">Loading...</pre>
              </aside>
            </main>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
              const map = L.map('map').setView([0, 0], 2);
              L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors'
              }).addTo(map);
              const markers = new Map();
              let mapCentered = false;

              async function refreshStatus() {
                const response = await fetch('/status.json');
                const status = await response.json();
                document.getElementById('status').textContent = JSON.stringify(status, null, 2);
                const activeKeys = new Set();
                for (const pin of status.detection_pins || []) {
                  const key = String(pin.track_id);
                  activeKeys.add(key);
                  const latlng = [pin.latitude, pin.longitude];
                  const text = `Human ${pin.track_id} last seen | ${Number(pin.confidence).toFixed(2)} | ${pin.source}`;
                  if (markers.has(key)) {
                    markers.get(key).setLatLng(latlng).bindPopup(text);
                  } else {
                    markers.set(key, L.circleMarker(latlng, {
                      radius: 9,
                      color: '#16a34a',
                      weight: 3,
                      fillColor: '#22c55e',
                      fillOpacity: 0.8
                    }).addTo(map).bindPopup(text));
                  }
                  if (!mapCentered) {
                    map.setView(latlng, 16);
                    mapCentered = true;
                  }
                }
                for (const [key, marker] of markers) {
                  if (!activeKeys.has(key)) {
                    map.removeLayer(marker);
                    markers.delete(key);
                  }
                }
              }
              setInterval(refreshStatus, 1000);
              refreshStatus();
            </script>
          </body>
        </html>
        """

    @app.get("/status.json")
    def status() -> Response:
        with state.lock:
            payload = dict(state.status)
        return jsonify(payload)

    @app.get("/video.mjpg")
    def video() -> Response:
        return Response(_mjpeg_frames(state), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def _mjpeg_frames(state: SharedState):
    while True:
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
            quality = state.jpeg_quality
        if frame is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for frames", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


class DashboardServer:
    def __init__(self, host: str, port: int, jpeg_quality: int) -> None:
        self.state = SharedState(jpeg_quality=jpeg_quality)
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        app = create_app(self.state)
        self._thread = threading.Thread(
            target=lambda: app.run(host=self.host, port=self.port, threaded=True, use_reloader=False),
            daemon=True,
        )
        self._thread.start()
