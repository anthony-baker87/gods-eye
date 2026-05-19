from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file


@dataclass
class SharedState:
    frame: np.ndarray | None = None
    status: dict[str, Any] = field(default_factory=dict)
    jpeg_quality: int = 80
    lock: threading.Lock = field(default_factory=threading.Lock)
    camera_clicks: list[dict[str, Any]] = field(default_factory=list)

    def update(self, frame: np.ndarray, status: dict[str, Any]) -> None:
        with self.lock:
            self.frame = frame.copy()
            self.status = dict(status)

    def add_camera_click(self, click: dict[str, Any]) -> None:
        with self.lock:
            self.camera_clicks.append(click)

    def drain_camera_clicks(self) -> list[dict[str, Any]]:
        with self.lock:
            clicks = list(self.camera_clicks)
            self.camera_clicks.clear()
        return clicks


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
              main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(420px, 32vw); gap: 16px; padding: 16px; }
              #video { width: 100%; height: auto; background: #050608; display: block; cursor: crosshair; }
              aside { display: grid; gap: 16px; align-content: start; }
              #map { height: min(56vh, 620px); min-height: 440px; background: #171d24; border-radius: 6px; overflow: hidden; }
              pre { white-space: pre-wrap; background: #171d24; padding: 12px; border-radius: 6px; max-height: 38vh; overflow: auto; }
              @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
            </style>
          </head>
          <body>
            <header><strong>Drone Tracker</strong><span>Local onboard dashboard</span></header>
            <main>
              <img id="video" src="/video.mjpg" alt="Live tracking stream">
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

              function formatTime(value) {
                if (!value) return 'unknown time';
                return new Date(Number(value) * 1000).toLocaleString();
              }

              function escapeHtml(value) {
                return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
                  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
                })[ch]);
              }

              function snapshotUrl(pin) {
                const snapshot = pin.snapshot || {};
                const path = snapshot.full_image_path || snapshot.crop_image_path;
                return path ? `/snapshot?path=${encodeURIComponent(path)}` : null;
              }

              function popupHtml(pin) {
                const confidence = pin.confidence == null ? '' : `<div>Confidence: ${Number(pin.confidence).toFixed(2)}</div>`;
                const pixel = pin.pixel ? `<div>Camera pixel: ${pin.pixel[0]}, ${pin.pixel[1]}</div>` : '';
                const imageUrl = snapshotUrl(pin);
                const image = imageUrl ? `<img src="${imageUrl}" style="width:260px;max-width:100%;margin-top:8px;border-radius:4px;">` : '';
                return `
                  <strong>${escapeHtml(pin.label || 'pin')}</strong>
                  <div>${escapeHtml(formatTime(pin.timestamp || pin.last_seen))}</div>
                  ${confidence}
                  <div>Source: ${escapeHtml(pin.source)}</div>
                  ${pixel}
                  ${image}
                `;
              }

              function upsertMarker(key, pin, options) {
                const latlng = [pin.latitude, pin.longitude];
                if (markers.has(key)) {
                  markers.get(key).setLatLng(latlng).bindPopup(popupHtml(pin));
                } else {
                  markers.set(key, L.circleMarker(latlng, options).addTo(map).bindPopup(popupHtml(pin)));
                }
                if (!mapCentered) {
                  map.setView(latlng, 16);
                  mapCentered = true;
                }
              }

              async function refreshStatus() {
                const response = await fetch('/status.json');
                const status = await response.json();
                document.getElementById('status').textContent = JSON.stringify(status, null, 2);
                const activeKeys = new Set();
                for (const pin of status.detection_pins || []) {
                  const key = `detection-${pin.track_id}`;
                  activeKeys.add(key);
                  upsertMarker(key, pin, {
                    radius: 9,
                    color: '#16a34a',
                    weight: 3,
                    fillColor: '#22c55e',
                    fillOpacity: 0.8
                  });
                }
                for (const pin of status.manual_pins || []) {
                  const key = `manual-${pin.pin_id}`;
                  activeKeys.add(key);
                  upsertMarker(key, pin, {
                    radius: 8,
                    color: '#2563eb',
                    weight: 3,
                    fillColor: '#60a5fa',
                    fillOpacity: 0.85
                  });
                }
                for (const [key, marker] of markers) {
                  if (!activeKeys.has(key)) {
                    map.removeLayer(marker);
                    markers.delete(key);
                  }
                }
              }

              document.getElementById('video').addEventListener('click', async (event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                const xRatio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
                const yRatio = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
                await fetch('/camera-click', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ x_ratio: xRatio, y_ratio: yRatio })
                });
                refreshStatus();
              });
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

    @app.post("/camera-click")
    def camera_click() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            x_ratio = float(payload["x_ratio"])
            y_ratio = float(payload["y_ratio"])
        except (KeyError, TypeError, ValueError):
            abort(400, "Expected x_ratio and y_ratio.")
        if not 0.0 <= x_ratio <= 1.0 or not 0.0 <= y_ratio <= 1.0:
            abort(400, "Click ratios must be between 0 and 1.")
        state.add_camera_click({"x_ratio": x_ratio, "y_ratio": y_ratio})
        return jsonify({"ok": True})

    @app.get("/snapshot")
    def snapshot() -> Response:
        path = request.args.get("path", "")
        snapshot_path = Path(path)
        if not snapshot_path.is_file():
            abort(404)
        return send_file(snapshot_path)

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
