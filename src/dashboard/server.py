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
            <style>
              body { margin: 0; background: #101418; color: #eef2f6; font-family: system-ui, sans-serif; }
              header { padding: 14px 18px; background: #171d24; display: flex; justify-content: space-between; }
              main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; padding: 16px; }
              img { width: 100%; height: auto; background: #050608; }
              pre { white-space: pre-wrap; background: #171d24; padding: 12px; border-radius: 6px; }
              @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
            </style>
          </head>
          <body>
            <header><strong>Drone Tracker</strong><span>Local onboard dashboard</span></header>
            <main>
              <img src="/video.mjpg" alt="Live tracking stream">
              <pre id="status">Loading...</pre>
            </main>
            <script>
              async function refreshStatus() {
                const response = await fetch('/status.json');
                document.getElementById('status').textContent = JSON.stringify(await response.json(), null, 2);
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

