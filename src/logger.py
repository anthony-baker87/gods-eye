from __future__ import annotations

import json
import time
from pathlib import Path

from src.tracking.centroid_tracker import Track


class TrackEventLogger:
    def __init__(self, enabled: bool, path: str) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "TrackEventLogger":
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle:
            self._handle.close()

    def log_tracks(self, frame_number: int, tracks: list[Track]) -> None:
        if not self.enabled or self._handle is None:
            return
        timestamp = time.time()
        for track in tracks:
            if track.lost_frames != 0:
                continue
            event = {
                "timestamp": timestamp,
                "frame_number": frame_number,
                "track_id": track.track_id,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "smoothed_confidence": track.smoothed_confidence,
                "confirmed": track.confirmed,
                "label": track.label,
            }
            self._handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._handle.flush()
