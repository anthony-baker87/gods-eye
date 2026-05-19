from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import SnapshotsConfig
from src.tracking import Track


class SnapshotWriter:
    def __init__(self, config: SnapshotsConfig) -> None:
        self.enabled = config.enabled
        self.path = Path(config.path).expanduser().resolve()
        self.jpeg_quality = config.jpeg_quality
        self._saved_track_ids: set[int] = set()
        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    def save_once(
        self,
        frame: np.ndarray,
        track: Track,
        frame_number: int,
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        if not self.enabled or track.track_id in self._saved_track_ids:
            return None
        self._saved_track_ids.add(track.track_id)

        timestamp = int(time.time())
        stem = f"track-{track.track_id}-frame-{frame_number}-{timestamp}"
        full_path = self.path / f"{stem}.jpg"
        crop_path = self.path / f"{stem}-crop.jpg"
        metadata_path = self.path / f"{stem}.json"

        annotated = frame.copy()
        x1, y1, x2, y2 = _clamp_bbox(track.bbox, frame.shape[1], frame.shape[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 120), 2)
        cv2.putText(
            annotated,
            f"ID {track.track_id} {track.smoothed_confidence:.2f}",
            (x1 + 6, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 220, 120),
            2,
        )
        cv2.imwrite(str(full_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        crop = frame[y1:y2, x1:x2]
        if crop.size:
            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        payload = {
            **metadata,
            "track_id": track.track_id,
            "bbox": list(track.bbox),
            "confidence": track.confidence,
            "smoothed_confidence": track.smoothed_confidence,
            "label": track.label,
            "frame_number": frame_number,
            "full_image_path": str(full_path),
            "crop_image_path": str(crop_path) if crop.size else None,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "full_image_path": str(full_path),
            "crop_image_path": str(crop_path) if crop.size else "",
            "metadata_path": str(metadata_path),
        }

    def save_manual_click(
        self,
        frame: np.ndarray,
        frame_number: int,
        pixel: tuple[int, int],
        metadata: dict[str, Any],
    ) -> dict[str, str] | None:
        if not self.enabled:
            return None

        timestamp = int(time.time())
        stem = f"manual-click-frame-{frame_number}-{timestamp}"
        full_path = self.path / f"{stem}.jpg"
        metadata_path = self.path / f"{stem}.json"

        annotated = frame.copy()
        x, y = pixel
        cv2.drawMarker(
            annotated,
            (x, y),
            (255, 180, 80),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
        )
        cv2.putText(
            annotated,
            "manual click",
            (max(0, x + 10), max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 80),
            2,
        )
        cv2.imwrite(str(full_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        payload = {
            **metadata,
            "pixel": [x, y],
            "frame_number": frame_number,
            "full_image_path": str(full_path),
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "full_image_path": str(full_path),
            "metadata_path": str(metadata_path),
        }


def _clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )
