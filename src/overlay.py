from __future__ import annotations

import cv2
import numpy as np

from src.tracking.centroid_tracker import Track


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    fps: float,
    inference_ms: float,
    detector_backend: str,
) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = (80, 220, 120) if track.lost_frames == 0 else (80, 160, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id} {track.confidence:.2f}"
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (x1 + 150, y1), color, -1)
        cv2.putText(output, label, (x1 + 6, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 15, 12), 2)

    stats = f"FPS {fps:.1f} | infer {inference_ms:.1f} ms | tracks {len(tracks)} | {detector_backend}"
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 650), 48), (0, 0, 0), -1)
    cv2.putText(output, stats, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
    return output

