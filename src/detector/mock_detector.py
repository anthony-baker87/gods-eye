from __future__ import annotations

import math
import time

import numpy as np

from src.detector.base import Detection


class MockDetector:
    """Deterministic moving person boxes for development without camera or accelerator."""

    backend_name = "mock"

    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = confidence
        self.start_time = time.monotonic()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        elapsed = time.monotonic() - self.start_time
        box_w = max(60, width // 8)
        box_h = max(120, height // 3)
        center_x = int(width * (0.5 + 0.32 * math.sin(elapsed * 0.9)))
        center_y = int(height * (0.52 + 0.12 * math.cos(elapsed * 0.7)))
        x1 = max(0, center_x - box_w // 2)
        y1 = max(0, center_y - box_h // 2)
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)

        detections = [Detection(bbox=(x1, y1, x2, y2), confidence=self.confidence)]
        if width >= 640:
            second_x = int(width * (0.25 + 0.12 * math.sin(elapsed * 0.55 + 2.0)))
            second_y = int(height * 0.58)
            x1b = max(0, second_x - box_w // 3)
            y1b = max(0, second_y - box_h // 3)
            detections.append(
                Detection(
                    bbox=(x1b, y1b, min(width - 1, x1b + box_w // 2), min(height - 1, y1b + box_h // 2)),
                    confidence=max(0.5, self.confidence - 0.08),
                )
            )
        return detections

