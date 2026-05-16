from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    label: str = "person"


class Detector(Protocol):
    backend_name: str

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return person detections for one BGR frame."""


def filter_person_detections(
    detections: list[Detection],
    confidence_threshold: float,
    person_class_id: int = 0,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.class_id == person_class_id and detection.confidence >= confidence_threshold
    ]

