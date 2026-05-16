from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import DetectionConfig
from src.detector.base import Detection, filter_person_detections


class HailoDetector:
    """Thin adapter for Raspberry Pi AI Kit/Hailo workflows.

    The Raspberry Pi Hailo examples evolve quickly, so this class keeps hardware-specific
    imports behind a small boundary. Install the official Hailo Raspberry Pi software and
    set detection.hailo.model_path to the HEF/model used by your chosen pipeline.
    """

    backend_name = "hailo"

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model_path = Path(config.hailo.model_path).expanduser() if config.hailo.model_path else None
        try:
            import hailo_platform  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Hailo backend requested, but hailo_platform is not importable. "
                "Install the Raspberry Pi AI Kit/Hailo runtime or run with --backend mock."
            ) from exc

        self._hailo_platform: Any = hailo_platform
        if self.model_path and not self.model_path.exists():
            raise FileNotFoundError(f"Hailo model path does not exist: {self.model_path}")

        raise NotImplementedError(
            "Hailo runtime is available, but this prototype adapter still needs the exact "
            "post-processing code for your chosen Hailo Raspberry Pi example/model. Use "
            "--backend mock for development, or wire this class to your Hailo detection pipeline."
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raw_detections: list[Detection] = []
        return filter_person_detections(
            raw_detections,
            confidence_threshold=self.config.confidence_threshold,
            person_class_id=self.config.person_class_id,
        )


class CpuDetector:
    """CPU fallback placeholder that currently delegates to OpenCV HOG when available."""

    backend_name = "cpu"

    def __init__(self, confidence_threshold: float = 0.45) -> None:
        import cv2

        self.confidence_threshold = confidence_threshold
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(max(0.0, min(1.0, weight)))
            if confidence >= self.confidence_threshold:
                detections.append(Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=confidence))
        return detections

