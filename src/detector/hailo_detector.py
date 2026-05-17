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
    """CPU fallback using OpenCV person, upper-body, and face detectors."""

    backend_name = "cpu"

    def __init__(self, confidence_threshold: float = 0.45) -> None:
        import cv2

        self.confidence_threshold = confidence_threshold
        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_dir = Path(cv2.data.haarcascades)
        self._face = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
        self._upper_body = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_upperbody.xml"))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections: list[Detection] = []

        detections.extend(self._detect_hog(frame))
        detections.extend(self._detect_cascade(gray, self._upper_body, confidence=0.65, label="upper_body"))
        detections.extend(self._detect_cascade(gray, self._face, confidence=0.85, label="face"))

        detections = [detection for detection in detections if detection.confidence >= self.confidence_threshold]
        return _non_max_suppression(detections, iou_threshold=0.35)

    def _detect_hog(self, frame: np.ndarray) -> list[Detection]:
        rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(max(0.0, min(1.0, weight)))
            if confidence >= self.confidence_threshold:
                detections.append(
                    Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=confidence, label="person")
                )
        return detections

    def _detect_cascade(
        self,
        gray: np.ndarray,
        cascade: Any,
        confidence: float,
        label: str,
    ) -> list[Detection]:
        if cascade.empty():
            return []
        rects = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
        return [
            Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=confidence, label=label)
            for (x, y, w, h) in rects
        ]


def _non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_iou(detection.bbox, other.bbox) < iou_threshold for other in selected):
            selected.append(detection)
    return selected


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    if intersection_area == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection_area / float(a_area + b_area - intersection_area)
