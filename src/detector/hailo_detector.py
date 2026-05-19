from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import DetectionConfig
from src.detector.base import Detection, filter_person_detections

LOGGER = logging.getLogger(__name__)


class HailoDetector:
    """HailoRT detector for YOLO-style person models."""

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
        if self.model_path is None:
            raise ValueError("Hailo backend requires detection.hailo.model_path.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Hailo model path does not exist: {self.model_path}")

        self._hef = hailo_platform.HEF(str(self.model_path))
        self._target = hailo_platform.VDevice()
        configure_params = hailo_platform.ConfigureParams.create_from_hef(
            self._hef,
            interface=hailo_platform.HailoStreamInterface.PCIe,
        )
        self._network_group = self._target.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()
        self._input_info = self._hef.get_input_vstream_infos()[0]
        self._output_infos = self._hef.get_output_vstream_infos()
        self._input_shape = _shape_tuple(getattr(self._input_info, "shape", (640, 640, 3)))
        self._input_height, self._input_width = _infer_hw(self._input_shape)
        self._input_vstreams_params = hailo_platform.InputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=False,
            format_type=hailo_platform.FormatType.UINT8,
        )
        self._output_vstreams_params = hailo_platform.OutputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=False,
            format_type=hailo_platform.FormatType.FLOAT32,
        )
        self._infer_context = hailo_platform.InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        )
        self._infer_pipeline = self._infer_context.__enter__()
        self._activation_context = self._network_group.activate(self._network_group_params)
        self._activation_context.__enter__()
        LOGGER.info(
            "Loaded Hailo model %s with input %s and outputs %s",
            self.model_path,
            self._input_shape,
            [getattr(info, "name", "output") for info in self._output_infos],
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        original_height, original_width = frame.shape[:2]
        resized = cv2.resize(frame, (self._input_width, self._input_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_frame = np.ascontiguousarray(rgb.astype(np.uint8))
        results = self._infer_pipeline.infer(input_frame)
        raw_detections = _parse_hailo_yolo_outputs(
            results,
            frame_width=original_width,
            frame_height=original_height,
            input_width=self._input_width,
            input_height=self._input_height,
        )
        return filter_person_detections(
            raw_detections,
            confidence_threshold=self.config.confidence_threshold,
            person_class_id=self.config.person_class_id,
        )

    def close(self) -> None:
        if hasattr(self, "_activation_context"):
            self._activation_context.__exit__(None, None, None)
            del self._activation_context
        if hasattr(self, "_infer_context"):
            self._infer_context.__exit__(None, None, None)
            del self._infer_context
        if hasattr(self, "_target"):
            self._target.release()


def _parse_hailo_yolo_outputs(
    results: dict[str, Any],
    frame_width: int,
    frame_height: int,
    input_width: int,
    input_height: int,
) -> list[Detection]:
    detections: list[Detection] = []
    for value in results.values():
        detections.extend(_parse_output_value(value, frame_width, frame_height, input_width, input_height))
    return _non_max_suppression(detections, iou_threshold=0.45)


def _parse_output_value(
    value: Any,
    frame_width: int,
    frame_height: int,
    input_width: int,
    input_height: int,
) -> list[Detection]:
    array = np.asarray(value)
    if array.dtype == object:
        return _parse_object_array(array, frame_width, frame_height)
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2 and squeezed.shape[-1] >= 6:
        return _parse_detection_rows(squeezed, frame_width, frame_height, input_width, input_height)
    if squeezed.ndim == 3 and squeezed.shape[-1] >= 6:
        return _parse_detection_rows(squeezed.reshape(-1, squeezed.shape[-1]), frame_width, frame_height, input_width, input_height)
    LOGGER.debug("Unsupported Hailo output shape: %s", array.shape)
    return []


def _parse_object_array(array: np.ndarray, frame_width: int, frame_height: int) -> list[Detection]:
    detections: list[Detection] = []
    for class_id, class_detections in enumerate(array.flat):
        rows = np.asarray(class_detections)
        if rows.size == 0:
            continue
        rows = rows.reshape(-1, rows.shape[-1])
        for row in rows:
            if len(row) < 5:
                continue
            y1, x1, y2, x2, confidence = [float(item) for item in row[:5]]
            detections.append(
                Detection(
                    bbox=_normalized_bbox_to_pixels(x1, y1, x2, y2, frame_width, frame_height),
                    confidence=confidence,
                    class_id=class_id,
                    label="human" if class_id == 0 else str(class_id),
                )
            )
    return detections


def _parse_detection_rows(
    rows: np.ndarray,
    frame_width: int,
    frame_height: int,
    input_width: int,
    input_height: int,
) -> list[Detection]:
    detections: list[Detection] = []
    for row in rows:
        values = [float(item) for item in row[:6]]
        if len(values) < 6:
            continue
        x1, y1, x2, y2, confidence, class_id = values[:6]
        if confidence <= 0:
            continue
        bbox = _bbox_to_pixels(x1, y1, x2, y2, frame_width, frame_height, input_width, input_height)
        detections.append(
            Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=int(class_id),
                label="human" if int(class_id) == 0 else str(int(class_id)),
            )
        )
    return detections


def _bbox_to_pixels(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    input_width: int,
    input_height: int,
) -> tuple[int, int, int, int]:
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        return _normalized_bbox_to_pixels(x1, y1, x2, y2, frame_width, frame_height)
    scale_x = frame_width / float(input_width)
    scale_y = frame_height / float(input_height)
    return _clamp_bbox(
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
        frame_width,
        frame_height,
    )


def _normalized_bbox_to_pixels(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    return _clamp_bbox(
        int(x1 * frame_width),
        int(y1 * frame_height),
        int(x2 * frame_width),
        int(y2 * frame_height),
        frame_width,
        frame_height,
    )


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in shape)


def _infer_hw(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) >= 3:
        return int(shape[0]), int(shape[1])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return 640, 640


class CpuDetector:
    """CPU fallback using OpenCV face detection, with optional HOG full-body detection."""

    backend_name = "cpu"

    def __init__(self, confidence_threshold: float = 0.45, enable_full_body: bool = False) -> None:
        import cv2

        self.confidence_threshold = confidence_threshold
        self.enable_full_body = enable_full_body
        self.max_inference_size = 640
        self._cv2 = cv2
        self._hog = None
        if self.enable_full_body:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_dir = Path(cv2.data.haarcascades)
        self._face = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cv2 = self._cv2
        inference_frame, scale = self._resize_for_inference(frame)
        gray = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections: list[Detection] = []

        if self.enable_full_body:
            detections.extend(self._detect_hog(inference_frame))
        detections.extend(self._detect_cascade(gray, self._face, confidence=0.85))

        detections = [detection for detection in detections if detection.confidence >= self.confidence_threshold]
        detections = _non_max_suppression(detections, iou_threshold=0.35)
        if scale == 1.0:
            return detections
        return [_scale_detection(detection, scale) for detection in detections]

    def _resize_for_inference(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        largest_side = max(width, height)
        if largest_side <= self.max_inference_size:
            return frame, 1.0
        scale = self.max_inference_size / float(largest_side)
        resized = self._cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=self._cv2.INTER_AREA)
        return resized, scale

    def _detect_hog(self, frame: np.ndarray) -> list[Detection]:
        if self._hog is None:
            return []
        rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(max(0.0, min(1.0, weight)))
            if confidence >= self.confidence_threshold:
                detections.append(
                    Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=confidence, label="human")
                )
        return detections

    def _detect_cascade(
        self,
        gray: np.ndarray,
        cascade: Any,
        confidence: float,
    ) -> list[Detection]:
        if cascade.empty():
            return []
        rects = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
        return [
            Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=confidence, label="human")
            for (x, y, w, h) in rects
        ]


def _non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_iou(detection.bbox, other.bbox) < iou_threshold for other in selected):
            selected.append(detection)
    return selected


def _scale_detection(detection: Detection, scale: float) -> Detection:
    inverse = 1.0 / scale
    x1, y1, x2, y2 = detection.bbox
    return Detection(
        bbox=(int(x1 * inverse), int(y1 * inverse), int(x2 * inverse), int(y2 * inverse)),
        confidence=detection.confidence,
        class_id=detection.class_id,
        label=detection.label,
    )


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
