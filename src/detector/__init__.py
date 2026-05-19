from __future__ import annotations

from src.detector.base import Detection, Detector


def create_detector(config: object) -> Detector:
    from src.detector.hailo_detector import CpuDetector, HailoDetector, RpicamHailoUdpDetector
    from src.detector.mock_detector import MockDetector

    backend = getattr(config, "backend")
    if backend == "mock":
        return MockDetector()
    if backend == "cpu":
        return CpuDetector(
            confidence_threshold=getattr(config, "confidence_threshold"),
            enable_full_body=getattr(config, "cpu_full_body", False),
        )
    if backend == "hailo":
        return HailoDetector(config)
    if backend == "rpicam_hailo":
        return RpicamHailoUdpDetector(config)
    raise ValueError(f"Unknown detector backend: {backend}")


__all__ = ["Detection", "Detector", "create_detector"]
