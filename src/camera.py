from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from src.config import CameraConfig

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass(slots=True)
class CameraFrame:
    frame: np.ndarray
    frame_number: int
    timestamp: float


class PiCameraSource:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._picam2 = None
        self._frame_number = 0

    def start(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CameraError(
                "Picamera2 is not installed. On Raspberry Pi OS, install python3-picamera2 "
                "or use --backend mock with synthetic camera fallback."
            ) from exc

        try:
            self._picam2 = Picamera2()
            video_config = self._picam2.create_video_configuration(
                main={"size": (self.config.width, self.config.height), "format": "RGB888"},
                controls={"FrameRate": self.config.fps},
            )
            self._picam2.configure(video_config)
            self._picam2.start()
        except Exception as exc:
            raise CameraError(f"Unable to start Picamera2/libcamera capture: {exc}") from exc

    def read(self) -> CameraFrame:
        if self._picam2 is None:
            raise CameraError("Camera has not been started.")
        try:
            rgb = self._picam2.capture_array()
        except Exception as exc:
            raise CameraError(f"Camera frame capture failed: {exc}") from exc

        import cv2

        self._frame_number += 1
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return CameraFrame(frame=bgr, frame_number=self._frame_number, timestamp=time.time())

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None


class SyntheticCameraSource:
    """Development camera used when Picamera2 hardware is unavailable."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._frame_number = 0
        self._started = False

    def start(self) -> None:
        self._started = True
        LOGGER.warning("Using synthetic camera frames. Install/configure Picamera2 for real camera input.")

    def read(self) -> CameraFrame:
        if not self._started:
            raise CameraError("Synthetic camera has not been started.")
        import cv2

        self._frame_number += 1
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        frame[:] = (32, 36, 42)
        t = self._frame_number / max(1, self.config.fps)
        cv2.putText(frame, "Synthetic camera", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        cv2.putText(frame, f"Frame {self._frame_number}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 220, 255), 2)
        x = int((self.config.width - 120) * (0.5 + 0.45 * np.sin(t * 0.8)))
        cv2.rectangle(frame, (x, self.config.height // 3), (x + 80, self.config.height // 3 + 160), (70, 70, 90), -1)
        time.sleep(1.0 / max(1, self.config.fps))
        return CameraFrame(frame=frame, frame_number=self._frame_number, timestamp=time.time())

    def stop(self) -> None:
        self._started = False


def create_camera(config: CameraConfig, allow_synthetic: bool = True) -> PiCameraSource | SyntheticCameraSource:
    camera = PiCameraSource(config)
    try:
        camera.start()
        return camera
    except CameraError:
        if not allow_synthetic:
            raise
        LOGGER.exception("Real camera unavailable; falling back to synthetic frames.")
        synthetic = SyntheticCameraSource(config)
        synthetic.start()
        return synthetic

