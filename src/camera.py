from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from src.config import CameraConfig, HailoConfig

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
        except Exception as exc:
            raise CameraError(
                "Picamera2 could not be imported. This often happens when apt-installed "
                "Picamera2 packages see a pip-installed numpy with an incompatible binary ABI. "
                "Use camera.source: rpicam, or rebuild the virtualenv with system packages."
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


class RpicamCameraSource:
    """Camera source backed by rpicam-vid MJPEG output."""

    def __init__(self, config: CameraConfig, hailo_config: HailoConfig | None = None) -> None:
        self.config = config
        self.hailo_config = hailo_config
        self._process: subprocess.Popen[bytes] | None = None
        self._frame_number = 0
        self._buffer = bytearray()

    def start(self) -> None:
        executable = shutil.which("rpicam-vid")
        if executable is None:
            raise CameraError("rpicam-vid is not installed or not on PATH.")

        command = [
            executable,
            "--timeout",
            "0",
            "--codec",
            "mjpeg",
            "--width",
            str(self.config.width),
            "--height",
            str(self.config.height),
            "--framerate",
            str(self.config.fps),
            "--nopreview",
            "--output",
            "-",
        ]
        if self.hailo_config is not None and self.hailo_config.post_process_file:
            command.extend(
                [
                    "--post-process-file",
                    self.hailo_config.post_process_file,
                    "--lores-width",
                    str(self.hailo_config.lores_width),
                    "--lores-height",
                    str(self.hailo_config.lores_height),
                ]
            )
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            raise CameraError(f"Unable to start rpicam-vid: {exc}") from exc

    def read(self) -> CameraFrame:
        if self._process is None or self._process.stdout is None:
            raise CameraError("rpicam camera has not been started.")
        while True:
            end = self._buffer.find(b"\xff\xd9")
            if end != -1:
                jpeg = bytes(self._buffer[: end + 2])
                del self._buffer[: end + 2]
                start = jpeg.find(b"\xff\xd8")
                if start > 0:
                    jpeg = jpeg[start:]
                frame = self._decode_jpeg(jpeg)
                self._frame_number += 1
                return CameraFrame(frame=frame, frame_number=self._frame_number, timestamp=time.time())

            chunk = self._process.stdout.read(4096)
            if not chunk:
                raise CameraError("rpicam-vid stopped before a complete frame was received.")
            self._buffer.extend(chunk)

    def _decode_jpeg(self, jpeg: bytes) -> np.ndarray:
        import cv2

        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise CameraError("Unable to decode frame from rpicam-vid MJPEG stream.")
        return frame

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
        self._process = None


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


def create_camera(
    config: CameraConfig,
    allow_synthetic: bool = True,
    hailo_config: HailoConfig | None = None,
) -> PiCameraSource | RpicamCameraSource | SyntheticCameraSource:
    source_order = {
        "auto": [PiCameraSource, RpicamCameraSource],
        "picamera2": [PiCameraSource],
        "rpicam": [RpicamCameraSource],
        "synthetic": [SyntheticCameraSource],
    }[config.source]

    last_error: CameraError | None = None
    for source_type in source_order:
        if source_type is RpicamCameraSource:
            camera = source_type(config, hailo_config)
        else:
            camera = source_type(config)
        try:
            camera.start()
            LOGGER.info("Using camera source: %s", config.source if config.source != "auto" else source_type.__name__)
            return camera
        except CameraError as exc:
            last_error = exc
            LOGGER.warning("%s unavailable: %s", source_type.__name__, exc)

    if allow_synthetic:
        LOGGER.warning("Real camera unavailable; falling back to synthetic frames. Last error: %s", last_error)
        synthetic = SyntheticCameraSource(config)
        synthetic.start()
        return synthetic

    if last_error is not None:
        raise last_error
    raise CameraError("No camera sources are configured.")
