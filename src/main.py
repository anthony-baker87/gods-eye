from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import cv2

from src.camera import create_camera
from src.config import load_config
from src.dashboard.server import DashboardServer
from src.detector import create_detector
from src.gps import create_gps_source
from src.logger import TrackEventLogger
from src.overlay import draw_overlay
from src.tracking import CentroidTracker
from src.utils import PerformanceStats

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi drone person tracker")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration.")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable local Flask dashboard.")
    parser.add_argument("--backend", choices=["hailo", "cpu", "mock"], help="Override detector backend.")
    parser.add_argument("--record-output", action="store_true", help="Record annotated video to output.path.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.backend:
        config.detection.backend = args.backend
    if args.no_dashboard:
        config.dashboard.enabled = False
    if args.record_output:
        config.output.record = True

    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        LOGGER.info("Received signal %s; shutting down.", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    detector = create_detector(config.detection)
    gps_source = create_gps_source(config.gps)
    tracker = CentroidTracker(
        max_lost_frames=config.tracking.max_lost_frames,
        max_distance=config.tracking.max_distance,
    )
    camera = create_camera(config.camera, allow_synthetic=config.detection.backend == "mock")
    dashboard = None
    if config.dashboard.enabled:
        dashboard = DashboardServer(config.dashboard.host, config.dashboard.port, config.dashboard.jpeg_quality)
        dashboard.start()
        LOGGER.info("Dashboard available at http://%s:%s", config.dashboard.host, config.dashboard.port)

    writer = None
    if config.output.record:
        output_path = Path(config.output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, config.camera.fps, (config.camera.width, config.camera.height))

    stats = PerformanceStats()
    stats.start()

    try:
        with TrackEventLogger(config.logging.enabled, config.logging.path) as event_logger:
            while not stop_requested:
                camera_frame = camera.read()
                inference_start = time.perf_counter()
                detections = detector.detect(camera_frame.frame)
                stats.inference_ms = (time.perf_counter() - inference_start) * 1000.0
                tracks = tracker.update(detections)
                current_location = gps_source.read()
                stats.mark_frame()

                annotated = draw_overlay(camera_frame.frame, tracks, stats.fps, stats.inference_ms, detector.backend_name)
                event_logger.log_tracks(camera_frame.frame_number, tracks)
                active_tracks = [track for track in tracks if track.lost_frames == 0]
                detection_pins = []
                if current_location is not None:
                    detection_pins = [
                        {
                            "track_id": track.track_id,
                            "latitude": current_location.latitude,
                            "longitude": current_location.longitude,
                            "altitude_m": current_location.altitude_m,
                            "confidence": round(track.confidence, 3),
                            "timestamp": current_location.timestamp,
                            "source": current_location.source,
                        }
                        for track in active_tracks
                    ]

                status = {
                    "fps": round(stats.fps, 2),
                    "frame_size": [int(camera_frame.frame.shape[1]), int(camera_frame.frame.shape[0])],
                    "active_track_count": len(active_tracks),
                    "current_detections": [
                        {
                            "track_id": track.track_id,
                            "bbox": list(track.bbox),
                            "confidence": round(track.confidence, 3),
                            "label": track.label,
                            "lost_frames": track.lost_frames,
                        }
                        for track in tracks
                    ],
                    "uptime": round(stats.uptime, 2),
                    "detector_backend": detector.backend_name,
                    "frame_number": camera_frame.frame_number,
                    "gps": None if current_location is None else current_location.as_dict(),
                    "detection_pins": detection_pins,
                }
                if dashboard:
                    dashboard.state.update(annotated, status)
                if writer:
                    writer.write(annotated)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; shutting down.")
    finally:
        camera.stop()
        if writer:
            writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
