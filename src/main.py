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
from src.snapshots import SnapshotWriter
from src.suppression import filter_suppression_zones
from src.tracking import CentroidTracker
from src.utils import PerformanceStats

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi drone person tracker")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration.")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable local Flask dashboard.")
    parser.add_argument("--backend", choices=["hailo", "rpicam_hailo", "cpu", "mock"], help="Override detector backend.")
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
        confirmation_frames=config.tracking.confirmation_frames,
        min_confirmed_confidence=config.tracking.min_confirmed_confidence,
        confidence_smoothing=config.tracking.confidence_smoothing,
    )
    snapshot_writer = SnapshotWriter(config.snapshots)
    camera = create_camera(
        config.camera,
        allow_synthetic=config.detection.backend == "mock",
        hailo_config=config.detection.hailo if config.detection.backend == "rpicam_hailo" else None,
    )
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
    pin_ttl_seconds = 60.0
    pin_min_track_age = 3
    recent_detection_pins: dict[int, dict[str, object]] = {}
    manual_pins: list[dict[str, object]] = []
    next_manual_pin_id = 1

    try:
        with TrackEventLogger(config.logging.enabled, config.logging.path) as event_logger:
            while not stop_requested:
                camera_frame = camera.read()
                inference_start = time.perf_counter()
                detections = detector.detect(camera_frame.frame)
                detections = filter_suppression_zones(
                    detections,
                    config.detection.suppression_zones,
                    frame_width=int(camera_frame.frame.shape[1]),
                    frame_height=int(camera_frame.frame.shape[0]),
                )
                stats.inference_ms = (time.perf_counter() - inference_start) * 1000.0
                tracks = tracker.update(detections, camera_frame.frame_number)
                current_location = gps_source.read()
                stats.mark_frame()
                active_tracks = [track for track in tracks if track.lost_frames == 0]
                active_human_tracks = [
                    track for track in active_tracks if track.label == "human" and track.confirmed
                ]

                annotated = draw_overlay(
                    camera_frame.frame,
                    active_human_tracks,
                    stats.fps,
                    stats.inference_ms,
                    detector.backend_name,
                    draw_tracks=detector.backend_name != "rpicam_hailo",
                )
                event_logger.log_tracks(camera_frame.frame_number, active_human_tracks)
                snapshot_paths_by_track = {
                    track.track_id: snapshot_writer.save_once(
                        camera_frame.frame,
                        track,
                        camera_frame.frame_number,
                        {
                            "gps": None if current_location is None else current_location.as_dict(),
                            "detector_backend": detector.backend_name,
                            "event": "confirmed_human",
                        },
                    )
                    for track in active_human_tracks
                }
                detection_pins = []
                if current_location is not None:
                    for track in active_human_tracks:
                        if track.hits < pin_min_track_age:
                            continue
                        previous_pin = recent_detection_pins.get(track.track_id, {})
                        snapshot_paths = snapshot_paths_by_track.get(track.track_id) or previous_pin.get("snapshot")
                        recent_detection_pins[track.track_id] = {
                            "track_id": track.track_id,
                            "latitude": current_location.latitude,
                            "longitude": current_location.longitude,
                            "altitude_m": current_location.altitude_m,
                            "confidence": round(track.smoothed_confidence, 3),
                            "label": f"Human track {track.track_id}",
                            "timestamp": current_location.timestamp,
                            "last_seen": time.time(),
                            "source": current_location.source,
                            "snapshot": snapshot_paths,
                        }
                if dashboard and current_location is not None:
                    for click in dashboard.state.drain_camera_clicks():
                        x_ratio = float(click.get("x_ratio", 0.0))
                        y_ratio = float(click.get("y_ratio", 0.0))
                        pixel = (
                            int(max(0.0, min(1.0, x_ratio)) * (camera_frame.frame.shape[1] - 1)),
                            int(max(0.0, min(1.0, y_ratio)) * (camera_frame.frame.shape[0] - 1)),
                        )
                        pin_id = next_manual_pin_id
                        next_manual_pin_id += 1
                        snapshot_paths = snapshot_writer.save_manual_click(
                            camera_frame.frame,
                            camera_frame.frame_number,
                            pixel,
                            {
                                "gps": current_location.as_dict(),
                                "event": "manual_camera_click",
                            },
                        )
                        manual_pins.append(
                            {
                                "pin_id": pin_id,
                                "latitude": current_location.latitude,
                                "longitude": current_location.longitude,
                                "altitude_m": current_location.altitude_m,
                                "label": f"Manual camera click {pin_id}",
                                "timestamp": time.time(),
                                "source": current_location.source,
                                "pixel": [pixel[0], pixel[1]],
                                "snapshot": snapshot_paths,
                            }
                        )
                cutoff = time.time() - pin_ttl_seconds
                recent_detection_pins = {
                    track_id: pin
                    for track_id, pin in recent_detection_pins.items()
                    if float(pin.get("last_seen", 0.0)) >= cutoff
                }
                detection_pins = list(recent_detection_pins.values())

                status = {
                    "fps": round(stats.fps, 2),
                    "frame_size": [int(camera_frame.frame.shape[1]), int(camera_frame.frame.shape[0])],
                    "active_track_count": len(active_human_tracks),
                    "current_detections": [
                        {
                            "track_id": track.track_id,
                            "bbox": list(track.bbox),
                            "confidence": round(track.confidence, 3),
                            "smoothed_confidence": round(track.smoothed_confidence, 3),
                            "hits": track.hits,
                            "confirmed": track.confirmed,
                            "label": track.label,
                            "lost_frames": track.lost_frames,
                        }
                        for track in active_human_tracks
                    ],
                    "uptime": round(stats.uptime, 2),
                    "detector_backend": detector.backend_name,
                    "frame_number": camera_frame.frame_number,
                    "gps": None if current_location is None else current_location.as_dict(),
                    "detection_pins": detection_pins,
                    "manual_pins": manual_pins,
                }
                if dashboard:
                    dashboard.state.update(annotated, status)
                if writer:
                    writer.write(annotated)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted; shutting down.")
    finally:
        camera.stop()
        close_detector = getattr(detector, "close", None)
        if callable(close_detector):
            close_detector()
        if writer:
            writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
