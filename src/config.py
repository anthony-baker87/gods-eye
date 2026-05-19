from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal dev hosts.
    yaml = None


@dataclass(slots=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    source: str = "auto"


@dataclass(slots=True)
class HailoConfig:
    model_path: str | None = None
    labels_path: str | None = None
    model_type: str = "yolo"
    post_process_file: str | None = None
    udp_host: str = "127.0.0.1"
    udp_port: int = 12347
    lores_width: int = 640
    lores_height: int = 640


@dataclass(slots=True)
class SuppressionZoneConfig:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class DetectionConfig:
    backend: str = "mock"
    confidence_threshold: float = 0.45
    person_class_id: int = 0
    cpu_full_body: bool = False
    suppression_zones: list[SuppressionZoneConfig] = field(default_factory=list)
    hailo: HailoConfig = field(default_factory=HailoConfig)


@dataclass(slots=True)
class TrackingConfig:
    max_lost_frames: int = 12
    max_distance: float = 90.0
    confirmation_frames: int = 3
    min_confirmed_confidence: float = 0.7
    confidence_smoothing: float = 0.35


@dataclass(slots=True)
class DashboardConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 80


@dataclass(slots=True)
class GpsConfig:
    enabled: bool = False
    provider: str = "static"
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    gpsd_host: str = "127.0.0.1"
    gpsd_port: int = 2947


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = "logs/tracks.jsonl"


@dataclass(slots=True)
class OutputConfig:
    record: bool = False
    path: str = "output/annotated.mp4"


@dataclass(slots=True)
class SnapshotsConfig:
    enabled: bool = True
    path: str = "output/snapshots"
    jpeg_quality: int = 90


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)


VALID_BACKENDS = {"hailo", "rpicam_hailo", "cpu", "mock"}


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = _load_yaml(handle.read())

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    camera_raw = _section(raw, "camera")
    detection_raw = _section(raw, "detection")
    tracking_raw = _section(raw, "tracking")
    dashboard_raw = _section(raw, "dashboard")
    gps_raw = _section(raw, "gps")
    logging_raw = _section(raw, "logging")
    output_raw = _section(raw, "output")
    snapshots_raw = _section(raw, "snapshots")
    hailo_raw = _section(detection_raw, "hailo")

    backend = str(detection_raw.get("backend", "mock")).lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unsupported detector backend '{backend}'. Use one of {sorted(VALID_BACKENDS)}.")

    config = AppConfig(
        camera=CameraConfig(
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
            fps=int(camera_raw.get("fps", 30)),
            source=str(camera_raw.get("source", "auto")).lower(),
        ),
        detection=DetectionConfig(
            backend=backend,
            confidence_threshold=float(detection_raw.get("confidence_threshold", 0.45)),
            person_class_id=int(detection_raw.get("person_class_id", 0)),
            cpu_full_body=bool(detection_raw.get("cpu_full_body", False)),
            suppression_zones=_parse_suppression_zones(detection_raw.get("suppression_zones", [])),
            hailo=HailoConfig(
                model_path=hailo_raw.get("model_path"),
                labels_path=hailo_raw.get("labels_path"),
                model_type=str(hailo_raw.get("model_type", "yolo")).lower(),
                post_process_file=hailo_raw.get("post_process_file"),
                udp_host=str(hailo_raw.get("udp_host", "127.0.0.1")),
                udp_port=int(hailo_raw.get("udp_port", 12347)),
                lores_width=int(hailo_raw.get("lores_width", 640)),
                lores_height=int(hailo_raw.get("lores_height", 640)),
            ),
        ),
        tracking=TrackingConfig(
            max_lost_frames=int(tracking_raw.get("max_lost_frames", 12)),
            max_distance=float(tracking_raw.get("max_distance", 90.0)),
            confirmation_frames=int(tracking_raw.get("confirmation_frames", 3)),
            min_confirmed_confidence=float(tracking_raw.get("min_confirmed_confidence", 0.7)),
            confidence_smoothing=float(tracking_raw.get("confidence_smoothing", 0.35)),
        ),
        dashboard=DashboardConfig(
            enabled=bool(dashboard_raw.get("enabled", True)),
            host=str(dashboard_raw.get("host", "0.0.0.0")),
            port=int(dashboard_raw.get("port", 8080)),
            jpeg_quality=int(dashboard_raw.get("jpeg_quality", 80)),
        ),
        gps=GpsConfig(
            enabled=bool(gps_raw.get("enabled", False)),
            provider=str(gps_raw.get("provider", "static")).lower(),
            latitude=_optional_float(gps_raw.get("latitude")),
            longitude=_optional_float(gps_raw.get("longitude")),
            altitude_m=_optional_float(gps_raw.get("altitude_m")),
            gpsd_host=str(gps_raw.get("gpsd_host", "127.0.0.1")),
            gpsd_port=int(gps_raw.get("gpsd_port", 2947)),
        ),
        logging=LoggingConfig(
            enabled=bool(logging_raw.get("enabled", True)),
            path=str(logging_raw.get("path", "logs/tracks.jsonl")),
        ),
        output=OutputConfig(
            record=bool(output_raw.get("record", False)),
            path=str(output_raw.get("path", "output/annotated.mp4")),
        ),
        snapshots=SnapshotsConfig(
            enabled=bool(snapshots_raw.get("enabled", True)),
            path=str(snapshots_raw.get("path", "output/snapshots")),
            jpeg_quality=int(snapshots_raw.get("jpeg_quality", 90)),
        ),
    )
    validate_config(config)
    return config


def _load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level.")
        return parsed
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset fallback for development machines without PyYAML.

    It supports the nested mappings and scalar values used by config.yaml. Install
    PyYAML on the Raspberry Pi for full YAML support.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML syntax on line {line_number}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation on line {line_number}: {raw_line}")
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_suppression_zones(raw: Any) -> list[SuppressionZoneConfig]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("detection.suppression_zones must be a list.")
    zones: list[SuppressionZoneConfig] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each suppression zone must be a mapping.")
        zones.append(
            SuppressionZoneConfig(
                name=str(item.get("name", f"zone_{index}")),
                x1=float(item.get("x1")),
                y1=float(item.get("y1")),
                x2=float(item.get("x2")),
                y2=float(item.get("y2")),
            )
        )
    return zones


def validate_config(config: AppConfig) -> None:
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("Camera width and height must be positive.")
    if config.camera.fps <= 0:
        raise ValueError("Camera FPS must be positive.")
    if config.camera.source not in {"auto", "picamera2", "rpicam", "synthetic"}:
        raise ValueError("camera.source must be one of: auto, picamera2, rpicam, synthetic.")
    if not 0.0 <= config.detection.confidence_threshold <= 1.0:
        raise ValueError("Detection confidence_threshold must be between 0 and 1.")
    for zone in config.detection.suppression_zones:
        if not 0.0 <= zone.x1 < zone.x2 <= 1.0 or not 0.0 <= zone.y1 < zone.y2 <= 1.0:
            raise ValueError("Suppression zones must use normalized coordinates between 0 and 1.")
    if config.tracking.max_lost_frames < 0:
        raise ValueError("tracking.max_lost_frames must be >= 0.")
    if config.tracking.max_distance <= 0:
        raise ValueError("tracking.max_distance must be positive.")
    if config.tracking.confirmation_frames <= 0:
        raise ValueError("tracking.confirmation_frames must be positive.")
    if not 0.0 <= config.tracking.min_confirmed_confidence <= 1.0:
        raise ValueError("tracking.min_confirmed_confidence must be between 0 and 1.")
    if not 0.0 <= config.tracking.confidence_smoothing <= 1.0:
        raise ValueError("tracking.confidence_smoothing must be between 0 and 1.")
    if not 1 <= config.dashboard.jpeg_quality <= 100:
        raise ValueError("dashboard.jpeg_quality must be between 1 and 100.")
    if config.gps.provider not in {"static", "gpsd"}:
        raise ValueError("gps.provider must be one of: static, gpsd.")
    if config.gps.enabled and config.gps.provider == "static":
        if config.gps.latitude is None or config.gps.longitude is None:
            raise ValueError("Static GPS requires gps.latitude and gps.longitude.")
    if config.gps.latitude is not None and not -90.0 <= config.gps.latitude <= 90.0:
        raise ValueError("gps.latitude must be between -90 and 90.")
    if config.gps.longitude is not None and not -180.0 <= config.gps.longitude <= 180.0:
        raise ValueError("gps.longitude must be between -180 and 180.")
    if config.gps.gpsd_port <= 0:
        raise ValueError("gps.gpsd_port must be positive.")
    if config.detection.hailo.udp_port <= 0:
        raise ValueError("detection.hailo.udp_port must be positive.")
    if config.detection.hailo.lores_width <= 0 or config.detection.hailo.lores_height <= 0:
        raise ValueError("detection.hailo lores dimensions must be positive.")
    if not 1 <= config.snapshots.jpeg_quality <= 100:
        raise ValueError("snapshots.jpeg_quality must be between 1 and 100.")
