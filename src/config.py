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


@dataclass(slots=True)
class HailoConfig:
    model_path: str | None = None
    labels_path: str | None = None


@dataclass(slots=True)
class DetectionConfig:
    backend: str = "mock"
    confidence_threshold: float = 0.45
    person_class_id: int = 0
    hailo: HailoConfig = field(default_factory=HailoConfig)


@dataclass(slots=True)
class TrackingConfig:
    max_lost_frames: int = 12
    max_distance: float = 90.0


@dataclass(slots=True)
class DashboardConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 80


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = "logs/tracks.jsonl"


@dataclass(slots=True)
class OutputConfig:
    record: bool = False
    path: str = "output/annotated.mp4"


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


VALID_BACKENDS = {"hailo", "cpu", "mock"}


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
    logging_raw = _section(raw, "logging")
    output_raw = _section(raw, "output")
    hailo_raw = _section(detection_raw, "hailo")

    backend = str(detection_raw.get("backend", "mock")).lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unsupported detector backend '{backend}'. Use one of {sorted(VALID_BACKENDS)}.")

    config = AppConfig(
        camera=CameraConfig(
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
            fps=int(camera_raw.get("fps", 30)),
        ),
        detection=DetectionConfig(
            backend=backend,
            confidence_threshold=float(detection_raw.get("confidence_threshold", 0.45)),
            person_class_id=int(detection_raw.get("person_class_id", 0)),
            hailo=HailoConfig(
                model_path=hailo_raw.get("model_path"),
                labels_path=hailo_raw.get("labels_path"),
            ),
        ),
        tracking=TrackingConfig(
            max_lost_frames=int(tracking_raw.get("max_lost_frames", 12)),
            max_distance=float(tracking_raw.get("max_distance", 90.0)),
        ),
        dashboard=DashboardConfig(
            enabled=bool(dashboard_raw.get("enabled", True)),
            host=str(dashboard_raw.get("host", "0.0.0.0")),
            port=int(dashboard_raw.get("port", 8080)),
            jpeg_quality=int(dashboard_raw.get("jpeg_quality", 80)),
        ),
        logging=LoggingConfig(
            enabled=bool(logging_raw.get("enabled", True)),
            path=str(logging_raw.get("path", "logs/tracks.jsonl")),
        ),
        output=OutputConfig(
            record=bool(output_raw.get("record", False)),
            path=str(output_raw.get("path", "output/annotated.mp4")),
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


def validate_config(config: AppConfig) -> None:
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("Camera width and height must be positive.")
    if config.camera.fps <= 0:
        raise ValueError("Camera FPS must be positive.")
    if not 0.0 <= config.detection.confidence_threshold <= 1.0:
        raise ValueError("Detection confidence_threshold must be between 0 and 1.")
    if config.tracking.max_lost_frames < 0:
        raise ValueError("tracking.max_lost_frames must be >= 0.")
    if config.tracking.max_distance <= 0:
        raise ValueError("tracking.max_distance must be positive.")
    if not 1 <= config.dashboard.jpeg_quality <= 100:
        raise ValueError("dashboard.jpeg_quality must be between 1 and 100.")
