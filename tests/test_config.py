from pathlib import Path

import pytest

from src.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_config_defaults_and_values() -> None:
    config = load_config(FIXTURES / "minimal_config.yaml")

    assert config.camera.width == 640
    assert config.camera.height == 480
    assert config.camera.fps == 30
    assert config.detection.backend == "mock"
    assert config.detection.confidence_threshold == 0.6
    assert config.detection.cpu_full_body is False


def test_invalid_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported detector backend"):
        load_config(FIXTURES / "invalid_backend.yaml")


def test_invalid_camera_source_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_camera_source.yaml"
    config_path.write_text(
        """
camera:
  source: raspicam
detection:
  backend: mock
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera.source"):
        load_config(config_path)


def test_static_gps_requires_coordinates(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_static_gps.yaml"
    config_path.write_text(
        """
gps:
  enabled: true
  provider: static
detection:
  backend: mock
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Static GPS"):
        load_config(config_path)


def test_suppression_zone_must_be_normalized(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_suppression_zone.yaml"
    config_path.write_text(
        """
detection:
  backend: mock
  suppression_zones:
    - name: drone_body
      x1: 0.1
      y1: 0.1
      x2: 1.2
      y2: 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Suppression zones"):
        load_config(config_path)
