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
