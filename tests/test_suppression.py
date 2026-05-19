from src.config import SuppressionZoneConfig
from src.detector.base import Detection
from src.suppression import filter_suppression_zones


def test_filter_suppression_zones_removes_detection_centered_inside_zone() -> None:
    detections = [
        Detection((10, 10, 30, 30), 0.9, label="human"),
        Detection((80, 80, 100, 100), 0.9, label="human"),
    ]
    zones = [SuppressionZoneConfig(name="corner", x1=0.0, y1=0.0, x2=0.5, y2=0.5)]

    filtered = filter_suppression_zones(detections, zones, frame_width=100, frame_height=100)

    assert filtered == [detections[1]]
