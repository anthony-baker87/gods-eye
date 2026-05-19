from __future__ import annotations

from src.config import SuppressionZoneConfig
from src.detector.base import Detection


def filter_suppression_zones(
    detections: list[Detection],
    zones: list[SuppressionZoneConfig],
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    if not zones:
        return detections
    return [
        detection
        for detection in detections
        if not any(_center_inside_zone(detection.bbox, zone, frame_width, frame_height) for zone in zones)
    ]


def _center_inside_zone(
    bbox: tuple[int, int, int, int],
    zone: SuppressionZoneConfig,
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    zone_x1 = zone.x1 * frame_width
    zone_y1 = zone.y1 * frame_height
    zone_x2 = zone.x2 * frame_width
    zone_y2 = zone.y2 * frame_height
    return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2
