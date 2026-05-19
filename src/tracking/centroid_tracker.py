from __future__ import annotations

from dataclasses import dataclass

from src.detector.base import Detection


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    confidence: float
    smoothed_confidence: float
    label: str = "person"
    lost_frames: int = 0
    age: int = 1
    hits: int = 1
    confirmed: bool = False
    first_confirmed_frame: int | None = None


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class CentroidTracker:
    def __init__(
        self,
        max_lost_frames: int = 12,
        max_distance: float = 90.0,
        confirmation_frames: int = 3,
        min_confirmed_confidence: float = 0.7,
        confidence_smoothing: float = 0.35,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.max_distance = max_distance
        self.confirmation_frames = confirmation_frames
        self.min_confirmed_confidence = min_confirmed_confidence
        self.confidence_smoothing = confidence_smoothing
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections: list[Detection], frame_number: int = 0) -> list[Track]:
        if not detections:
            self._mark_all_lost()
            return self.tracks

        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())
        candidate_pairs: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                candidate_pairs.append((_distance(track.centroid, _centroid(detection.bbox)), track_id, detection_index))

        for distance, track_id, detection_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if distance > self.max_distance:
                continue
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            detection = detections[detection_index]
            existing = self._tracks[track_id]
            smoothed_confidence = self._smooth(existing.smoothed_confidence, detection.confidence)
            hits = existing.hits + 1
            confirmed = existing.confirmed or (
                hits >= self.confirmation_frames and smoothed_confidence >= self.min_confirmed_confidence
            )
            first_confirmed_frame = existing.first_confirmed_frame
            if confirmed and first_confirmed_frame is None:
                first_confirmed_frame = frame_number
            self._tracks[track_id] = Track(
                track_id=track_id,
                bbox=detection.bbox,
                centroid=_centroid(detection.bbox),
                confidence=detection.confidence,
                smoothed_confidence=smoothed_confidence,
                label=detection.label,
                lost_frames=0,
                age=existing.age + 1,
                hits=hits,
                confirmed=confirmed,
                first_confirmed_frame=first_confirmed_frame,
            )
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.lost_frames += 1
            if track.lost_frames > self.max_lost_frames:
                del self._tracks[track_id]

        for detection_index in sorted(unmatched_detections):
            detection = detections[detection_index]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(
                track_id=track_id,
                bbox=detection.bbox,
                centroid=_centroid(detection.bbox),
                confidence=detection.confidence,
                smoothed_confidence=detection.confidence,
                label=detection.label,
                confirmed=self.confirmation_frames <= 1 and detection.confidence >= self.min_confirmed_confidence,
                first_confirmed_frame=frame_number
                if self.confirmation_frames <= 1 and detection.confidence >= self.min_confirmed_confidence
                else None,
            )

        return self.tracks

    def _smooth(self, previous: float, current: float) -> float:
        alpha = self.confidence_smoothing
        return previous * (1.0 - alpha) + current * alpha

    def _mark_all_lost(self) -> None:
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            track.lost_frames += 1
            if track.lost_frames > self.max_lost_frames:
                del self._tracks[track_id]
