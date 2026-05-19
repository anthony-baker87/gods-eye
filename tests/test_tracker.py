from src.detector.base import Detection
from src.tracking import CentroidTracker


def test_tracker_preserves_id_for_nearby_detection() -> None:
    tracker = CentroidTracker(max_lost_frames=2, max_distance=50)
    first = tracker.update([Detection((10, 10, 50, 80), 0.9)])
    second = tracker.update([Detection((14, 12, 54, 82), 0.88)])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].lost_frames == 0


def test_tracker_confirms_after_repeated_detections() -> None:
    tracker = CentroidTracker(max_lost_frames=2, max_distance=50, confirmation_frames=2, min_confirmed_confidence=0.7)
    first = tracker.update([Detection((10, 10, 50, 80), 0.9)], frame_number=1)
    second = tracker.update([Detection((12, 10, 52, 80), 0.8)], frame_number=2)

    assert first[0].confirmed is False
    assert second[0].confirmed is True
    assert second[0].first_confirmed_frame == 2


def test_tracker_smooths_confidence() -> None:
    tracker = CentroidTracker(
        max_lost_frames=2,
        max_distance=50,
        confirmation_frames=10,
        confidence_smoothing=0.5,
    )
    tracker.update([Detection((10, 10, 50, 80), 1.0)])
    track = tracker.update([Detection((10, 10, 50, 80), 0.0)])[0]

    assert track.smoothed_confidence == 0.5


def test_tracker_removes_track_after_max_lost_frames() -> None:
    tracker = CentroidTracker(max_lost_frames=1, max_distance=50)
    tracker.update([Detection((10, 10, 50, 80), 0.9)])
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []


def test_tracker_assigns_new_id_for_far_detection() -> None:
    tracker = CentroidTracker(max_lost_frames=2, max_distance=25)
    first = tracker.update([Detection((10, 10, 50, 80), 0.9)])
    second = tracker.update([Detection((250, 250, 300, 340), 0.92)])

    assert first[0].track_id != second[-1].track_id
