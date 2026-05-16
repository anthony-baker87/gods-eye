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

