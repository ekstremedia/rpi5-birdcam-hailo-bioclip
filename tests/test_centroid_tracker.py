"""Tests for CentroidTracker — bird arrival/departure tracking."""


def _det(x_min, y_min, x_max, y_max, confidence=0.9):
    """Helper to build a detection dict."""
    return {
        "x_min": x_min, "y_min": y_min,
        "x_max": x_max, "y_max": y_max,
        "confidence": confidence,
    }


class TestRegistration:
    def test_first_detection_creates_track(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        assert len(tracker.objects) == 1
        assert 0 in tracker.objects

    def test_arrival_event_on_new_bird(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        assert tracker.arrivals == [0]
        assert tracker.total_arrivals == 1

    def test_two_simultaneous_birds(self, tracker):
        tracker.update([
            _det(100, 100, 200, 200),
            _det(500, 500, 600, 600),
        ])
        assert len(tracker.objects) == 2
        assert tracker.total_arrivals == 2


class TestMatching:
    def test_same_position_keeps_track(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        tracker.update([_det(105, 105, 205, 205)])
        assert len(tracker.objects) == 1
        assert 0 in tracker.objects
        assert tracker.arrivals == []  # no new arrival on second frame

    def test_far_away_detection_is_new_track(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        # Second detection far away (>max_distance=100)
        tracker.update([
            _det(100, 100, 200, 200),
            _det(500, 500, 600, 600),
        ])
        assert len(tracker.objects) == 2
        assert tracker.arrivals == [1]  # second bird arrived

    def test_confidence_keeps_best(self, tracker):
        tracker.update([_det(100, 100, 200, 200, confidence=0.6)])
        tracker.update([_det(105, 105, 205, 205, confidence=0.9)])
        assert tracker.confidences[0] == 0.9

    def test_confidence_does_not_decrease(self, tracker):
        tracker.update([_det(100, 100, 200, 200, confidence=0.9)])
        tracker.update([_det(105, 105, 205, 205, confidence=0.5)])
        assert tracker.confidences[0] == 0.9


class TestDisappearance:
    def test_departure_after_max_disappeared(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        assert tracker.total_arrivals == 1

        # Bird disappears for max_disappeared+1 frames
        for _ in range(4):  # max_disappeared=3, so 4 empty frames triggers deregister
            tracker.update([])
        assert len(tracker.objects) == 0
        assert 0 in tracker.departures  # last frame triggered departure

    def test_no_departure_within_threshold(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        # Disappear for just 2 frames (under max_disappeared=3)
        tracker.update([])
        tracker.update([])
        assert len(tracker.objects) == 1
        assert tracker.departures == []

    def test_reappear_resets_disappeared_count(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        tracker.update([])
        tracker.update([])
        # Bird reappears before max_disappeared
        tracker.update([_det(105, 105, 205, 205)])
        assert len(tracker.objects) == 1
        assert tracker.disappeared[0] == 0


class TestEvents:
    def test_arrivals_departures_reset_each_frame(self, tracker):
        tracker.update([_det(100, 100, 200, 200)])
        assert tracker.arrivals == [0]
        tracker.update([_det(105, 105, 205, 205)])
        assert tracker.arrivals == []
        assert tracker.departures == []

    def test_empty_update_with_no_tracks(self, tracker):
        tracker.update([])
        assert len(tracker.objects) == 0
        assert tracker.arrivals == []
        assert tracker.departures == []
