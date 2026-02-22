"""Tests for _parse_detections — Hailo YOLOv8 output parsing."""

import numpy as np

from bird_monitor import BirdMonitor, CONFIDENCE_THRESHOLD, FRAME_WIDTH, FRAME_HEIGHT


def _make_raw_output(class_detections):
    """Build a list-of-80-arrays raw output (Hailo NMS format).

    class_detections: dict mapping class_id -> list of (y_min, x_min, y_max, x_max, conf)
    """
    output = [np.empty((0, 5)) for _ in range(80)]
    for cls_id, dets in class_detections.items():
        output[cls_id] = np.array(dets, dtype=np.float32)
    return output


# Use a standalone instance just to access _parse_detections.
# BirdMonitor.__init__ needs hardware, so we create one via __new__.
_monitor = object.__new__(BirdMonitor)


class TestEmptyOutput:
    def test_empty_list(self):
        result = _monitor._parse_detections([np.empty((0, 5)) for _ in range(80)])
        assert result == []

    def test_empty_list_literal(self):
        result = _monitor._parse_detections([])
        assert result == []


class TestSingleBird:
    def test_bird_detected(self):
        raw = _make_raw_output({
            14: [(0.1, 0.2, 0.3, 0.4, 0.85)]
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 1
        d = dets[0]
        assert d["class_id"] == 14
        assert d["class_name"] == "bird"
        assert abs(d["confidence"] - 0.85) < 1e-4

    def test_coordinates_scaled(self):
        raw = _make_raw_output({
            14: [(0.1, 0.2, 0.3, 0.4, 0.85)]
        })
        dets = _monitor._parse_detections(raw)
        d = dets[0]
        assert abs(d["x_min"] - 0.2 * FRAME_WIDTH) < 0.01
        assert abs(d["y_min"] - 0.1 * FRAME_HEIGHT) < 0.01
        assert abs(d["x_max"] - 0.4 * FRAME_WIDTH) < 0.01
        assert abs(d["y_max"] - 0.3 * FRAME_HEIGHT) < 0.01


class TestMultipleClasses:
    def test_bird_and_person(self):
        raw = _make_raw_output({
            0: [(0.1, 0.1, 0.5, 0.5, 0.9)],   # person
            14: [(0.2, 0.3, 0.4, 0.6, 0.8)],   # bird
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 2
        classes = {d["class_id"] for d in dets}
        assert classes == {0, 14}

    def test_multiple_birds(self):
        raw = _make_raw_output({
            14: [
                (0.1, 0.2, 0.3, 0.4, 0.85),
                (0.5, 0.6, 0.7, 0.8, 0.72),
            ]
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 2
        assert all(d["class_id"] == 14 for d in dets)


class TestConfidenceFiltering:
    def test_below_threshold_filtered(self):
        raw = _make_raw_output({
            14: [(0.1, 0.2, 0.3, 0.4, CONFIDENCE_THRESHOLD - 0.01)]
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 0

    def test_at_threshold_passes(self):
        raw = _make_raw_output({
            14: [(0.1, 0.2, 0.3, 0.4, CONFIDENCE_THRESHOLD)]
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 1

    def test_mixed_confidence(self):
        raw = _make_raw_output({
            14: [
                (0.1, 0.2, 0.3, 0.4, 0.9),   # above
                (0.5, 0.6, 0.7, 0.8, 0.01),   # below
            ]
        })
        dets = _monitor._parse_detections(raw)
        assert len(dets) == 1
        assert abs(dets[0]["confidence"] - 0.9) < 1e-4
