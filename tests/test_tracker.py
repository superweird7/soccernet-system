import sys
import types

import numpy as np

from core.detector import Detection


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.confidence = np.asarray(confidence, dtype=float)
        self.class_id = np.asarray(class_id, dtype=int)
        self.tracker_id = None

    def __len__(self):
        return len(self.xyxy)


class FakeByteTrack:
    calls = []
    next_ids = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeByteTrack.calls.append(("init", kwargs))

    def update_with_detections(self, detections):
        FakeByteTrack.calls.append(
            (
                "update",
                {
                    "xyxy": detections.xyxy.copy(),
                    "confidence": detections.confidence.copy(),
                    "class_id": detections.class_id.copy(),
                },
            )
        )
        ids = FakeByteTrack.next_ids.pop(0)
        detections.tracker_id = np.asarray(ids, dtype=int)
        return detections


def install_fake_supervision(monkeypatch):
    FakeByteTrack.calls = []
    FakeByteTrack.next_ids = []
    fake_supervision = types.SimpleNamespace(
        ByteTrack=FakeByteTrack,
        Detections=FakeDetections,
    )
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)


def test_tracker_configures_bytetrack_with_max_gap(monkeypatch):
    install_fake_supervision(monkeypatch)

    from core.tracker import PlayerTracker

    tracker = PlayerTracker(max_gap=30)

    assert tracker.max_gap == 30
    assert FakeByteTrack.calls[0][0] == "init"
    assert FakeByteTrack.calls[0][1]["lost_track_buffer"] == 30


def test_tracker_updates_player_detections_and_returns_integer_ids(monkeypatch):
    install_fake_supervision(monkeypatch)
    FakeByteTrack.next_ids = [[7, 8]]

    from core.tracker import PlayerTracker

    tracker = PlayerTracker(max_gap=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = [
        Detection((10, 20, 30, 60), 1, 0.81, "player"),
        Detection((50, 60, 90, 130), 0, 0.44, "ball"),
        Detection((100, 110, 150, 190), 1, 0.73, "player"),
    ]

    tracked = tracker.update(detections, frame)

    assert [player.track_id for player in tracked] == [7, 8]
    assert all(isinstance(player.track_id, int) for player in tracked)
    assert [player.bbox for player in tracked] == [
        (10.0, 20.0, 30.0, 60.0),
        (100.0, 110.0, 150.0, 190.0),
    ]
    assert [player.class_id for player in tracked] == [1, 1]
    assert [player.confidence for player in tracked] == [0.81, 0.73]
    assert [player.team_label for player in tracked] == ["unknown", "unknown"]
    assert [player.role_label for player in tracked] == ["unknown", "unknown"]
    assert FakeByteTrack.calls[-1][1]["class_id"].tolist() == [1, 1]


def test_tracker_returns_empty_list_without_player_detections(monkeypatch):
    install_fake_supervision(monkeypatch)

    from core.tracker import PlayerTracker

    tracker = PlayerTracker(max_gap=30)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = [Detection((50, 60, 90, 130), 0, 0.44, "ball")]

    assert tracker.update(detections, frame) == []
    assert FakeByteTrack.calls == [("init", {"lost_track_buffer": 30})]


def test_tracked_player_dataclass_import_contract():
    from core.tracker import TrackedPlayer

    player = TrackedPlayer(
        track_id=12,
        bbox=(1.0, 2.0, 3.0, 4.0),
        class_id=1,
        confidence=0.95,
        team_label="unknown",
        role_label="unknown",
    )

    assert player.track_id == 12
    assert player.bbox == (1.0, 2.0, 3.0, 4.0)
    assert player.class_id == 1
    assert player.confidence == 0.95
    assert player.team_label == "unknown"
    assert player.role_label == "unknown"
