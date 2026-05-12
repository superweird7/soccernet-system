from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from core.detector import Detection, FOOTBALL_PLAYER_CLASS


@dataclass(frozen=True)
class TrackedPlayer:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    team_label: str = "unknown"
    role_label: str = "unknown"


class PlayerTracker:
    def __init__(self, max_gap: int = 30):
        import supervision as sv

        self.max_gap = int(max_gap)
        self._sv = sv
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `ByteTrack` was deprecated*",
                category=FutureWarning,
            )
            self._tracker = sv.ByteTrack(lost_track_buffer=self.max_gap)

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackedPlayer]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray")

        player_detections = [
            detection
            for detection in detections
            if detection.class_id == FOOTBALL_PLAYER_CLASS
        ]
        if not player_detections:
            return []

        sv_detections = self._to_supervision_detections(player_detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            return []

        return [
            TrackedPlayer(
                track_id=int(track_id),
                bbox=tuple(float(value) for value in bbox),
                class_id=int(class_id),
                confidence=float(confidence),
            )
            for bbox, class_id, confidence, track_id in zip(
                tracked.xyxy,
                tracked.class_id,
                tracked.confidence,
                tracker_ids,
            )
            if track_id is not None and int(track_id) >= 0
        ]

    def _to_supervision_detections(self, detections: Sequence[Detection]):
        return self._sv.Detections(
            xyxy=np.asarray([detection.bbox for detection in detections], dtype=float),
            confidence=np.asarray(
                [detection.confidence for detection in detections],
                dtype=float,
            ),
            class_id=np.asarray([detection.class_id for detection in detections], dtype=int),
        )
