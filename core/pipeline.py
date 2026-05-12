from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.annotator import FrameAnnotator
from core.detector import FOOTBALL_BALL_CLASS, Detection, FootballDetector
from core.homography import BroadcastCalibrator
from core.radar import RadarRenderer
from core.team_classifier import PRTReidClassifier
from core.tracker import PlayerTracker


class AnalysisPipeline(QThread):
    frame_ready = pyqtSignal(object, object)
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        config: dict,
        detector: Any | None = None,
        tracker: Any | None = None,
        team_classifier: Any | None = None,
        homography: Any | None = None,
        annotator: Any | None = None,
        radar: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.project_root = Path(self.config.get("project_root", Path.cwd()))
        self._defer_component_setup = bool(self.config.get("defer_component_setup", False))
        self.detector = detector
        self.tracker = tracker
        self.team_classifier = team_classifier
        self.homography = homography
        self.annotator = annotator
        self.radar = radar
        if not self._defer_component_setup:
            self._ensure_components()

        self.frame_idx = 0
        self._video_path: Path | None = None
        self._image_paths: list[Path] = []
        self._capture: cv2.VideoCapture | None = None
        self._source_type: str | None = None
        self._should_stop = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def total_frames(self) -> int:
        if self._source_type == "images":
            return len(self._image_paths)
        if self._capture is not None:
            return max(0, int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        return 0

    def load_video(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Video source does not exist: {source}")

        self._release_capture()
        self._video_path = source
        self.frame_idx = 0
        self._should_stop = False
        self._pause_event.set()

        if source.is_dir():
            self._source_type = "images"
            self._image_paths = sorted(
                p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            if not self._image_paths:
                raise RuntimeError(f"No image frames found in {source}")
            return

        self._source_type = "video"
        self._image_paths = []
        self._capture = cv2.VideoCapture(str(source))
        if not self._capture.isOpened():
            self._release_capture()
            raise RuntimeError(f"Could not open video: {source}")

    def set_color_config(self, color_config: dict) -> None:
        self.config.update(color_config)
        if self.team_classifier is not None and hasattr(self.team_classifier, "update_manual_colors"):
            self.team_classifier.update_manual_colors(color_config)
        self.annotator = FrameAnnotator(self.config)
        self.radar = self._build_radar()

    def run(self) -> None:
        if self._source_type is None:
            self.error_occurred.emit("No video source loaded")
            return

        self._should_stop = False
        last_tick = time.perf_counter()
        try:
            self._ensure_components()
            while not self._should_stop and self._has_next_frame():
                self._pause_event.wait()
                if self._should_stop:
                    break

                frame_idx = self.frame_idx
                frame = self._read_frame(frame_idx)
                if frame is None:
                    break

                tick = time.perf_counter()
                fps = 1.0 / max(tick - last_tick, 1e-6)
                last_tick = tick

                self._process_frame(frame, frame_idx, fps)
                self.frame_idx = frame_idx + 1
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._should_stop = True
        self._pause_event.set()

    def seek(self, frame_idx: int) -> None:
        requested = max(0, int(frame_idx))
        frame_count = self.total_frames
        if frame_count > 0:
            requested = min(requested, frame_count - 1)
        self.frame_idx = requested
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> None:
        self._ensure_components()
        detections = self.detector.detect(frame)
        tracked_players = self.tracker.update(detections, frame)

        self.homography.update(frame, frame_idx)
        cal_status = self.homography.get_status()

        player_rows = self._classify_players(frame, tracked_players, frame_idx)
        player_rows = self._attach_pitch_positions(player_rows)
        ball = self._extract_ball(detections)
        ball_pos = self._transform_ball(ball)

        print(
            f"[pipeline] frame={frame_idx} detections={len(detections)} "
            f"tracked_players={len(tracked_players)} annotated_players={len(player_rows)}",
            flush=True,
        )

        annotated = self.annotator.annotate(frame, player_rows, ball=ball, cal_status=cal_status)
        radar = self.radar.render(
            player_rows,
            ball_pos=ball_pos,
            calibrated=cal_status != "unavailable" and self.homography.is_calibrated(),
        )

        self.frame_ready.emit(annotated, radar)
        self.stats_updated.emit(
            {
                "players_detected": len(player_rows),
                "ball_visible": ball is not None,
                "fps": float(fps),
                "cal_status": cal_status,
                "frame_idx": frame_idx,
            }
        )

    def _classify_players(self, frame: np.ndarray, tracked_players: list[Any], frame_idx: int) -> list[dict]:
        players = []
        for player in tracked_players:
            bbox = tuple(float(value) for value in self._get_value(player, "bbox"))
            track_id = int(self._get_value(player, "track_id"))
            team_label, role_label, team_confidence = self.team_classifier.predict_from_frame(
                frame,
                bbox,
                track_id,
                frame_idx,
            )
            players.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "class_id": self._get_value(player, "class_id", 1),
                    "confidence": float(self._get_value(player, "confidence", 0.0)),
                    "team_label": team_label,
                    "role_label": role_label,
                    "team_confidence": float(team_confidence),
                }
            )
        return players

    def _attach_pitch_positions(self, players: list[dict]) -> list[dict]:
        if not players:
            return []

        foot_points = np.asarray([self._foot_point(player["bbox"]) for player in players], dtype=np.float64)
        transformed = self.homography.transform_points(foot_points)
        if transformed is None:
            return players
        if transformed is not None and len(transformed) == len(players):
            candidates = [
                {**player, "pitch_x": float(pitch_point[0]), "pitch_y": float(pitch_point[1])}
                for player, pitch_point in zip(players, transformed)
            ]
        else:
            candidates = []
            for player, foot_point in zip(players, foot_points):
                point = self.homography.transform_points(foot_point.reshape(1, 2))
                if point is None or len(point) != 1:
                    continue
                candidates.append(
                    {**player, "pitch_x": float(point[0, 0]), "pitch_y": float(point[0, 1])}
                )

        return [player for player in candidates if self._inside_pipeline_bounds(player)]

    def _extract_ball(self, detections: list[Detection]) -> dict | None:
        balls = [
            detection
            for detection in detections
            if getattr(detection, "class_id", None) == FOOTBALL_BALL_CLASS
            or getattr(detection, "label", None) == "ball"
        ]
        if not balls:
            return None
        ball = max(balls, key=lambda detection: float(detection.confidence))
        return {"bbox": ball.bbox, "label": "ball", "confidence": float(ball.confidence)}

    def _transform_ball(self, ball: dict | None) -> tuple[float, float] | None:
        if ball is None:
            return None
        x1, y1, x2, y2 = ball["bbox"]
        center = np.asarray([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float64)
        transformed = self.homography.transform_points(center)
        if transformed is None or len(transformed) != 1:
            return None
        x, y = float(transformed[0, 0]), float(transformed[0, 1])
        if not self._inside_pitch_bounds(x, y):
            return None
        return x, y

    def _build_detector(self) -> FootballDetector:
        model_path = self._resolve_path(self.config.get("detector_model_path", "models/yolo11x.pt"))
        return FootballDetector(
            model_path,
            conf_player=float(self.config.get("detection_confidence_player", 0.35)),
            conf_ball=float(self.config.get("detection_confidence_ball", 0.20)),
            device=self.config.get("device", "cuda"),
            imgsz=int(self.config.get("input_resize_width", 1280)),
        )

    def _build_radar(self) -> RadarRenderer:
        return RadarRenderer(
            self._resolve_path(self.config.get("pitch_template_path", "assets/pitch_template.png")),
            self.config,
        )

    def _ensure_components(self) -> None:
        if self.detector is None:
            self.detector = self._build_detector()
        if self.tracker is None:
            self.tracker = PlayerTracker(max_gap=int(self.config.get("tracker_max_gap", 30)))
        if self.team_classifier is None:
            self.team_classifier = PRTReidClassifier(
                self.config,
                device=self.config.get("device", "cuda"),
                async_enabled=True,
            )
        if self.homography is None:
            self.homography = BroadcastCalibrator(
                self.config,
                device=self.config.get("device", "cuda"),
            )
        if self.annotator is None:
            self.annotator = FrameAnnotator(self.config)
        if self.radar is None:
            self.radar = self._build_radar()

    def _read_frame(self, frame_idx: int) -> np.ndarray | None:
        if self._source_type == "images":
            if frame_idx >= len(self._image_paths):
                return None
            return cv2.imread(str(self._image_paths[frame_idx]), cv2.IMREAD_COLOR)

        if self._capture is None:
            return None
        current = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current != frame_idx:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._capture.read()
        return frame if ok else None

    def _has_next_frame(self) -> bool:
        frame_limit = int(self.config.get("frame_limit", -1))
        if frame_limit >= 0 and self.frame_idx >= frame_limit:
            return False
        if self._source_type == "images":
            return self.frame_idx < len(self._image_paths)
        return True

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _inside_pipeline_bounds(player: dict) -> bool:
        return AnalysisPipeline._inside_pitch_bounds(float(player["pitch_x"]), float(player["pitch_y"]))

    @staticmethod
    def _inside_pitch_bounds(x: float, y: float) -> bool:
        return -200.0 <= x <= 10700.0 and -200.0 <= y <= 7000.0

    @staticmethod
    def _foot_point(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, _, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
