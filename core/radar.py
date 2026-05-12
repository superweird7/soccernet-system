from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RadarRenderer:
    TEAM_A_COLOR = (204, 153, 0)
    TEAM_B_COLOR = (255, 255, 255)
    TEAM_B_GK_COLOR = (128, 0, 128)
    REFEREE_COLOR = (0, 0, 0)
    BALL_COLOR = (0, 255, 255)
    UNKNOWN_COLOR = (128, 128, 128)
    OVERLAY_RED = (0, 0, 220)

    def __init__(self, pitch_template_path: str | Path, config: dict) -> None:
        self.pitch_template_path = Path(pitch_template_path)
        self.config = dict(config)
        self.pitch_width_cm = float(config.get("pitch_width_cm", 10500))
        self.pitch_height_cm = float(config.get("pitch_height_cm", 6800))
        self.radar_width = int(config.get("radar_width", 800))
        self.radar_height = int(config.get("radar_height", 520))
        self.show_tracker_ids = bool(config.get("show_tracker_ids", True))
        self.smoothing_frames = max(1, int(config.get("position_smoothing_frames", 3)))
        self.team_a_color = self._color("team_a_color_bgr", self.TEAM_A_COLOR)
        self.team_a_gk_color = self._color("team_a_gk_color_bgr", self.team_a_color)
        self.team_b_color = self._color("team_b_color_bgr", self.TEAM_B_COLOR)
        self.team_b_gk_color = self._color("team_b_gk_color_bgr", self.TEAM_B_GK_COLOR)
        self.referee_color = self._color("referee_color_bgr", self.REFEREE_COLOR)

        template = cv2.imread(str(self.pitch_template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(f"Could not load pitch template: {self.pitch_template_path}")
        if template.shape[:2] != (self.radar_height, self.radar_width):
            template = cv2.resize(template, (self.radar_width, self.radar_height))
        self._template = template
        self._position_history: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_frames)
        )

    def render(
        self,
        players: list[Any],
        ball_pos: Any | None = None,
        calibrated: bool = True,
    ) -> np.ndarray:
        radar = self._template.copy()

        for player in players:
            pitch_pos = self._extract_pitch_position(player)
            if pitch_pos is None:
                continue
            track_id = self._extract_track_id(player)
            if track_id is not None:
                pitch_pos = self._smooth_position(track_id, pitch_pos)
            pixel = self._pitch_to_pixel(pitch_pos)
            self._draw_player_marker(radar, pixel, player)
            if self.show_tracker_ids and track_id is not None:
                self._draw_tracker_id(radar, pixel, track_id)

        if ball_pos is not None:
            ball_pitch_pos = self._extract_ball_position(ball_pos)
            if ball_pitch_pos is not None:
                self._draw_circle(radar, self._pitch_to_pixel(ball_pitch_pos), 5, self.BALL_COLOR)

        if not calibrated:
            self._draw_calibration_overlay(radar)

        return radar

    def _smooth_position(self, track_id: int, position: np.ndarray) -> np.ndarray:
        history = self._position_history[track_id]
        history.append(position)
        return np.mean(np.asarray(history, dtype=np.float64), axis=0)

    def _pitch_to_pixel(self, position_cm: np.ndarray) -> tuple[int, int]:
        x_cm = float(np.clip(position_cm[0], 0.0, self.pitch_width_cm))
        y_cm = float(np.clip(position_cm[1], 0.0, self.pitch_height_cm))
        x_px = int(round((x_cm / self.pitch_width_cm) * self.radar_width))
        y_px = int(round((y_cm / self.pitch_height_cm) * self.radar_height))
        return (
            int(np.clip(x_px, 0, self.radar_width - 1)),
            int(np.clip(y_px, 0, self.radar_height - 1)),
        )

    def _draw_player_marker(self, radar: np.ndarray, pixel: tuple[int, int], player: Any) -> None:
        color, radius, shape = self._marker_style(player)
        if shape == "diamond":
            self._draw_diamond(radar, pixel, radius, color)
        else:
            self._draw_circle(radar, pixel, radius, color)

    def _marker_style(self, player: Any) -> tuple[tuple[int, int, int], int, str]:
        team_label = str(self._get_value(player, "team_label", "unknown") or "unknown")
        role_label = str(self._get_value(player, "role_label", "player") or "player")

        if team_label == "referee" or role_label == "referee":
            return self.referee_color, 8, "circle"

        if team_label in {"team_a", "team_a_gk"}:
            shape = "diamond" if team_label == "team_a_gk" or role_label == "goalkeeper" else "circle"
            radius = 10 if shape == "diamond" else 8
            color = self.team_a_gk_color if shape == "diamond" else self.team_a_color
            return color, radius, shape

        if team_label in {"team_b", "team_b_gk"}:
            shape = "diamond" if team_label == "team_b_gk" or role_label == "goalkeeper" else "circle"
            radius = 10 if shape == "diamond" else 8
            color = self.team_b_gk_color if shape == "diamond" else self.team_b_color
            return color, radius, shape

        return self.UNKNOWN_COLOR, 6, "circle"

    @staticmethod
    def _draw_circle(
        image: np.ndarray,
        pixel: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cv2.circle(image, pixel, radius, color, thickness=-1, lineType=cv2.LINE_8)

    @staticmethod
    def _draw_diamond(
        image: np.ndarray,
        pixel: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        x, y = pixel
        points = np.asarray(
            [[x, y - radius], [x + radius, y], [x, y + radius], [x - radius, y]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, points, color, lineType=cv2.LINE_8)

    @staticmethod
    def _draw_tracker_id(image: np.ndarray, pixel: tuple[int, int], track_id: int) -> None:
        x, y = pixel
        origin = (x + 12, y + 4)
        cv2.putText(
            image,
            str(track_id),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(track_id),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_calibration_overlay(self, radar: np.ndarray) -> None:
        overlay = np.full_like(radar, self.OVERLAY_RED)
        cv2.addWeighted(overlay, 0.42, radar, 0.58, 0.0, dst=radar)
        text = "Calibration Unavailable"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, (self.radar_width - text_size[0]) // 2)
        y = max(text_size[1] + 8, (self.radar_height + text_size[1]) // 2)
        cv2.putText(radar, text, (x, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(radar, text, (x, y), font, scale, self.OVERLAY_RED, thickness, cv2.LINE_AA)

    def _extract_pitch_position(self, player: Any) -> np.ndarray | None:
        x = self._get_value(player, "pitch_x", None)
        y = self._get_value(player, "pitch_y", None)
        if x is not None and y is not None:
            return np.asarray([float(x), float(y)], dtype=np.float64)

        for key in ("pitch_position", "pitch_coords", "position_cm"):
            value = self._get_value(player, key, None)
            if value is not None:
                return self._coerce_position(value)
        return None

    def _extract_ball_position(self, ball_pos: Any) -> np.ndarray | None:
        if isinstance(ball_pos, dict) or hasattr(ball_pos, "__dict__"):
            return self._extract_pitch_position(ball_pos)
        return self._coerce_position(ball_pos)

    @staticmethod
    def _coerce_position(value: Any) -> np.ndarray | None:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size < 2:
            return None
        return array[:2]

    def _extract_track_id(self, player: Any) -> int | None:
        track_id = self._get_value(player, "track_id", None)
        if track_id is None:
            return None
        return int(track_id)

    def _color(self, config_key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        values = self.config.get(config_key, default)
        return tuple(int(np.clip(value, 0, 255)) for value in values)

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
