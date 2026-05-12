from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FrameAnnotator:
    CAL_STATUS = {
        "tvcalib": ("CAL: TVCalib", (0, 150, 0)),
        "manual": ("CAL: Manual", (0, 165, 255)),
        "unavailable": ("CAL: None", (0, 0, 220)),
        "none": ("CAL: None", (0, 0, 220)),
    }
    BALL_COLOR = (0, 255, 255)
    UNKNOWN_COLOR = (160, 160, 160)

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.team_a_color = self._color("team_a_color_bgr", (204, 153, 0))
        self.team_b_color = self._color("team_b_color_bgr", (255, 255, 255))
        self.team_a_gk_color = self._color("team_a_gk_color_bgr", self.team_a_color)
        self.team_b_gk_color = self._color("team_b_gk_color_bgr", (128, 0, 128))
        self.referee_color = self._color("referee_color_bgr", (0, 0, 0))

    def annotate(
        self,
        frame: np.ndarray,
        players: list[Any],
        ball: Any | None = None,
        cal_status: str = "unavailable",
    ) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray")

        annotated = frame.copy()
        for player in players:
            bbox = self._extract_bbox(player)
            if bbox is None:
                continue
            color = self._player_color(player)
            x1, y1, x2, y2 = self._clamp_bbox(bbox, annotated.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            self._draw_feet_ellipse(annotated, (x1, y1, x2, y2), color)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_8)
            self._draw_player_label(annotated, (x1, y1, x2, y2), self.format_player_label(player), color)

        if ball is not None:
            ball_bbox = self._extract_bbox(ball)
            if ball_bbox is not None:
                self._draw_ball_marker(annotated, self._clamp_bbox(ball_bbox, annotated.shape))

        self._draw_calibration_status(annotated, cal_status)
        if self._has_kit_clash():
            self._draw_kit_clash_warning(annotated)
        return annotated

    def format_player_label(self, player: Any) -> str:
        team_label = str(self._get_value(player, "team_label", "unknown") or "unknown")
        role_label = str(self._get_value(player, "role_label", "player") or "player")
        track_id = self._get_value(player, "track_id", None)

        if team_label == "referee" or role_label == "referee":
            return "REF"

        if team_label in {"team_a", "team_a_gk"}:
            prefix = "A GK" if team_label == "team_a_gk" or role_label == "goalkeeper" else "A"
        elif team_label in {"team_b", "team_b_gk"}:
            prefix = "B GK" if team_label == "team_b_gk" or role_label == "goalkeeper" else "B"
        else:
            prefix = "UNK"

        return prefix if track_id is None else f"{prefix} {int(track_id)}"

    def _draw_feet_ellipse(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, _, x2, y2 = bbox
        center = ((x1 + x2) // 2, y2)
        axes = (max(8, (x2 - x1) // 2), max(3, (x2 - x1) // 7))
        overlay = image.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1, lineType=cv2.LINE_8)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)

    def _draw_player_label(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, _, _ = bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
        label_x = x1
        label_y = max(0, y1 - text_size[1] - 8)
        background_top_left = (label_x, label_y)
        background_bottom_right = (
            min(image.shape[1] - 1, label_x + text_size[0] + 8),
            min(image.shape[0] - 1, label_y + text_size[1] + baseline + 8),
        )
        cv2.rectangle(image, background_top_left, background_bottom_right, color, -1, lineType=cv2.LINE_8)
        text_color = (0, 0, 0) if self._luma(color) > 145 else (255, 255, 255)
        cv2.putText(
            image,
            label,
            (label_x + 4, label_y + text_size[1] + 3),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_ball_marker(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, _ = bbox
        center_x = (x1 + x2) // 2
        tip_y = max(0, y1 - 2)
        top_y = max(0, tip_y - 12)
        points = np.asarray(
            [
                [center_x, tip_y],
                [center_x - 9, top_y],
                [center_x + 9, top_y],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, points, self.BALL_COLOR, lineType=cv2.LINE_8)
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_8)

    def _draw_calibration_status(self, image: np.ndarray, cal_status: str) -> None:
        label, color = self.CAL_STATUS.get(str(cal_status).lower(), self.CAL_STATUS["unavailable"])
        cv2.rectangle(image, (10, 10), (180, 40), color, -1, lineType=cv2.LINE_8)
        cv2.putText(
            image,
            label,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_kit_clash_warning(self, image: np.ndarray) -> None:
        text = "⚠ Kit Clash"
        fallback_text = "! Kit Clash"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(fallback_text, font, scale, thickness)
        x2 = image.shape[1] - 10
        x1 = max(0, x2 - text_size[0] - 16)
        cv2.rectangle(image, (x1, 10), (x2, 40), (0, 215, 255), -1, lineType=cv2.LINE_8)
        if self._draw_pillow_text(image, text, (x1 + 8, 14)):
            return
        cv2.putText(image, fallback_text, (x1 + 8, 31), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_pillow_text(image: np.ndarray, text: str, xy: tuple[int, int]) -> bool:
        font_candidates = [
            "C:/Windows/Fonts/seguisym.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        try:
            font = None
            for candidate in font_candidates:
                try:
                    font = ImageFont.truetype(candidate, 18)
                    break
                except OSError:
                    continue
            if font is None:
                return False
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_image)
            draw.text(xy, text, font=font, fill=(0, 0, 0))
            image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
            return True
        except Exception:
            return False

    def _player_color(self, player: Any) -> tuple[int, int, int]:
        team_label = str(self._get_value(player, "team_label", "unknown") or "unknown")
        role_label = str(self._get_value(player, "role_label", "player") or "player")
        if team_label == "referee" or role_label == "referee":
            return self.referee_color
        if team_label == "team_a_gk":
            return self.team_a_gk_color
        if team_label == "team_b_gk":
            return self.team_b_gk_color
        if team_label == "team_a":
            return self.team_a_color
        if team_label == "team_b":
            return self.team_b_color
        if role_label == "goalkeeper":
            return self.team_a_gk_color
        return self.UNKNOWN_COLOR

    def _has_kit_clash(self) -> bool:
        team_a = np.asarray(self.team_a_color, dtype=np.float64)
        team_b = np.asarray(self.team_b_color, dtype=np.float64)
        return float(np.linalg.norm(team_a - team_b)) < 40.0

    def _color(self, config_key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        values = self.config.get(config_key, default)
        return tuple(int(np.clip(value, 0, 255)) for value in values)

    @staticmethod
    def _extract_bbox(obj: Any) -> tuple[float, float, float, float] | None:
        bbox = FrameAnnotator._get_value(obj, "bbox", None)
        if bbox is None:
            return None
        values = tuple(float(value) for value in bbox)
        if len(values) != 4:
            raise ValueError("bbox must contain x1, y1, x2, y2")
        return values

    @staticmethod
    def _clamp_bbox(
        bbox: tuple[float, float, float, float],
        shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _luma(color: tuple[int, int, int]) -> float:
        blue, green, red = color
        return 0.114 * blue + 0.587 * green + 0.299 * red
