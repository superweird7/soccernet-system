from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.video_panel import _pixmap_from_bgr


ROLE_TO_CONFIG = {
    "Team A Player": "team_a_color_bgr",
    "Team A GK": "team_a_gk_color_bgr",
    "Team B Player": "team_b_color_bgr",
    "Team B GK": "team_b_gk_color_bgr",
    "Referee": "referee_color_bgr",
}


class TeamColorPicker(QDialog):
    def __init__(self, player_crops: list[np.ndarray], config: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick Team Colors")
        self.config = dict(config)
        self.player_crops = [crop.copy() for crop in player_crops if crop is not None and crop.size]
        self.assignments: dict[int, str] = {}

        self.role_combo = QComboBox()
        self.role_combo.addItems(list(ROLE_TO_CONFIG.keys()))
        self.crop_buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a role, then click player crops to assign colors."))
        layout.addWidget(self.role_combo)
        layout.addWidget(self._crop_grid(), 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 480)

    def selected_color_config(self) -> dict:
        color_config = {}
        grouped: dict[str, list[np.ndarray]] = {}
        for index, role in self.assignments.items():
            grouped.setdefault(role, []).append(self.player_crops[index].reshape(-1, 3).mean(axis=0))

        for role, colors in grouped.items():
            key = ROLE_TO_CONFIG[role]
            mean_bgr = np.mean(np.vstack(colors), axis=0)
            color_config[key] = [int(round(value)) for value in mean_bgr]
        return color_config

    def _crop_grid(self) -> QScrollArea:
        container = QWidget()
        grid = QGridLayout(container)
        columns = 4
        if not self.player_crops:
            grid.addWidget(QLabel("No player crops available yet."), 0, 0)
        for index, crop in enumerate(self.player_crops):
            button = QPushButton("Unassigned")
            button.setToolTip("Click to assign the selected role")
            button.setIcon(QIcon(_pixmap_from_bgr(self._thumbnail(crop))))
            button.setIconSize(button.sizeHint())
            button.clicked.connect(lambda checked=False, idx=index: self.assign_role(idx))
            self.crop_buttons.append(button)
            grid.addWidget(button, index // columns, index % columns)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def assign_role(self, crop_index: int) -> None:
        role = self.role_combo.currentText()
        self.assignments[crop_index] = role
        self.crop_buttons[crop_index].setText(role)

    @staticmethod
    def _thumbnail(crop: np.ndarray) -> np.ndarray:
        height = 96
        width = max(48, int(crop.shape[1] * (height / max(1, crop.shape[0]))))
        resized = cv2.resize(crop, (width, height))
        canvas = np.full((112, 112, 3), 230, dtype=np.uint8)
        x = max(0, (canvas.shape[1] - width) // 2)
        canvas[8 : 8 + resized.shape[0], x : x + min(width, canvas.shape[1] - x)] = resized[
            :, : min(width, canvas.shape[1] - x)
        ]
        return canvas
