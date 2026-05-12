from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from app.video_panel import _pixmap_from_bgr


class RadarPanel(QFrame):
    FIXED_WIDTH = 420

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap = None
        self.setObjectName("radarPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(self.FIXED_WIDTH)
        self.setMaximumWidth(self.FIXED_WIDTH)

        self.image_label = QLabel("2D radar")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background:#102616; color:#d5ead7;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.image_label, 1)

    def update_radar(self, radar_image: np.ndarray) -> None:
        self._pixmap = _pixmap_from_bgr(radar_image)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
