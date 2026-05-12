from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QFrame):
    KIT_COLOR_KEYS = [
        ("A", "team_a_color_bgr", [204, 153, 0]),
        ("A GK", "team_a_gk_color_bgr", [0, 215, 255]),
        ("B", "team_b_color_bgr", [255, 255, 255]),
        ("B GK", "team_b_gk_color_bgr", [128, 0, 128]),
        ("REF", "referee_color_bgr", [0, 0, 0]),
    ]

    browse_requested = pyqtSignal()
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    colors_requested = pyqtSignal()
    confidence_changed = pyqtSignal(str, float)
    calibration_mode_changed = pyqtSignal(str)
    seek_requested = pyqtSignal(int)
    kit_color_changed = pyqtSignal(dict)

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self.config = dict(config)
        self.setObjectName("controlPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(250)
        self.setMaximumWidth(280)

        self.browse_button = QPushButton("Browse Video")
        self.pick_colors_button = QPushButton("Pick Team Colors")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.save_button = QPushButton("Save Output")
        self.benchmark_button = QPushButton("Benchmark")

        self.team_a_input = QLineEdit(self.config.get("team_a_name", "Team A"))
        self.team_b_input = QLineEdit(self.config.get("team_b_name", "Team B"))
        self.kit_colors = {
            key: self._normalize_bgr(self.config.get(key, default))
            for _, key, default in self.KIT_COLOR_KEYS
        }
        self.kit_swatch_buttons: dict[str, QPushButton] = {}
        self.player_slider, self.player_value_spin = self._confidence_control(
            self.config.get("detection_confidence_player", 0.35)
        )
        self.ball_slider, self.ball_value_spin = self._confidence_control(
            self.config.get("detection_confidence_ball", 0.20)
        )
        self._seek_frame_max = 0
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_value = QLabel("0 / 0")

        self.players_value = QLabel("0")
        self.ball_value = QLabel("No")
        self.calibration_value = QLabel("unavailable")
        self.fps_value = QLabel("0.0")
        self.frame_value = QLabel("0")
        self.source_value = QLabel("No source")
        self.source_value.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.source_value)
        layout.addWidget(self._seek_group())
        layout.addWidget(self._team_group())
        layout.addWidget(self.pick_colors_button)
        layout.addWidget(self._swatch_group())
        layout.addWidget(self._calibration_group())
        layout.addWidget(self._confidence_group())
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.benchmark_button)
        layout.addWidget(self._stats_group())
        layout.addStretch(1)

        self.browse_button.clicked.connect(self.browse_requested)
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.pick_colors_button.clicked.connect(self.colors_requested)
        self.player_slider.valueChanged.connect(lambda value: self._slider_changed("player", value))
        self.ball_slider.valueChanged.connect(lambda value: self._slider_changed("ball", value))
        self.player_value_spin.valueChanged.connect(lambda value: self._spin_changed(self.player_slider, "player", value))
        self.ball_value_spin.valueChanged.connect(lambda value: self._spin_changed(self.ball_slider, "ball", value))
        self.seek_slider.valueChanged.connect(self._seek_changed)

    def update_stats(self, stats: dict) -> None:
        self.players_value.setText(str(stats.get("players_detected", 0)))
        self.ball_value.setText("Yes" if stats.get("ball_visible") else "No")
        self.calibration_value.setText(str(stats.get("cal_status", "unavailable")))
        self.fps_value.setText(f"{float(stats.get('fps', 0.0)):.1f}")
        frame_idx = int(stats.get("frame_idx", 0))
        self.frame_value.setText(str(frame_idx))
        self.set_frame_position(frame_idx)

    def set_source_path(self, path: str) -> None:
        self.source_value.setText(path)

    def set_frame_range(self, total_frames: int) -> None:
        frame_count = max(0, int(total_frames))
        self._seek_frame_max = max(0, frame_count - 1)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setRange(0, self._seek_frame_max)
        self.seek_slider.setEnabled(frame_count > 1)
        self.seek_slider.setValue(0)
        self.seek_slider.blockSignals(False)
        self._update_seek_value(0)

    def set_frame_position(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self._seek_frame_max))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)
        self._update_seek_value(frame_idx)

    def color_config(self) -> dict:
        config = {
            "team_a_name": self.team_a_input.text(),
            "team_b_name": self.team_b_input.text(),
            "detection_confidence_player": self.player_value_spin.value(),
            "detection_confidence_ball": self.ball_value_spin.value(),
        }
        config.update({key: list(value) for key, value in self.kit_colors.items()})
        return config

    def set_kit_color(self, config_key: str, bgr, emit: bool = True) -> None:
        if config_key not in self.kit_colors:
            raise KeyError(f"Unknown kit color key: {config_key}")
        color = self._normalize_bgr(bgr)
        self.kit_colors[config_key] = color
        self.config[config_key] = list(color)
        self._set_swatch_style(self.kit_swatch_buttons[config_key], color)
        if emit:
            self.kit_color_changed.emit({config_key: list(color)})

    def set_kit_colors(self, color_config: dict, emit: bool = False) -> None:
        for key in self.kit_colors:
            if key in color_config:
                self.set_kit_color(key, color_config[key], emit=emit)

    def _team_group(self) -> QGroupBox:
        group = QGroupBox("Teams")
        form = QFormLayout(group)
        form.addRow("A", self.team_a_input)
        form.addRow("B", self.team_b_input)
        return group

    def _swatch_group(self) -> QGroupBox:
        group = QGroupBox("Colors")
        grid = QGridLayout(group)
        for row, (label, key, _) in enumerate(self.KIT_COLOR_KEYS):
            grid.addWidget(QLabel(label), row, 0)
            button = self._swatch_button(self.kit_colors[key])
            button.clicked.connect(lambda checked=False, selected_key=key: self._choose_kit_color(selected_key))
            self.kit_swatch_buttons[key] = button
            grid.addWidget(button, row, 1)
        return group

    def _calibration_group(self) -> QGroupBox:
        group = QGroupBox("Calibration")
        layout = QVBoxLayout(group)
        buttons = QButtonGroup(group)
        for mode in ["auto", "manual", "tvcalib"]:
            button = QRadioButton(mode.title() if mode != "tvcalib" else "TVCalib")
            button.setChecked(mode == self.config.get("calibration_mode", "auto"))
            buttons.addButton(button)
            layout.addWidget(button)
            button.toggled.connect(lambda checked, selected=mode: checked and self.calibration_mode_changed.emit(selected))
        return group

    def _confidence_group(self) -> QGroupBox:
        group = QGroupBox("Confidence")
        form = QFormLayout(group)
        form.addRow("Players", self._control_row(self.player_slider, self.player_value_spin))
        form.addRow("Ball", self._control_row(self.ball_slider, self.ball_value_spin))
        return group

    def _seek_group(self) -> QGroupBox:
        group = QGroupBox("Timeline")
        layout = QVBoxLayout(group)
        layout.addWidget(self.seek_slider)
        layout.addWidget(self.seek_value, alignment=Qt.AlignmentFlag.AlignRight)
        return group

    def _stats_group(self) -> QGroupBox:
        group = QGroupBox("Live Stats")
        form = QFormLayout(group)
        form.addRow("Players", self.players_value)
        form.addRow("Ball", self.ball_value)
        form.addRow("Cal", self.calibration_value)
        form.addRow("FPS", self.fps_value)
        form.addRow("Frame", self.frame_value)
        return group

    def _swatch_button(self, bgr) -> QPushButton:
        button = QPushButton()
        button.setFixedSize(42, 22)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip("Change kit color")
        self._set_swatch_style(button, self._normalize_bgr(bgr))
        return button

    def _choose_kit_color(self, config_key: str) -> None:
        blue, green, red = self.kit_colors[config_key]
        selected = QColorDialog.getColor(QColor(red, green, blue), self, "Choose kit color")
        if not selected.isValid():
            return
        self.set_kit_color(config_key, [selected.blue(), selected.green(), selected.red()])

    @staticmethod
    def _set_swatch_style(button: QPushButton, bgr) -> None:
        blue, green, red = [int(value) for value in bgr]
        button.setStyleSheet(
            f"background: rgb({red}, {green}, {blue}); border: 1px solid #333;"
        )

    @staticmethod
    def _normalize_bgr(bgr) -> list[int]:
        values = list(bgr)
        if len(values) != 3:
            raise ValueError("kit color must contain three BGR values")
        return [int(max(0, min(255, value))) for value in values]

    @staticmethod
    def _control_row(slider: QSlider, spin: QDoubleSpinBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return row

    @staticmethod
    def _confidence_control(value: float) -> tuple[QSlider, QDoubleSpinBox]:
        value = max(0.0, min(float(value), 1.0))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(value * 100)))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setFixedWidth(64)
        return slider, spin

    def _slider_changed(self, name: str, value: int) -> None:
        spin = self.player_value_spin if name == "player" else self.ball_value_spin
        float_value = value / 100.0
        if abs(spin.value() - float_value) > 0.001:
            spin.blockSignals(True)
            spin.setValue(float_value)
            spin.blockSignals(False)
        self.confidence_changed.emit(name, float_value)

    def _spin_changed(self, slider: QSlider, name: str, value: float) -> None:
        int_value = int(round(value * 100))
        if slider.value() != int_value:
            slider.blockSignals(True)
            slider.setValue(int_value)
            slider.blockSignals(False)
        self.confidence_changed.emit(name, float(value))

    def _seek_changed(self, frame_idx: int) -> None:
        self._update_seek_value(frame_idx)
        self.seek_requested.emit(int(frame_idx))

    def _update_seek_value(self, frame_idx: int) -> None:
        self.seek_value.setText(f"{int(frame_idx)} / {self._seek_frame_max}")
