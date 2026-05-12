from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from app.control_panel import ControlPanel
from app.radar_panel import RadarPanel
from app.team_color_picker import TeamColorPicker
from app.video_panel import VideoPanel
from core.pipeline import AnalysisPipeline


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: dict,
        pipeline_factory: Callable[[dict], object] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = dict(config)
        self.config.setdefault("project_root", str(Path.cwd()))
        self.pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self.pipeline = None
        self.source_path: Path | None = None
        self.player_crops = []

        self.setWindowTitle("Iraqi Football Vision Pro")
        self.resize(1440, 850)
        self._build_menus()
        self._build_layout()
        self._connect_control_panel()

    def load_source(self, path: str | Path) -> None:
        source = Path(path)
        self.source_path = source
        if self.pipeline is not None and hasattr(self.pipeline, "isRunning") and self.pipeline.isRunning():
            self.pipeline.stop()

        self.pipeline = self.pipeline_factory(self._pipeline_config())
        self._connect_pipeline(self.pipeline)
        self.pipeline.load_video(source)
        self.control_panel.set_source_path(str(source))
        self.control_panel.set_frame_range(int(getattr(self.pipeline, "total_frames", 0)))
        self.control_panel.set_frame_position(0)
        self.statusBar().showMessage(f"Loaded {source}")

    def start_pipeline(self) -> None:
        if self.source_path is None:
            self.show_error("Load a video or image sequence first.")
            return
        if self.pipeline is None:
            self.load_source(self.source_path)
        if hasattr(self.pipeline, "isRunning") and self.pipeline.isRunning():
            return
        self.pipeline.start()
        self.statusBar().showMessage("Analysis running")

    def stop_pipeline(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.statusBar().showMessage("Analysis stopped")

    def open_source_dialog(self) -> None:
        start_dir = self.config.get("default_video_dir", str(Path.cwd()))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            start_dir,
            "Video and Images (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All files (*.*)",
        )
        selected = file_path
        if not selected:
            selected = QFileDialog.getExistingDirectory(self, "Open Image Sequence", start_dir)
        if selected:
            self.load_source(selected)

    def pick_team_colors(self) -> None:
        dialog = TeamColorPicker(self.player_crops, self.config, self)
        if dialog.exec():
            color_config = dialog.selected_color_config()
            if color_config:
                self.apply_color_config(color_config)

    def apply_color_config(self, color_config: dict) -> None:
        self.config.update(color_config)
        if hasattr(self.control_panel, "set_kit_colors"):
            self.control_panel.set_kit_colors(color_config, emit=False)
        if self.pipeline is not None:
            self.pipeline.set_color_config(color_config)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Football Vision Pro", message)
        self.statusBar().showMessage(message)

    def closeEvent(self, event) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.control_panel = ControlPanel(self.config)
        self.video_panel = VideoPanel()
        self.radar_panel = RadarPanel()

        layout.addWidget(self.control_panel)
        layout.addWidget(self.video_panel, 1)
        layout.addWidget(self.radar_panel)
        self.setCentralWidget(central)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open Video")
        save_action = file_menu.addAction("Save Output")
        exit_action = file_menu.addAction("Exit")
        open_action.triggered.connect(self.open_source_dialog)
        save_action.triggered.connect(lambda: self.statusBar().showMessage("Save Output will be available after export support lands."))
        exit_action.triggered.connect(self.close)

        analysis_menu = self.menuBar().addMenu("Analysis")
        pick_action = analysis_menu.addAction("Pick Team Colors")
        calibrate_action = analysis_menu.addAction("Calibrate Camera")
        benchmark_action = analysis_menu.addAction("Run Benchmark")
        pick_action.triggered.connect(self.pick_team_colors)
        calibrate_action.triggered.connect(lambda: self.statusBar().showMessage("Manual calibration dashboard is a later step."))
        benchmark_action.triggered.connect(lambda: self.statusBar().showMessage("Benchmark can be run from benchmark/benchmark_soccernet.py."))

        help_menu = self.menuBar().addMenu("Help")
        about_action = help_menu.addAction("About")
        camera_action = help_menu.addAction("Camera Setup Guide")
        about_action.triggered.connect(
            lambda: QMessageBox.about(self, "About", "Iraqi Football Vision Pro")
        )
        camera_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "Camera Setup Guide",
                "Use broadcast views with visible pitch lines. TVCalib updates in the background.",
            )
        )

    def _connect_control_panel(self) -> None:
        self.control_panel.browse_requested.connect(self.open_source_dialog)
        self.control_panel.start_requested.connect(self.start_pipeline)
        self.control_panel.stop_requested.connect(self.stop_pipeline)
        self.control_panel.colors_requested.connect(self.pick_team_colors)
        self.control_panel.confidence_changed.connect(self._confidence_changed)
        self.control_panel.calibration_mode_changed.connect(self._calibration_mode_changed)
        self.control_panel.seek_requested.connect(self._seek_requested)
        self.control_panel.kit_color_changed.connect(self.apply_color_config)

    def _connect_pipeline(self, pipeline) -> None:
        pipeline.frame_ready.connect(self._on_frame_ready)
        pipeline.stats_updated.connect(self.control_panel.update_stats)
        pipeline.error_occurred.connect(self.show_error)

    def _on_frame_ready(self, annotated_frame, radar_image) -> None:
        self.video_panel.update_frame(annotated_frame)
        self.radar_panel.update_radar(radar_image)

    def _confidence_changed(self, name: str, value: float) -> None:
        key = "detection_confidence_player" if name == "player" else "detection_confidence_ball"
        self.config[key] = float(value)
        if self.pipeline is not None:
            self.pipeline.config[key] = float(value)

    def _calibration_mode_changed(self, mode: str) -> None:
        self.config["calibration_mode"] = mode
        if self.pipeline is not None:
            self.pipeline.config["calibration_mode"] = mode

    def _seek_requested(self, frame_idx: int) -> None:
        if self.pipeline is None:
            return
        self.pipeline.seek(frame_idx)
        self.statusBar().showMessage(f"Seeked to frame {int(frame_idx)}")

    def _pipeline_config(self) -> dict:
        config = dict(self.config)
        config["defer_component_setup"] = True
        return config

    @staticmethod
    def _default_pipeline_factory(config: dict):
        return AnalysisPipeline(config)
