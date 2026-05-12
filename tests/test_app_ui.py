import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakePipeline(QObject):
    frame_ready = pyqtSignal(object, object)
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.loaded_path = None
        self.started = False
        self.stopped = False
        self.color_config = None
        self.total_frames = 120
        self.seeked_frames = []

    def load_video(self, path):
        self.loaded_path = Path(path)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def isRunning(self):
        return self.started and not self.stopped

    def set_color_config(self, color_config):
        self.color_config = dict(color_config)

    def seek(self, frame_idx):
        self.seeked_frames.append(int(frame_idx))


def test_video_and_radar_panels_accept_numpy_frames(qapp):
    from app.radar_panel import RadarPanel
    from app.video_panel import VideoPanel

    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    radar = np.zeros((52, 80, 3), dtype=np.uint8)
    radar[:, :] = (0, 200, 0)

    video_panel = VideoPanel()
    radar_panel = RadarPanel()
    video_panel.resize(640, 360)
    radar_panel.resize(420, 300)

    video_panel.update_frame(frame)
    radar_panel.update_radar(radar)

    assert not video_panel.image_label.pixmap().isNull()
    assert not radar_panel.image_label.pixmap().isNull()
    assert radar_panel.minimumWidth() == 420
    assert radar_panel.maximumWidth() == 420


def test_control_panel_emits_actions_and_updates_stats(qapp):
    from app.control_panel import ControlPanel

    panel = ControlPanel(
        {
            "team_a_name": "Iraq",
            "team_b_name": "Bolivia",
            "detection_confidence_player": 0.35,
            "detection_confidence_ball": 0.20,
        }
    )
    events = []
    panel.browse_requested.connect(lambda: events.append("browse"))
    panel.start_requested.connect(lambda: events.append("start"))
    panel.stop_requested.connect(lambda: events.append("stop"))

    panel.browse_button.click()
    panel.start_button.click()
    panel.stop_button.click()
    panel.update_stats(
        {
            "players_detected": 17,
            "ball_visible": True,
            "fps": 28.4,
            "cal_status": "tvcalib",
            "frame_idx": 24600,
        }
    )

    assert events == ["browse", "start", "stop"]
    assert panel.players_value.text() == "17"
    assert panel.ball_value.text() == "Yes"
    assert panel.calibration_value.text() == "tvcalib"
    assert panel.frame_value.text() == "24600"


def test_control_panel_confidence_controls_use_zero_to_one_defaults(qapp):
    from app.control_panel import ControlPanel

    panel = ControlPanel({})
    changes = []
    panel.confidence_changed.connect(lambda name, value: changes.append((name, value)))

    assert panel.player_slider.minimum() == 0
    assert panel.player_slider.maximum() == 100
    assert panel.ball_slider.minimum() == 0
    assert panel.ball_slider.maximum() == 100
    assert panel.player_slider.value() == 35
    assert panel.ball_slider.value() == 20
    assert panel.player_value_spin.value() == 0.35
    assert panel.ball_value_spin.value() == 0.20

    panel.player_slider.setValue(0)
    panel.ball_slider.setValue(100)

    assert changes[-2:] == [("player", 0.0), ("ball", 1.0)]


def test_control_panel_kit_color_controls_emit_config_and_refresh_swatches(qapp):
    from app.control_panel import ControlPanel

    panel = ControlPanel({})
    changes = []
    panel.kit_color_changed.connect(lambda payload: changes.append(payload))

    panel.set_kit_color("team_a_color_bgr", [1, 2, 3])

    assert changes[-1] == {"team_a_color_bgr": [1, 2, 3]}
    assert panel.color_config()["team_a_color_bgr"] == [1, 2, 3]
    assert "rgb(3, 2, 1)" in panel.kit_swatch_buttons["team_a_color_bgr"].styleSheet()


def test_control_panel_seek_slider_emits_user_seeks_and_tracks_stats(qapp):
    from app.control_panel import ControlPanel

    panel = ControlPanel({})
    seeks = []
    panel.seek_requested.connect(lambda frame_idx: seeks.append(frame_idx))

    panel.set_frame_range(100)
    assert panel.seek_slider.isEnabled()
    assert panel.seek_slider.minimum() == 0
    assert panel.seek_slider.maximum() == 99
    assert panel.seek_value.text() == "0 / 99"

    panel.seek_slider.setValue(42)
    assert seeks == [42]
    assert panel.seek_value.text() == "42 / 99"

    seeks.clear()
    panel.update_stats({"frame_idx": 43})

    assert panel.seek_slider.value() == 43
    assert panel.seek_value.text() == "43 / 99"
    assert seeks == []


def test_team_color_picker_assigns_crop_role_and_returns_color_config(qapp):
    from app.team_color_picker import TeamColorPicker

    crop = np.zeros((24, 16, 3), dtype=np.uint8)
    crop[:, :] = (12, 34, 56)
    picker = TeamColorPicker([crop], {})

    picker.role_combo.setCurrentText("Team A Player")
    picker.crop_buttons[0].click()

    config = picker.selected_color_config()
    assert config["team_a_color_bgr"] == [12, 34, 56]


def test_main_window_three_panel_layout_and_pipeline_signal_wiring(qapp, tmp_path):
    from app.main_window import MainWindow

    created = []

    def factory(config):
        pipeline = FakePipeline(config)
        created.append(pipeline)
        return pipeline

    window = MainWindow(
        {
            "team_a_name": "Iraq",
            "team_b_name": "Bolivia",
            "pitch_template_path": "assets/pitch_template.png",
            "radar_width": 800,
            "radar_height": 520,
            "pitch_width_cm": 10500,
            "pitch_height_cm": 6800,
        },
        pipeline_factory=factory,
    )
    source = tmp_path / "frames"
    source.mkdir()

    window.load_source(source)
    window.start_pipeline()
    created[0].frame_ready.emit(
        np.zeros((108, 192, 3), dtype=np.uint8),
        np.zeros((52, 80, 3), dtype=np.uint8),
    )
    created[0].stats_updated.emit(
        {"players_detected": 3, "ball_visible": False, "fps": 12.0, "cal_status": "manual", "frame_idx": 4}
    )
    window.stop_pipeline()

    assert window.windowTitle() == "Iraqi Football Vision Pro"
    assert created[0].loaded_path == source
    assert created[0].started is True
    assert created[0].stopped is True
    assert window.radar_panel.width() == 420 or window.radar_panel.minimumWidth() == 420
    assert window.control_panel.players_value.text() == "3"
    assert window.control_panel.seek_slider.value() == 4
    assert window.menuBar().actions()[0].text() == "File"


def test_main_window_loads_seek_range_and_forwards_slider_requests(qapp, tmp_path):
    from app.main_window import MainWindow

    created = []

    def factory(config):
        pipeline = FakePipeline(config)
        pipeline.total_frames = 75
        created.append(pipeline)
        return pipeline

    window = MainWindow(
        {
            "pitch_template_path": "assets/pitch_template.png",
            "radar_width": 800,
            "radar_height": 520,
            "pitch_width_cm": 10500,
            "pitch_height_cm": 6800,
        },
        pipeline_factory=factory,
    )
    source = tmp_path / "frames"
    source.mkdir()

    window.load_source(source)
    assert window.control_panel.seek_slider.maximum() == 74

    window.control_panel.seek_slider.setValue(12)

    assert created[0].seeked_frames == [12]


def test_main_window_forwards_confidence_slider_values_to_pipeline(qapp, tmp_path):
    from app.main_window import MainWindow

    created = []

    def factory(config):
        pipeline = FakePipeline(config)
        created.append(pipeline)
        return pipeline

    window = MainWindow(
        {
            "detection_confidence_player": 0.35,
            "detection_confidence_ball": 0.20,
            "pitch_template_path": "assets/pitch_template.png",
            "radar_width": 800,
            "radar_height": 520,
            "pitch_width_cm": 10500,
            "pitch_height_cm": 6800,
        },
        pipeline_factory=factory,
    )
    source = tmp_path / "frames"
    source.mkdir()
    window.load_source(source)

    window.control_panel.player_slider.setValue(36)
    window.control_panel.ball_slider.setValue(21)

    assert created[0].config["detection_confidence_player"] == 0.36
    assert created[0].config["detection_confidence_ball"] == 0.21


def test_main_window_forwards_manual_kit_colors_to_pipeline(qapp, tmp_path):
    from app.main_window import MainWindow

    created = []

    def factory(config):
        pipeline = FakePipeline(config)
        created.append(pipeline)
        return pipeline

    window = MainWindow(
        {
            "pitch_template_path": "assets/pitch_template.png",
            "radar_width": 800,
            "radar_height": 520,
            "pitch_width_cm": 10500,
            "pitch_height_cm": 6800,
        },
        pipeline_factory=factory,
    )
    source = tmp_path / "frames"
    source.mkdir()
    window.load_source(source)

    window.control_panel.set_kit_color("team_b_gk_color_bgr", [11, 22, 33])

    assert window.config["team_b_gk_color_bgr"] == [11, 22, 33]
    assert created[0].color_config == {"team_b_gk_color_bgr": [11, 22, 33]}


def test_main_create_app_returns_window_without_starting_event_loop(qapp):
    import main

    app, window = main.create_app([])

    assert app is QApplication.instance()
    assert window.windowTitle() == "Iraqi Football Vision Pro"
    window.close()
