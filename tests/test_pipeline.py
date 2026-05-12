from pathlib import Path

import cv2
import numpy as np

from core.detector import Detection
from core.tracker import TrackedPlayer


def write_frames(directory: Path, count: int, shape=(48, 64, 3)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        frame = np.full(shape, index, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"{index:06d}.jpg"), frame)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [
            Detection((10, 10, 20, 40), 1, 0.9, "player"),
            Detection((30, 10, 40, 40), 1, 0.8, "player"),
            Detection((42, 20, 50, 28), 0, 0.7, "ball"),
        ]


class FakeTracker:
    def update(self, detections, frame):
        return [
            TrackedPlayer(1, (10, 10, 20, 40), 1, 0.9),
            TrackedPlayer(2, (30, 10, 40, 40), 1, 0.8),
        ]


class FakeTeamClassifier:
    def __init__(self):
        self.calls = []
        self.color_config = None
        self.shutdown_called = False
        self.flush_called = False
        self.reset_calls = 0

    def predict_from_frame(self, frame, bbox, track_id, frame_idx):
        self.calls.append((track_id, frame_idx))
        if track_id == 1:
            return "team_a", "player", 0.8
        return "team_b", "player", 0.8

    def update_manual_colors(self, color_config):
        self.color_config = dict(color_config)

    def flush(self, timeout=None):
        self.flush_called = True

    def reset(self):
        self.reset_calls += 1

    def shutdown(self):
        self.shutdown_called = True


class FakeHomography:
    def __init__(self, transformed=None, status="tvcalib", calibrated=True):
        self.update_calls = []
        self.transformed = transformed
        self.status = status
        self.calibrated = calibrated

    def update(self, frame, frame_idx):
        self.update_calls.append(frame_idx)

    def transform_points(self, points):
        if self.transformed is None:
            return None
        if callable(self.transformed):
            return self.transformed(points)
        return np.asarray(self.transformed, dtype=np.float64)[: len(points)]

    def get_status(self):
        return self.status

    def is_calibrated(self):
        return self.calibrated


class FakeAnnotator:
    def __init__(self):
        self.calls = []

    def annotate(self, frame, players, ball=None, cal_status="unavailable"):
        self.calls.append({"players": list(players), "ball": ball, "cal_status": cal_status})
        annotated = frame.copy()
        annotated[0, 0] = (255, 255, 255)
        return annotated


class FakeRadar:
    def __init__(self):
        self.calls = []

    def render(self, players, ball_pos=None, calibrated=True):
        self.calls.append({"players": list(players), "ball_pos": ball_pos, "calibrated": calibrated})
        return np.zeros((52, 80, 3), dtype=np.uint8)


def make_pipeline(tmp_path, frame_count=3):
    from core.pipeline import AnalysisPipeline

    frames = tmp_path / "frames"
    write_frames(frames, frame_count)
    team_classifier = FakeTeamClassifier()
    homography = FakeHomography(
        transformed=np.asarray([[1000.0, 1000.0], [11000.0, 1000.0]], dtype=np.float64),
        status="tvcalib",
        calibrated=True,
    )
    annotator = FakeAnnotator()
    radar = FakeRadar()
    pipeline = AnalysisPipeline(
        {"frame_limit": frame_count, "pitch_template_path": "assets/pitch_template.png"},
        detector=FakeDetector(),
        tracker=FakeTracker(),
        team_classifier=team_classifier,
        homography=homography,
        annotator=annotator,
        radar=radar,
    )
    pipeline.load_video(frames)
    return pipeline, team_classifier, homography, annotator, radar


def test_pipeline_runs_image_sequence_and_emits_frame_ready_and_stats(tmp_path):
    pipeline, team_classifier, homography, annotator, radar = make_pipeline(tmp_path, frame_count=3)
    frames = []
    stats = []
    errors = []
    pipeline.frame_ready.connect(lambda annotated, radar_image: frames.append((annotated, radar_image)))
    pipeline.stats_updated.connect(lambda payload: stats.append(payload))
    pipeline.error_occurred.connect(lambda message: errors.append(message))

    pipeline.run()

    assert len(frames) == 3
    assert len(stats) == 3
    assert errors == []
    assert stats[-1]["players_detected"] == 1
    assert stats[-1]["ball_visible"] is True
    assert stats[-1]["cal_status"] == "tvcalib"
    assert stats[-1]["frame_idx"] == 2
    assert stats[-1]["fps"] > 0
    assert [call[1] for call in team_classifier.calls] == [0, 0, 1, 1, 2, 2]
    assert homography.update_calls == [0, 1, 2]
    assert len(annotator.calls[-1]["players"]) == 1
    assert len(radar.calls[-1]["players"]) == 1
    assert team_classifier.flush_called is False
    assert team_classifier.shutdown_called is False


def test_pipeline_logs_counts_and_keeps_annotations_when_calibration_unavailable(tmp_path, capsys):
    pipeline, _, homography, annotator, radar = make_pipeline(tmp_path, frame_count=1)
    homography.transformed = None
    homography.status = "unavailable"
    homography.calibrated = False
    stats = []
    pipeline.stats_updated.connect(lambda payload: stats.append(payload))

    pipeline.run()

    captured = capsys.readouterr()
    assert "[pipeline] frame=0 detections=3 tracked_players=2 annotated_players=2" in captured.out
    assert stats[-1]["players_detected"] == 2
    assert len(annotator.calls[-1]["players"]) == 2
    assert len(radar.calls[-1]["players"]) == 2
    assert radar.calls[-1]["calibrated"] is False


def test_pipeline_filters_sideline_people_when_uncalibrated(tmp_path, capsys):
    pipeline, _, homography, annotator, radar = make_pipeline(tmp_path, frame_count=1)
    homography.transformed = None
    homography.status = "unavailable"
    homography.calibrated = False
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[:, :] = (40, 40, 40)
    frame[35:, :25] = (75, 150, 85)
    pipeline._image_paths = [tmp_path / "dummy.jpg"]
    pipeline._source_type = "images"
    pipeline._read_frame = lambda frame_idx: frame.copy()

    stats = []
    pipeline.stats_updated.connect(lambda payload: stats.append(payload))

    pipeline.run()

    captured = capsys.readouterr()
    assert "[pipeline] frame=0 detections=3 tracked_players=1 annotated_players=1" in captured.out
    assert stats[-1]["players_detected"] == 1
    assert len(annotator.calls[-1]["players"]) == 1
    assert annotator.calls[-1]["players"][0]["track_id"] == 1
    assert len(radar.calls[-1]["players"]) == 1


def test_pipeline_pause_resume_stop_and_seek(tmp_path):
    pipeline, team_classifier, _, _, _ = make_pipeline(tmp_path, frame_count=5)
    assert pipeline.total_frames == 5

    pipeline.pause()
    assert pipeline.is_paused

    pipeline.resume()
    assert not pipeline.is_paused

    pipeline.seek(3)
    assert pipeline.frame_idx == 3
    assert team_classifier.reset_calls == 1

    pipeline.seek(99)
    assert pipeline.frame_idx == 4
    assert team_classifier.reset_calls == 2

    frames = []
    pipeline.frame_ready.connect(lambda annotated, radar_image: frames.append((annotated, radar_image)))
    pipeline.run()
    assert len(frames) == 1

    pipeline.stop()
    assert pipeline.should_stop


def test_pipeline_load_video_sets_source_calibration_id_and_resets_source_state(tmp_path):
    pipeline, team_classifier, _, _, _ = make_pipeline(tmp_path, frame_count=1)
    next_frames = tmp_path / "Iraq vs. Bolivia 2026"
    write_frames(next_frames, 2)

    pipeline.load_video(next_frames)

    assert pipeline.config["source_path"] == str(next_frames)
    assert pipeline.config["calibration_source_id"] == "Iraq vs. Bolivia 2026"
    assert team_classifier.reset_calls == 1
    assert pipeline.homography is None


def test_pipeline_set_color_config_forwards_to_team_classifier_and_rebuilds_drawers(tmp_path):
    pipeline, team_classifier, _, annotator_before, radar_before = make_pipeline(tmp_path, frame_count=1)
    color_config = {"team_a_color_bgr": [1, 2, 3]}

    pipeline.set_color_config(color_config)

    assert team_classifier.color_config == color_config
    assert pipeline.annotator is not annotator_before
    assert pipeline.radar is not radar_before
