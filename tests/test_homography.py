import time

import numpy as np


class FakeBackend:
    def __init__(self, homographies, delay=0.0):
        self.homographies = list(homographies)
        self.delay = delay
        self.calls = []

    def estimate(self, frame):
        self.calls.append(frame.shape)
        if self.delay:
            time.sleep(self.delay)
        result = self.homographies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class DownloadFailBackend:
    def estimate(self, frame):
        raise RuntimeError("TVCalib weights download failed: timed out")


def base_config(tmp_path):
    return {
        "pitch_width_cm": 10500,
        "pitch_height_cm": 6800,
        "pitch_boundary_margin_cm": 200,
        "tvcalib_update_every_n_frames": 30,
        "tvcalib_resize_enabled": False,
        "calibration_save_dir": str(tmp_path),
    }


def test_calibrator_returns_unavailable_without_tvcalib_or_manual_json(tmp_path):
    from core.homography import BroadcastCalibrator

    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=FakeBackend([RuntimeError("no lines")]),
        manual_path=tmp_path / "missing.json",
    )

    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=1)
    calibrator.flush(timeout=1)

    assert calibrator.get_status() == "unavailable"
    assert not calibrator.is_calibrated()
    assert calibrator.transform_points(np.asarray([[10.0, 20.0]])) is None


def test_calibrator_loads_manual_json_when_tvcalib_fails(tmp_path):
    from core.homography import BroadcastCalibrator

    manual_path = tmp_path / "manual.json"
    manual_path.write_text(
        '{"homography": [[10, 0, 0], [0, 10, 0], [0, 0, 1]]}',
        encoding="utf-8",
    )
    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=FakeBackend([RuntimeError("tvcalib failed")]),
        manual_path=manual_path,
    )

    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=1)
    calibrator.flush(timeout=1)

    assert calibrator.get_status() == "manual"
    np.testing.assert_allclose(
        calibrator.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[100.0, 200.0]]),
    )


def test_transform_points_filters_outside_pitch_margin(tmp_path):
    from core.homography import BroadcastCalibrator

    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=FakeBackend([np.asarray([[10, 0, 0], [0, 10, 0], [0, 0, 1]])]),
    )
    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=1)
    calibrator.flush(timeout=1)

    transformed = calibrator.transform_points(
        np.asarray(
            [
                [10.0, 20.0],
                [1080.0, 200.0],
                [0.0, -30.0],
            ]
        )
    )

    np.testing.assert_allclose(transformed, np.asarray([[100.0, 200.0]]))


def test_frame_zero_attempts_tvcalib_synchronously_before_background_updates(tmp_path):
    from core.homography import BroadcastCalibrator

    h1 = np.asarray([[10, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
    backend = FakeBackend([h1], delay=0.1)
    calibrator = BroadcastCalibrator(base_config(tmp_path), backend=backend)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    start = time.perf_counter()
    calibrator.update(frame, frame_idx=0)
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.08
    assert calibrator.get_status() == "tvcalib"
    assert len(backend.calls) == 1
    np.testing.assert_allclose(
        calibrator.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[100.0, 200.0]]),
    )


def test_frame_zero_tvcalib_failure_loads_any_saved_manual_json(tmp_path):
    from core.homography import BroadcastCalibrator

    manual_path = tmp_path / "SNGS-116.json"
    manual_path.write_text(
        '{"homography": [[5, 0, 0], [0, 5, 0], [0, 0, 1]]}',
        encoding="utf-8",
    )
    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=FakeBackend([RuntimeError("frame zero failed")]),
    )

    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=0)

    assert calibrator.get_status() == "manual"
    assert calibrator.get_last_error() == "frame zero failed"
    np.testing.assert_allclose(
        calibrator.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[50.0, 100.0]]),
    )


def test_tvcalib_backend_receives_resized_1280_by_720_frame_and_h_is_scaled_back(tmp_path):
    from core.homography import BroadcastCalibrator

    backend = FakeBackend([np.eye(3)])
    config = {
        "pitch_width_cm": 10500,
        "pitch_height_cm": 6800,
        "pitch_boundary_margin_cm": 200,
        "tvcalib_resize_enabled": True,
        "tvcalib_input_width": 1280,
        "tvcalib_input_height": 720,
        "calibration_save_dir": str(tmp_path),
    }
    calibrator = BroadcastCalibrator(config, backend=backend)

    calibrator.update(np.zeros((360, 640, 3), dtype=np.uint8), frame_idx=0)

    assert backend.calls == [(720, 1280, 3)]
    np.testing.assert_allclose(
        calibrator.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[20.0, 40.0]]),
    )


def test_tvcalib_failure_records_exact_error_without_manual_fallback(tmp_path):
    from core.homography import BroadcastCalibrator

    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=DownloadFailBackend(),
        manual_path=tmp_path / "missing.json",
    )

    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=0)

    assert calibrator.get_status() == "unavailable"
    assert calibrator.get_last_error() == "TVCalib weights download failed: timed out"


def test_tvcalib_backend_reports_missing_weights_without_hanging(tmp_path):
    from core.homography import TVCalibBackend

    backend = TVCalibBackend(tmp_path, auto_download=False)

    try:
        backend.estimate(np.zeros((20, 30, 3), dtype=np.uint8))
    except RuntimeError as exc:
        assert "TVCalib weights missing" in str(exc)
    else:
        raise AssertionError("expected missing weights error")


def test_update_runs_in_background_and_only_every_thirty_frames(tmp_path):
    from core.homography import BroadcastCalibrator

    h1 = np.asarray([[10, 0, 0], [0, 10, 0], [0, 0, 1]], dtype=float)
    h2 = np.asarray([[20, 0, 0], [0, 20, 0], [0, 0, 1]], dtype=float)
    backend = FakeBackend([h1, h2], delay=0.2)
    calibrator = BroadcastCalibrator(base_config(tmp_path), backend=backend)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    start = time.perf_counter()
    calibrator.update(frame, frame_idx=1)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert calibrator.get_status() == "unavailable"

    calibrator.flush(timeout=1)
    assert calibrator.get_status() == "tvcalib"
    assert len(backend.calls) == 1

    calibrator.update(frame, frame_idx=20)
    calibrator.flush(timeout=1)
    assert len(backend.calls) == 1

    calibrator.update(frame, frame_idx=31)
    calibrator.flush(timeout=1)
    assert len(backend.calls) == 2
    np.testing.assert_allclose(
        calibrator.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[200.0, 400.0]]),
    )


def test_save_and_load_calibration_roundtrip_uses_manual_status(tmp_path):
    from core.homography import BroadcastCalibrator

    path = tmp_path / "roundtrip.json"
    calibrator = BroadcastCalibrator(
        base_config(tmp_path),
        backend=FakeBackend([np.asarray([[2, 0, 5], [0, 3, 7], [0, 0, 1]])]),
    )
    calibrator.update(np.zeros((20, 30, 3), dtype=np.uint8), frame_idx=1)
    calibrator.flush(timeout=1)

    calibrator.save_calibration(path)
    loaded = BroadcastCalibrator(base_config(tmp_path), backend=FakeBackend([]))
    loaded.load_calibration(path)

    assert loaded.get_status() == "manual"
    np.testing.assert_allclose(
        loaded.transform_points(np.asarray([[10.0, 20.0]])),
        np.asarray([[25.0, 67.0]]),
    )
