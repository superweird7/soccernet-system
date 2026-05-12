import sys
import types
from pathlib import Path

import numpy as np
import pytest


class FakeCuda:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available


class FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor(
            [
                [10, 20, 30, 60],
                [40, 50, 70, 80],
                [90, 100, 120, 130],
                [140, 150, 160, 170],
            ]
        )
        self.cls = FakeTensor([0, 32, 0, 32])
        self.conf = FakeTensor([0.36, 0.21, 0.34, 0.19])


class FakeResult:
    boxes = FakeBoxes()


class FakeYOLO:
    calls = []

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.ckpt_path = "missing-fake-yolo11x.pt"
        FakeYOLO.calls.append(("init", self.model_path))

    def to(self, device):
        FakeYOLO.calls.append(("to", device))
        return self

    def predict(self, frame, device, verbose, conf, classes, imgsz):
        FakeYOLO.calls.append(
            (
                "predict",
                {
                    "device": device,
                    "verbose": verbose,
                    "conf": conf,
                    "classes": tuple(classes),
                    "imgsz": imgsz,
                    "shape": frame.shape,
                },
            )
        )
        return [FakeResult()]


@pytest.fixture(autouse=True)
def reset_fake_yolo():
    FakeYOLO.calls = []


def install_fake_modules(monkeypatch, cuda_available=True):
    fake_torch = types.SimpleNamespace(cuda=FakeCuda(cuda_available))
    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)


def test_detector_hard_fails_when_cuda_unavailable(monkeypatch, tmp_path):
    install_fake_modules(monkeypatch, cuda_available=False)

    from core.detector import FootballDetector

    with pytest.raises(RuntimeError, match="CUDA is required"):
        FootballDetector(tmp_path / "yolo11x.pt")


def test_detector_loads_yolo11x_when_weights_missing(monkeypatch, tmp_path):
    install_fake_modules(monkeypatch, cuda_available=True)

    from core.detector import FootballDetector

    model_path = tmp_path / "models" / "yolo11x.pt"
    FootballDetector(model_path)

    assert FakeYOLO.calls[0] == ("init", "yolo11x.pt")
    assert FakeYOLO.calls[1] == ("to", "cuda")
    assert model_path.parent.is_dir()


def test_detector_uses_existing_model_path(monkeypatch, tmp_path):
    install_fake_modules(monkeypatch, cuda_available=True)

    from core.detector import FootballDetector

    model_path = tmp_path / "models" / "yolo11x.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"existing weights")

    FootballDetector(model_path)

    assert FakeYOLO.calls[0] == ("init", str(model_path))


def test_detect_maps_coco_person_and_sports_ball_with_separate_thresholds(
    monkeypatch, tmp_path
):
    install_fake_modules(monkeypatch, cuda_available=True)

    from core.detector import FootballDetector

    detector = FootballDetector(
        tmp_path / "models" / "yolo11x.pt",
        conf_player=0.35,
        conf_ball=0.20,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert [(d.class_id, d.label) for d in detections] == [
        (1, "player"),
        (0, "ball"),
    ]
    assert [d.confidence for d in detections] == [0.36, 0.21]
    assert detections[0].bbox == (10.0, 20.0, 30.0, 60.0)
    assert FakeYOLO.calls[-1][1]["classes"] == (0, 32)
    assert FakeYOLO.calls[-1][1]["conf"] == 0.20


def test_detector_uses_architecture_inference_size(monkeypatch, tmp_path):
    install_fake_modules(monkeypatch, cuda_available=True)

    from core.detector import FootballDetector

    detector = FootballDetector(tmp_path / "models" / "yolo11x.pt", imgsz=1280)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detector.detect(frame)

    assert FakeYOLO.calls[-1][1]["imgsz"] == 1280


def test_detection_dataclass_is_hashable_import_contract():
    from core.detector import Detection

    detection = Detection(
        bbox=(1.0, 2.0, 3.0, 4.0),
        class_id=1,
        confidence=0.9,
        label="player",
    )

    assert detection.bbox == (1.0, 2.0, 3.0, 4.0)
    assert detection.class_id == 1
    assert detection.confidence == 0.9
    assert detection.label == "player"
