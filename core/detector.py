from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
FOOTBALL_BALL_CLASS = 0
FOOTBALL_PLAYER_CLASS = 1
YOLOV11X_ASSET = "yolo11x.pt"


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    label: str


class FootballDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_player: float = 0.35,
        conf_ball: float = 0.20,
        device: str = "cuda",
        imgsz: int = 1280,
    ):
        self.model_path = Path(model_path)
        self.conf_player = float(conf_player)
        self.conf_ball = float(conf_ball)
        self.device = device
        self.imgsz = int(imgsz)

        self._assert_cuda_available()
        self._model = self._load_model()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray")

        results = self._model.predict(
            frame,
            device=self.device,
            verbose=False,
            conf=min(self.conf_player, self.conf_ball),
            classes=[COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS],
            imgsz=self.imgsz,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        xyxy = _as_numpy(boxes.xyxy)
        class_ids = _as_numpy(boxes.cls).astype(int)
        confidences = _as_numpy(boxes.conf)

        detections: list[Detection] = []
        for bbox, coco_class_id, confidence in zip(xyxy, class_ids, confidences):
            mapped = self._map_detection(coco_class_id, float(confidence))
            if mapped is None:
                continue

            class_id, label = mapped
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in bbox),
                    class_id=class_id,
                    confidence=float(confidence),
                    label=label,
                )
            )

        return detections

    def _assert_cuda_available(self) -> None:
        if self.device != "cuda":
            raise RuntimeError("CUDA is required; CPU fallback is disabled")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; torch.cuda.is_available() returned False")

    def _load_model(self):
        from ultralytics import YOLO

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            model = YOLO(str(self.model_path))
        else:
            model = YOLO(YOLOV11X_ASSET)
            self._copy_downloaded_weights(model)

        return model.to(self.device)

    def _copy_downloaded_weights(self, model) -> None:
        candidates = [
            getattr(model, "ckpt_path", None),
            Path.cwd() / YOLOV11X_ASSET,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            source = Path(candidate)
            if source.is_file() and source.resolve() != self.model_path.resolve():
                shutil.copy2(source, self.model_path)
                return

    def _map_detection(self, coco_class_id: int, confidence: float) -> tuple[int, str] | None:
        if coco_class_id == COCO_PERSON_CLASS and confidence >= self.conf_player:
            return FOOTBALL_PLAYER_CLASS, "player"
        if coco_class_id == COCO_SPORTS_BALL_CLASS and confidence >= self.conf_ball:
            return FOOTBALL_BALL_CLASS, "ball"
        return None


def _as_numpy(values: Sequence | np.ndarray) -> np.ndarray:
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        return values.numpy()
    return np.asarray(values)
