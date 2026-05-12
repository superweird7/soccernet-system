from __future__ import annotations

import math
import os
import contextlib
import io
import time
import urllib.request
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

UNKNOWN_RESULT = ("unknown", "unknown", 0.0)
VALID_TEAM_LABELS = {
    "team_a",
    "team_b",
    "team_a_gk",
    "team_b_gk",
    "referee",
    "unknown",
}


@dataclass(frozen=True)
class ReIDResult:
    embedding: np.ndarray
    role_label: str
    confidence: float


class ColorEmbeddingBackend:
    def extract(self, crop: np.ndarray) -> tuple[np.ndarray, str, float]:
        mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
        std = crop.reshape(-1, 3).std(axis=0).astype(np.float32)
        embedding = np.concatenate([mean / 255.0, std / 255.0])
        return embedding, "player", 0.49


class SafePRTReIDBackend:
    baseline_url = "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
    hrnet_url = "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1"

    def __init__(self, model_dir: str | Path, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.weights_path = self.model_dir / "prtreid-soccernet-baseline.pth.tar"
        self.hrnet_path = self.model_dir / "hrnetv2_w32_imagenet_pretrained.pth"
        self._extractor = None
        self._cfg = None
        self.disabled_error: str | None = None
        self._color_backend = ColorEmbeddingBackend()

    def extract(self, crop: np.ndarray) -> tuple[np.ndarray, str, float]:
        if self.disabled_error is not None:
            return self._color_backend.extract(crop)

        try:
            self._ensure_extractor()
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            reid_output = self._extractor([rgb_crop])
            return self._parse_reid_output(reid_output)
        except Exception as exc:  # pragma: no cover - exercised in integration environments
            self.disabled_error = str(exc)
            return self._color_backend.extract(crop)

    def ensure_weights(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        _download_if_missing(self.baseline_url, self.weights_path)
        _download_if_missing(self.hrnet_url, self.hrnet_path)

    def _ensure_extractor(self) -> None:
        if self._extractor is not None:
            return

        self.ensure_weights()

        from prtreid.scripts.main import build_config
        from prtreid.tools.feature_extractor import FeatureExtractor
        from yacs.config import CfgNode as CN

        cfg = CN()
        cfg.project = CN()
        cfg.project.job_id = int(time.time() * 1000)
        cfg.project.logger = CN()
        cfg.project.logger.use_tensorboard = False
        cfg.project.logger.use_wandb = False
        cfg.data = CN()
        cfg.data.root = str(self.model_dir / "reid")
        cfg.data.type = "image"
        cfg.data.sources = ["market1501"]
        cfg.data.targets = ["market1501"]
        cfg.data.height = 256
        cfg.data.width = 128
        cfg.data.save_dir = str(self.model_dir / "runtime")
        cfg.data.workers = 0
        cfg.model = CN()
        cfg.model.name = "bpbreid"
        cfg.model.pretrained = True
        cfg.model.load_config = True
        cfg.model.load_weights = str(self.weights_path)
        cfg.model.bpbreid = CN()
        cfg.model.bpbreid.backbone = "hrnet32"
        cfg.model.bpbreid.hrnet_pretrained_path = str(self.model_dir)
        cfg.model.bpbreid.test_embeddings = ["globl"]

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore", FutureWarning)
            self._cfg = build_config(config=cfg)
            self._extractor = FeatureExtractor(
                self._cfg,
                model_path=str(self.weights_path),
                device=self.device,
                image_size=(self._cfg.data.height, self._cfg.data.width),
                verbose=False,
            )

    def _parse_reid_output(self, reid_output) -> tuple[np.ndarray, str, float]:
        import torch
        from prtreid.utils.tools import extract_test_embeddings

        embeddings, _, _, _, role_scores = extract_test_embeddings(
            reid_output,
            self._cfg.model.bpbreid.test_embeddings,
        )
        embedding = embeddings.detach().cpu().numpy().reshape(embeddings.shape[0], -1)[0]

        if role_scores is None:
            return embedding, "unknown", 0.0

        role_mapping = {
            0: "ball",
            1: "goalkeeper",
            2: "other",
            3: "player",
            4: "referee",
        }
        scores = role_scores["globl"].detach().cpu()
        if len(scores.shape) > 1:
            scores = scores[0]
        probabilities = torch.softmax(scores, dim=0)
        role_idx = int(torch.argmax(probabilities).item())
        role_confidence = float(torch.max(probabilities).item())
        return embedding, role_mapping.get(role_idx, "unknown"), role_confidence


class PRTReidClassifier:
    def __init__(
        self,
        config: dict,
        device: str = "cuda",
        backend=None,
        async_enabled: bool = True,
    ):
        self.config = config
        self.device = device
        self.model_dir = Path(config.get("prtreid_model_path", "models/prtreid"))
        self.confidence_threshold = float(config.get("prtreid_confidence_threshold", 0.5))
        self.role_confidence_threshold = float(config.get("prtreid_role_confidence_threshold", 0.75))
        self.prefer_kit_color_classification = bool(config.get("prefer_kit_color_classification", False))
        self.special_kit_color_max_distance = float(config.get("special_kit_color_max_distance", 95.0))
        self.special_kit_color_margin = float(config.get("special_kit_color_margin", 0.72))
        self.referee_dark_max_channel = float(config.get("referee_dark_max_channel", 92.0))
        self.referee_dark_max_distance = float(config.get("referee_dark_max_distance", 135.0))
        self.warmup_frames = int(config.get("prtreid_warmup_frames", 120))
        self.update_every_n_frames = int(config.get("prtreid_update_every_n_frames", 5))
        self.team_a_side = str(config.get("team_a_side", "left")).lower()
        if self.team_a_side not in {"left", "right"}:
            self.team_a_side = "left"
        self.async_enabled = async_enabled
        self.backend = backend or SafePRTReIDBackend(self.model_dir, device=device)
        self._executor = ThreadPoolExecutor(max_workers=1) if async_enabled else None
        self._pending: dict[int, Future] = {}
        self._cache: dict[int, tuple[str, str, float]] = {}
        self._last_update_frame: dict[int, int] = {}
        self._track_embeddings: dict[int, list[np.ndarray]] = {}
        self._track_roles: dict[int, list[tuple[str, float]]] = {}
        self._track_x_positions: dict[int, list[float]] = {}
        self._track_frame_widths: dict[int, list[float]] = {}
        self._samples_seen = 0
        self._warmup_start_frame: int | None = None
        self._latest_frame_idx: int | None = None
        self._centroids: np.ndarray | None = None
        self._cluster_to_team: dict[int, str] = {}

    @property
    def is_warmed_up(self) -> bool:
        return self._centroids is not None

    @staticmethod
    def crop_player_region(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        jersey_bottom = y1 + int((y2 - y1) * 0.70)
        jersey_bottom = max(y1 + 1, min(height, jersey_bottom))
        return frame[y1:jersey_bottom, x1:x2].copy()

    def predict_from_frame(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        track_id: int,
        frame_idx: int,
    ) -> tuple[str, str, float]:
        self._store_track_position(int(track_id), bbox, frame.shape[1])
        return self.predict(self.crop_player_region(frame, bbox), track_id, frame_idx)

    def warm_up(self, crops: Iterable[np.ndarray], track_ids: Iterable[int]) -> None:
        for crop, track_id in zip(crops, track_ids):
            result = self._extract(crop)
            self._store_embedding(int(track_id), result)
        self.finalize_warmup(force=True)

    def finalize_warmup(self, force: bool = False) -> bool:
        if self.is_warmed_up:
            return True
        if not force and not self._warmup_frame_ready():
            return False

        track_vectors = []
        track_ids = []
        for track_id, embeddings in self._track_embeddings.items():
            if not embeddings:
                continue
            track_ids.append(track_id)
            track_vectors.append(np.mean(np.vstack(embeddings), axis=0))

        if len(track_vectors) < 2:
            return False

        vectors = np.vstack(track_vectors)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(vectors)
        self._centroids = kmeans.cluster_centers_

        self._cluster_to_team = self._build_cluster_to_team(kmeans.labels_, track_ids)
        for track_id, vector, cluster in zip(track_ids, vectors, kmeans.labels_):
            role_label, role_confidence = self._majority_role(track_id)
            self._cache[track_id] = self._compose_prediction(
                vector,
                role_label,
                role_confidence,
                crop=None,
            )
        return True

    def predict(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_idx: int | None = None,
    ) -> tuple[str, str, float]:
        track_id = int(track_id)
        frame_idx = 0 if frame_idx is None else int(frame_idx)
        self._mark_warmup_frame(frame_idx)
        self.poll()
        if not self.is_warmed_up:
            self.finalize_warmup(force=False)

        cached = self._cache.get(track_id)
        last_update = self._last_update_frame.get(track_id)
        if cached is not None and last_update is not None:
            if frame_idx - last_update < self.update_every_n_frames:
                return cached

        if self.async_enabled:
            if track_id not in self._pending:
                self._pending[track_id] = self._executor.submit(
                    self._extract,
                    crop.copy(),
                )
                self._last_update_frame[track_id] = frame_idx
            return cached or UNKNOWN_RESULT

        result = self._extract(crop)
        prediction = self._apply_result(track_id, frame_idx, result, crop)
        return prediction

    def poll(self) -> None:
        complete_track_ids = [
            track_id for track_id, future in self._pending.items() if future.done()
        ]
        for track_id in complete_track_ids:
            future = self._pending.pop(track_id)
            result = future.result()
            frame_idx = self._last_update_frame.get(track_id, 0)
            self._apply_result(track_id, frame_idx, result, crop=None)

    def flush(self, timeout: float | None = None) -> None:
        start = time.perf_counter()
        while self._pending:
            self.poll()
            if not self._pending:
                return
            if timeout is not None and time.perf_counter() - start > timeout:
                raise TimeoutError("Timed out waiting for PRTReid background jobs")
            time.sleep(0.01)

    def update_manual_colors(self, color_config: dict) -> None:
        self.config.update(color_config)

    def reset(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        self._cache.clear()
        self._last_update_frame.clear()
        self._track_embeddings.clear()
        self._track_roles.clear()
        self._track_x_positions.clear()
        self._track_frame_widths.clear()
        self._samples_seen = 0
        self._warmup_start_frame = None
        self._latest_frame_idx = None
        self._centroids = None
        self._cluster_to_team.clear()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _extract(self, crop: np.ndarray) -> ReIDResult:
        if crop.size == 0:
            return ReIDResult(np.zeros(6, dtype=np.float32), "unknown", 0.0)
        embedding, role_label, confidence = self.backend.extract(crop)
        return ReIDResult(np.asarray(embedding, dtype=np.float32).ravel(), role_label, float(confidence))

    def _apply_result(
        self,
        track_id: int,
        frame_idx: int,
        result: ReIDResult,
        crop: np.ndarray | None,
    ) -> tuple[str, str, float]:
        self._store_embedding(track_id, result)
        if not self.is_warmed_up:
            self.finalize_warmup(force=False)
        prediction = self._compose_prediction(
            result.embedding,
            result.role_label,
            result.confidence,
            crop,
        )
        self._cache[track_id] = prediction
        self._last_update_frame[track_id] = frame_idx
        return prediction

    def _store_embedding(self, track_id: int, result: ReIDResult) -> None:
        self._track_embeddings.setdefault(track_id, []).append(result.embedding)
        self._track_roles.setdefault(track_id, []).append((result.role_label, result.confidence))
        self._samples_seen += 1

    def _mark_warmup_frame(self, frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        if self._warmup_start_frame is None:
            self._warmup_start_frame = frame_idx
        self._latest_frame_idx = frame_idx if self._latest_frame_idx is None else max(self._latest_frame_idx, frame_idx)

    def _warmup_frame_ready(self) -> bool:
        if self._warmup_start_frame is None or self._latest_frame_idx is None:
            return False
        return (self._latest_frame_idx - self._warmup_start_frame + 1) >= self.warmup_frames

    def _store_track_position(self, track_id: int, bbox: Sequence[float], frame_width: int | float) -> None:
        x1, _, x2, _ = [float(value) for value in bbox]
        self._track_x_positions.setdefault(track_id, []).append((x1 + x2) / 2.0)
        self._track_frame_widths.setdefault(track_id, []).append(float(frame_width))

    def _build_cluster_to_team(self, labels: Sequence[int], track_ids: Sequence[int]) -> dict[int, str]:
        team_a_cluster = self._team_a_cluster_from_screen_counts(labels, track_ids)
        clusters = [int(cluster) for cluster in sorted(set(labels))]
        if team_a_cluster is None:
            side_order = self._cluster_order_from_screen_positions(labels, track_ids)
            if side_order is None:
                side_order = [int(cluster) for cluster in np.argsort(self._centroids[:, 0])]
            team_a_cluster = int(side_order[-1] if self.team_a_side == "right" else side_order[0])

        team_b_cluster = next(cluster for cluster in clusters if cluster != team_a_cluster)
        return {int(team_a_cluster): "team_a", int(team_b_cluster): "team_b"}

    def _team_a_cluster_from_screen_counts(
        self,
        labels: Sequence[int],
        track_ids: Sequence[int],
    ) -> int | None:
        cluster_counts: dict[int, dict[str, int]] = {}
        cluster_avg_x: dict[int, list[float]] = {}
        for track_id, cluster in zip(track_ids, labels):
            positions = self._track_x_positions.get(int(track_id), [])
            widths = self._track_frame_widths.get(int(track_id), [])
            if not positions or not widths or any(not math.isfinite(width) for width in widths):
                continue

            left_votes = sum(1 for x, width in zip(positions, widths) if x < width / 2.0)
            right_votes = len(positions) - left_votes
            side = "left" if left_votes >= right_votes else "right"
            cluster = int(cluster)
            cluster_counts.setdefault(cluster, {"left": 0, "right": 0})[side] += 1
            cluster_avg_x.setdefault(cluster, []).extend(positions)

        if len(cluster_counts) < 2:
            return None

        side = "right" if self.team_a_side == "right" else "left"
        reverse_x = side == "right"
        return max(
            cluster_counts,
            key=lambda cluster: (
                cluster_counts[cluster][side],
                np.mean(cluster_avg_x.get(cluster, [0.0])) * (1 if reverse_x else -1),
            ),
        )

    def _cluster_order_from_screen_positions(
        self,
        labels: Sequence[int],
        track_ids: Sequence[int],
    ) -> list[int] | None:
        cluster_positions: dict[int, list[float]] = {}
        for track_id, cluster in zip(track_ids, labels):
            positions = self._track_x_positions.get(int(track_id), [])
            if not positions:
                continue
            cluster_positions.setdefault(int(cluster), []).extend(positions)

        if len(cluster_positions) < 2:
            return None

        return [
            cluster
            for cluster, _ in sorted(
                (
                    (cluster, float(np.mean(positions)))
                    for cluster, positions in cluster_positions.items()
                ),
                key=lambda item: item[1],
            )
        ]

    def _compose_prediction(
        self,
        embedding: np.ndarray,
        role_label: str,
        confidence: float,
        crop: np.ndarray | None,
    ) -> tuple[str, str, float]:
        normalized_role = _normalize_role(role_label)
        if crop is not None and self.prefer_kit_color_classification:
            return self._color_fallback(crop, "player")

        confident_role = confidence >= self.role_confidence_threshold
        if normalized_role == "referee" and confident_role:
            return "referee", "referee", max(confidence, self.confidence_threshold)

        color_role = normalized_role if confident_role else "player"
        if confidence < self.confidence_threshold and crop is not None:
            return self._color_fallback(crop, color_role)

        team_label, team_confidence = self._team_from_embedding(embedding)
        if team_label == "unknown" and crop is not None:
            return self._color_fallback(crop, color_role)

        if normalized_role == "goalkeeper" and confident_role and team_label in {"team_a", "team_b"}:
            team_label = f"{team_label}_gk"
        return team_label, normalized_role, max(confidence, team_confidence)

    def _team_from_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        if self._centroids is None:
            return "unknown", 0.0
        distances = np.linalg.norm(self._centroids - embedding.reshape(1, -1), axis=1)
        cluster = int(np.argmin(distances))
        label = self._cluster_to_team.get(cluster, "unknown")
        total = float(np.sum(distances))
        if not math.isfinite(total) or total == 0.0:
            confidence = 1.0
        else:
            confidence = 1.0 - float(distances[cluster] / total)
        return label, max(0.0, min(1.0, confidence))

    def _color_fallback(
        self,
        crop: np.ndarray,
        role_label: str,
    ) -> tuple[str, str, float]:
        mean_bgr = self._extract_kit_color_bgr(crop)
        candidates = {
            "team_a": self.config.get("team_a_color_bgr", [204, 153, 0]),
            "team_b": self.config.get("team_b_color_bgr", [235, 233, 223]),
            "team_a_gk": self.config.get("team_a_gk_color_bgr", [0, 215, 255]),
            "team_b_gk": self.config.get("team_b_gk_color_bgr", [128, 0, 128]),
            "referee": self.config.get("referee_color_bgr", [0, 0, 0]),
        }
        distances = {
            key: float(np.linalg.norm(mean_bgr - np.asarray(value, dtype=float)))
            for key, value in candidates.items()
        }
        label = self._select_color_label(distances, mean_bgr)
        if label == "referee":
            return "referee", "referee", self.confidence_threshold
        if label.endswith("_gk"):
            return label, "goalkeeper", self.confidence_threshold
        if role_label == "goalkeeper":
            label = f"{label}_gk" if label in {"team_a", "team_b"} else label
        return label, role_label, self.confidence_threshold

    def _select_color_label(self, distances: dict[str, float], mean_bgr: np.ndarray) -> str:
        player_labels = ("team_a", "team_b")
        player_label = min(player_labels, key=lambda key: distances[key])
        player_distance = distances[player_label]
        if self._is_dark_referee_color(mean_bgr, distances):
            return "referee"
        best_label = min(distances, key=distances.get)
        if best_label in player_labels:
            return best_label

        special_distance = distances[best_label]
        decisive = (
            special_distance <= self.special_kit_color_max_distance
            and special_distance <= player_distance * self.special_kit_color_margin
        )
        return best_label if decisive else player_label

    def _is_dark_referee_color(self, mean_bgr: np.ndarray, distances: dict[str, float]) -> bool:
        referee_color = np.asarray(self.config.get("referee_color_bgr", [0, 0, 0]), dtype=np.float64)
        if float(np.max(referee_color)) > 50.0:
            return False
        if float(np.max(mean_bgr)) > self.referee_dark_max_channel:
            return False
        return distances.get("referee", float("inf")) <= self.referee_dark_max_distance

    @classmethod
    def _extract_kit_color_bgr(cls, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(3, dtype=np.float64)
        height, width = crop.shape[:2]
        y1 = int(round(height * 0.06))
        y2 = max(y1 + 1, int(round(height * 0.72)))
        x1 = int(round(width * 0.16))
        x2 = max(x1 + 1, int(round(width * 0.84)))
        torso = crop[y1:y2, x1:x2]
        pixels = cls._non_green_pixels(torso)
        min_pixels = max(12, int(torso.shape[0] * torso.shape[1] * 0.08))
        if len(pixels) < min_pixels:
            pixels = cls._non_green_pixels(crop)
        if len(pixels) == 0:
            pixels = torso.reshape(-1, 3)
        return np.median(pixels.astype(np.float64), axis=0)

    @classmethod
    def _non_green_pixels(cls, image: np.ndarray) -> np.ndarray:
        pixels = image.reshape(-1, 3)
        if len(pixels) == 0:
            return pixels
        mask = ~cls._green_mask(image).reshape(-1)
        brightness = pixels.max(axis=1) > 35
        return pixels[mask & brightness]

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        return (g > 55) & (g > r * 1.12) & (g > b * 1.03) & ((g - r) > 18)

    def _majority_role(self, track_id: int) -> tuple[str, float]:
        roles = self._track_roles.get(track_id, [])
        if not roles:
            return "unknown", 0.0
        best_role = max(
            {role for role, _ in roles},
            key=lambda role: sum(conf for candidate, conf in roles if candidate == role),
        )
        confidences = [conf for role, conf in roles if role == best_role]
        return _normalize_role(best_role), float(np.mean(confidences))


class TeamClassifier(PRTReidClassifier):
    pass


def _normalize_role(role_label: str | None) -> str:
    if role_label in {"player", "goalkeeper", "referee"}:
        return role_label
    return "unknown" if role_label in {None, "ball", "other"} else str(role_label)


def _download_if_missing(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)
