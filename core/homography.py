from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import requests


class HomographyBackend(Protocol):
    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        ...


class TVCalibBackend:
    """Thin TVCalib integration point.

    TVCalib needs a line-localization model and a line-to-camera decoding step.
    This backend validates the installed package and keeps the weight artifact in
    the expected models directory; callers fall back to manual JSON when it
    cannot produce a homography for a frame.
    """

    WEIGHTS_URL = "https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt"

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        download_timeout: float = 15.0,
        auto_download: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.download_timeout = float(download_timeout)
        self.auto_download = bool(auto_download)
        self.weights_path = self.model_dir / "train_59.pt"

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_weights()
        try:
            import tvcalib.inference  # noqa: F401
            import tvcalib.module  # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised in integration only
            raise RuntimeError(f"TVCalib import failed: {exc}") from exc
        raise RuntimeError(
            "TVCalib is installed, but automatic line decoding did not return a homography"
        )

    def _ensure_weights(self) -> None:
        if self.weights_path.exists():
            return
        if not self.auto_download:
            raise RuntimeError(
                f"TVCalib weights missing: {self.weights_path}. "
                "Set tvcalib_auto_download_weights=true to retry automatic download."
            )
        tmp_path = self.weights_path.with_suffix(".tmp")
        try:
            with requests.get(
                self.WEIGHTS_URL,
                stream=True,
                timeout=(3.0, self.download_timeout),
            ) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"TVCalib weights download failed: {exc}") from exc
        if tmp_path.stat().st_size == 0:
            raise RuntimeError("TVCalib weights download failed: empty response")
        tmp_path.replace(self.weights_path)


class BroadcastCalibrator:
    VALID_STATUSES = {"tvcalib", "manual", "unavailable"}

    def __init__(
        self,
        config: dict,
        device: str = "cuda",
        backend: HomographyBackend | None = None,
        manual_path: str | Path | None = None,
    ) -> None:
        self.config = dict(config)
        self.device = device
        self.pitch_width_cm = float(config.get("pitch_width_cm", 10500))
        self.pitch_height_cm = float(config.get("pitch_height_cm", 6800))
        self.boundary_margin_cm = float(config.get("pitch_boundary_margin_cm", 200))
        self.update_every_n_frames = int(config.get("tvcalib_update_every_n_frames", 30))
        self.resize_enabled = bool(config.get("tvcalib_resize_enabled", True))
        self.tvcalib_input_width = int(config.get("tvcalib_input_width", 1280))
        self.tvcalib_input_height = int(config.get("tvcalib_input_height", 720))
        self.source_id = self._normalize_source_id(config.get("calibration_source_id", ""))
        configured_manual_path = config.get("manual_calibration_path") or config.get("calibration_manual_path")
        manual_value = manual_path if manual_path is not None else configured_manual_path
        self._manual_path_explicit = manual_value is not None
        self.manual_path = (
            self._resolve_config_path(manual_value)
            if manual_value is not None
            else self._default_manual_path()
        )
        self.backend = backend if backend is not None else TVCalibBackend(
            config.get("tvcalib_model_path", "models/tvcalib/"),
            device=device,
            download_timeout=float(config.get("tvcalib_download_timeout_sec", 15.0)),
            auto_download=bool(config.get("tvcalib_auto_download_weights", False)),
        )

        self._lock = threading.Lock()
        self._H: np.ndarray | None = None
        self._status = "unavailable"
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._last_started_frame: int | None = None

        if self.config.get("calibration_mode") == "manual":
            for candidate in self._manual_calibration_candidates():
                self.load_calibration(candidate)
                break

    def update(self, frame: np.ndarray, frame_idx: int) -> None:
        if int(frame_idx) == 0 and self._last_started_frame is None:
            self._last_started_frame = 0
            self._estimate_frame(frame.copy())
            return

        if not self._should_start(frame_idx):
            return

        frame_copy = frame.copy()
        self._last_started_frame = int(frame_idx)
        self._thread = threading.Thread(
            target=self._estimate_in_background,
            args=(frame_copy,),
            daemon=True,
        )
        self._thread.start()

    def transform_points(self, points: np.ndarray) -> np.ndarray | None:
        with self._lock:
            H = None if self._H is None else self._H.copy()
        if H is None:
            return None

        points_array = np.asarray(points, dtype=np.float64)
        if points_array.size == 0:
            return points_array.reshape(0, 2)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError("points must be an Nx2 array")

        transformed = cv2.perspectiveTransform(points_array.reshape(-1, 1, 2), H)
        transformed = transformed.reshape(-1, 2)
        margin = self.boundary_margin_cm
        inside = (
            (transformed[:, 0] >= -margin)
            & (transformed[:, 0] <= self.pitch_width_cm + margin)
            & (transformed[:, 1] >= -margin)
            & (transformed[:, 1] <= self.pitch_height_cm + margin)
        )
        return transformed[inside]

    def is_calibrated(self) -> bool:
        with self._lock:
            return self._H is not None

    def get_status(self) -> str:
        with self._lock:
            return self._status

    def get_homography(self) -> np.ndarray | None:
        with self._lock:
            return None if self._H is None else self._H.copy()

    def get_last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def save_calibration(self, path: str | Path) -> None:
        with self._lock:
            if self._H is None:
                raise RuntimeError("No homography available to save")
            homography = self._H.tolist()
            status = self._status
        payload = {
            "homography": homography,
            "status": status,
            "pitch_width_cm": self.pitch_width_cm,
            "pitch_height_cm": self.pitch_height_cm,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_calibration(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        H = self._validate_homography(payload["homography"])
        with self._lock:
            self._H = H
            self._status = "manual"
            self._last_error = None

    def flush(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _default_manual_path(self) -> Path:
        save_dir = Path(self.config.get("calibration_save_dir", "calibration/saved/"))
        project_root = self.config.get("project_root")
        if project_root is not None and not save_dir.is_absolute():
            save_dir = Path(project_root) / save_dir
        return save_dir / "default.json"

    def _resolve_config_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        project_root = self.config.get("project_root")
        return Path(project_root) / path if project_root is not None else path

    def _should_start(self, frame_idx: int) -> bool:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return False
        if self._last_started_frame is None:
            return True
        return int(frame_idx) - self._last_started_frame >= self.update_every_n_frames

    def _estimate_in_background(self, frame: np.ndarray) -> None:
        self._estimate_frame(frame)

    def _estimate_frame(self, frame: np.ndarray) -> None:
        try:
            tvcalib_frame, original_to_tvcalib = self._prepare_tvcalib_frame(frame)
            H = self.backend.estimate(tvcalib_frame)
            if H is None:
                raise RuntimeError("TVCalib returned no homography")
            H = self._validate_homography(H)
            H = self._validate_homography(H @ original_to_tvcalib)
        except Exception as exc:
            self._handle_tvcalib_failure(exc)
            return

        with self._lock:
            self._H = H
            self._status = "tvcalib"
            self._last_error = None

    def _handle_tvcalib_failure(self, exc: Exception) -> None:
        manual_error: Exception | None = None
        for manual_path in self._manual_calibration_candidates():
            try:
                self.load_calibration(manual_path)
                with self._lock:
                    self._last_error = str(exc)
                return
            except Exception as manual_exc:
                manual_error = manual_exc

        error = (
            f"{exc}; manual fallback failed: {manual_error}"
            if manual_error is not None
            else str(exc)
        )

        with self._lock:
            if self._H is None:
                self._status = "unavailable"
            self._last_error = error

    def _manual_calibration_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        saved_dir = self.manual_path.parent

        def add(path: Path) -> None:
            if path.exists() and path not in candidates:
                candidates.append(path)

        if self._manual_path_explicit:
            add(self.manual_path)

        if self.source_id:
            for stem in self._source_id_file_stems():
                add(saved_dir / f"{stem}.json")
        elif not self._manual_path_explicit:
            add(self.manual_path)
        return candidates

    def _source_id_file_stems(self) -> list[str]:
        stems: list[str] = []
        for stem in [self.source_id, self._safe_filename_stem(self.source_id)]:
            if stem and stem not in stems:
                stems.append(stem)
        return stems

    @staticmethod
    def _normalize_source_id(value: object) -> str:
        source_id = str(value or "").strip()
        if not source_id:
            return ""
        path = Path(source_id)
        if len(path.parts) > 1 or path.suffix:
            return path.stem
        return path.name

    @staticmethod
    def _safe_filename_stem(value: str) -> str:
        invalid = '<>:"/\\|?*'
        return "".join("_" if char in invalid else char for char in value).strip()

    def _prepare_tvcalib_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy ndarray")

        height, width = frame.shape[:2]
        if (
            not self.resize_enabled
            or (width == self.tvcalib_input_width and height == self.tvcalib_input_height)
        ):
            return frame.copy(), np.eye(3, dtype=np.float64)

        scale_x = self.tvcalib_input_width / float(width)
        scale_y = self.tvcalib_input_height / float(height)
        resized = cv2.resize(
            frame,
            (self.tvcalib_input_width, self.tvcalib_input_height),
            interpolation=cv2.INTER_AREA,
        )
        original_to_tvcalib = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return resized, original_to_tvcalib

    @staticmethod
    def _validate_homography(homography: object) -> np.ndarray:
        H = np.asarray(homography, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError("homography must be a 3x3 matrix")
        if not np.isfinite(H).all():
            raise ValueError("homography contains non-finite values")
        if abs(float(H[2, 2])) > 1e-12:
            H = H / H[2, 2]
        return H
