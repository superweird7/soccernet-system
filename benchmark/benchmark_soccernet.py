from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.team_classifier import PRTReidClassifier
from core.homography import BroadcastCalibrator


PERSON_ROLES = {"player", "goalkeeper", "referee"}


def role_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(pred == gt for pred, gt in zip(predictions, ground_truth))
    return correct / len(ground_truth)


def pitch_point_cm_from_annotation(annotation: dict) -> np.ndarray:
    pitch = annotation["bbox_pitch"]
    return np.asarray(
        [
            (float(pitch["x_bottom_middle"]) + 52.5) * 100.0,
            (float(pitch["y_bottom_middle"]) + 34.0) * 100.0,
        ],
        dtype=np.float64,
    )


def image_foot_point_from_annotation(annotation: dict) -> np.ndarray:
    bbox = annotation["bbox_image"]
    return np.asarray(
        [
            float(bbox["x"]) + float(bbox["w"]) / 2.0,
            float(bbox["y"]) + float(bbox["h"]),
        ],
        dtype=np.float64,
    )


def mean_position_error_m(predicted_cm: np.ndarray, truth_cm: np.ndarray) -> float:
    if len(predicted_cm) == 0:
        return float("inf")
    distances_cm = np.linalg.norm(
        np.asarray(predicted_cm, dtype=np.float64) - np.asarray(truth_cm, dtype=np.float64),
        axis=1,
    )
    return round(float(distances_cm.mean() / 100.0), 4)


def best_binary_team_accuracy(
    predictions: list[str],
    ground_truth: list[str],
) -> tuple[float, dict[str, str]]:
    pairs = [
        (pred, gt)
        for pred, gt in zip(predictions, ground_truth)
        if pred in {"team_a", "team_b"} and gt in {"left", "right"}
    ]
    if not pairs:
        return 0.0, {}

    best_accuracy = -1.0
    best_mapping: dict[str, str] = {}
    for left_label, right_label in itertools.permutations(["left", "right"]):
        mapping = {"team_a": left_label, "team_b": right_label}
        correct = sum(mapping[pred] == gt for pred, gt in pairs)
        accuracy = correct / len(pairs)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
    return best_accuracy, best_mapping


class _UnavailableTVCalibBackend:
    def estimate(self, frame):
        raise RuntimeError("TVCalib unavailable; using manual SoccerNet pitch-line JSON")


def normalization_transform(points: list[np.ndarray]) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64)
    center = np.mean(points_array, axis=0)
    mean_distance = float(np.mean(np.linalg.norm(points_array - center, axis=1)))
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 0 else 1.0
    return np.asarray(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_homography_from_line_correspondences(
    lines: list[tuple[np.ndarray, np.ndarray]],
    pitch_normalization: np.ndarray,
    image_normalization: np.ndarray,
) -> np.ndarray:
    A = np.zeros((len(lines) * 2, 9), dtype=np.float64)
    for index, (pitch_line_raw, image_line_raw) in enumerate(lines):
        pitch_line = np.linalg.inv(pitch_normalization).T @ pitch_line_raw
        image_line = np.linalg.inv(image_normalization).T @ image_line_raw
        u, v, w = pitch_line
        x, y, z = image_line
        A[2 * index] = [0.0, x * w, -x * v, 0.0, y * w, -v * y, 0.0, z * w, -v * z]
        A[2 * index + 1] = [x * w, 0.0, -x * u, y * w, 0.0, -u * y, z * w, 0.0, -u * z]

    _, singular_values, vh = np.linalg.svd(A)
    homography = None
    for index in range(singular_values.shape[0] - 1, -2, -1):
        candidate = vh[index].reshape(3, 3)
        if singular_values[index] > 0:
            homography = candidate
            break
    if homography is None:
        raise RuntimeError("Could not estimate homography from line correspondences")

    homography = np.linalg.inv(image_normalization) @ homography @ pitch_normalization
    return homography / homography[2, 2]


def fit_homography_from_pitch_annotation(pitch_annotation: dict, image_info: dict) -> np.ndarray:
    from SoccerNet.Evaluation.utils_calibration import SoccerPitch

    field = SoccerPitch()
    line_matches = []
    pitch_points = []
    image_points = []
    width = float(image_info["width"])
    height = float(image_info["height"])

    for line_name, points in pitch_annotation.get("lines", {}).items():
        if len(points) < 2 or line_name not in field.line_extremities_keys:
            continue
        pitch_line = field.get_2d_homogeneous_line(line_name)
        if pitch_line is None:
            continue

        p1 = np.asarray([points[0]["x"] * width, points[0]["y"] * height, 1.0])
        p2 = np.asarray([points[-1]["x"] * width, points[-1]["y"] * height, 1.0])
        image_line = np.cross(p1, p2)
        if not np.isfinite(image_line).all():
            continue

        line_matches.append((np.asarray(pitch_line, dtype=np.float64), image_line))
        image_points.extend([p1, p2])
        for pitch_key in field.line_extremities_keys[line_name]:
            pitch_points.append(field.point_dict[pitch_key][:2])

    if len(line_matches) < 4:
        raise RuntimeError(f"Need at least 4 pitch lines, got {len(line_matches)}")

    pitch_to_image_m = estimate_homography_from_line_correspondences(
        line_matches,
        normalization_transform(pitch_points),
        normalization_transform(image_points),
    )
    image_to_pitch_m = np.linalg.inv(pitch_to_image_m)
    centered_m_to_project_cm = np.asarray(
        [[100.0, 0.0, 5250.0], [0.0, 100.0, 3400.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H = centered_m_to_project_cm @ image_to_pitch_m
    return H / H[2, 2]


def ensure_sequence_manual_calibration(
    labels: dict,
    image_by_id: dict,
    manual_path: Path,
) -> Path:
    if manual_path.exists():
        return manual_path

    pitch_annotations = [
        ann
        for ann in labels["annotations"]
        if ann.get("supercategory") == "pitch" and ann.get("image_id") in image_by_id
    ]
    if not pitch_annotations:
        raise RuntimeError("No SoccerNet pitch-line annotations available for manual calibration")

    pitch_annotation = pitch_annotations[0]
    image_info = image_by_id[pitch_annotation["image_id"]]
    homography = fit_homography_from_pitch_annotation(pitch_annotation, image_info)
    manual_path.parent.mkdir(parents=True, exist_ok=True)
    manual_path.write_text(
        json.dumps(
            {
                "homography": homography.tolist(),
                "status": "manual",
                "source": "soccernet_pitch_lines",
                "image_id": pitch_annotation["image_id"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manual_path


def run_sequence(sequence: str, frames: int, project_root: Path) -> dict:
    config = yaml.safe_load((project_root / "config.yaml").read_text(encoding="utf-8"))
    sequence_dir = project_root / "datasets" / "soccernet" / sequence
    labels_path = sequence_dir / "Labels-GameState.json"
    with labels_path.open("r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    image_dir = sequence_dir / labels.get("info", {}).get("im_dir", "img1")

    image_by_id = {
        image["image_id"]: image
        for image in labels["images"][:frames]
    }
    annotations = [
        ann
        for ann in labels["annotations"]
        if ann.get("image_id") in image_by_id
        and ann.get("attributes", {}).get("role") in PERSON_ROLES
        and ann.get("bbox_image")
    ]
    manual_path = project_root / config.get("calibration_save_dir", "calibration/saved/") / f"{sequence}.json"
    ensure_sequence_manual_calibration(labels, image_by_id, manual_path)
    calibrator = BroadcastCalibrator(
        config,
        device=config.get("device", "cuda"),
        backend=_UnavailableTVCalibBackend(),
        manual_path=manual_path,
    )

    classifier = PRTReidClassifier(config, device=config.get("device", "cuda"), async_enabled=False)

    crops = []
    track_ids = []
    rows = []
    frame_cache = {}
    for ann in annotations:
        image_info = image_by_id[ann["image_id"]]
        frame_name = image_info["file_name"]
        if frame_name not in frame_cache:
            frame_path = image_dir / image_info["file_name"]
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not load frame: {frame_path}")
            frame_cache[frame_name] = frame
            if not calibrator.is_calibrated():
                calibrator.update(frame, frame_idx=1)
                calibrator.flush(timeout=5)

        bbox = ann["bbox_image"]
        xyxy = (
            bbox["x"],
            bbox["y"],
            bbox["x"] + bbox["w"],
            bbox["y"] + bbox["h"],
        )
        crop = classifier.crop_player_region(frame_cache[frame_name], xyxy)
        crops.append(crop)
        track_ids.append(int(ann["track_id"]))
        rows.append(ann)

    classifier.warm_up(crops, track_ids)

    role_predictions = []
    role_truth = []
    team_predictions = []
    team_truth = []
    for ann in rows:
        predicted_team, predicted_role, _ = classifier._cache.get(
            int(ann["track_id"]),
            ("unknown", "unknown", 0.0),
        )
        role_gt = ann["attributes"]["role"]
        role_predictions.append(predicted_role)
        role_truth.append(role_gt)

        team_gt = ann["attributes"].get("team")
        team_pred = strip_goalkeeper_suffix(predicted_team)
        if role_gt in {"player", "goalkeeper"} and team_gt in {"left", "right"}:
            team_predictions.append(team_pred)
            team_truth.append(team_gt)

    team_acc, team_mapping = best_binary_team_accuracy(team_predictions, team_truth)
    role_acc = role_accuracy(role_predictions, role_truth)

    predicted_positions = []
    truth_positions = []
    for ann in rows:
        transformed = calibrator.transform_points(image_foot_point_from_annotation(ann).reshape(1, 2))
        if transformed is None or len(transformed) != 1:
            continue
        predicted_positions.append(transformed[0])
        truth_positions.append(pitch_point_cm_from_annotation(ann))
    position_error = mean_position_error_m(
        np.asarray(predicted_positions, dtype=np.float64),
        np.asarray(truth_positions, dtype=np.float64),
    )
    return {
        "sequence": sequence,
        "frames": frames,
        "annotations": len(rows),
        "team_eval_count": len(team_truth),
        "team_accuracy": team_acc,
        "team_mapping": team_mapping,
        "role_accuracy": role_acc,
        "position_eval_count": len(predicted_positions),
        "mean_position_error_m": position_error,
        "calibration_status": calibrator.get_status(),
        "calibration_manual_path": str(manual_path),
        "prtreid_backend_error": getattr(classifier.backend, "disabled_error", None),
    }


def strip_goalkeeper_suffix(label: str) -> str:
    if label == "team_a_gk":
        return "team_a"
    if label == "team_b_gk":
        return "team_b"
    return label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    result = run_sequence(args.sequence, args.frames, project_root)
    print(f"sequence={result['sequence']}")
    print(f"frames={result['frames']}")
    print(f"annotations={result['annotations']}")
    print(f"team_eval_count={result['team_eval_count']}")
    print(f"team_accuracy={result['team_accuracy'] * 100:.2f}%")
    print(f"role_accuracy={result['role_accuracy'] * 100:.2f}%")
    print(f"position_eval_count={result['position_eval_count']}")
    print(f"mean_position_error_m={result['mean_position_error_m']:.4f}")
    print(f"calibration_status={result['calibration_status']}")
    print(f"team_mapping={result['team_mapping']}")
    if result["prtreid_backend_error"]:
        print(f"prtreid_backend_error={result['prtreid_backend_error']}")

    if (
        result["team_accuracy"] <= 0.85
        or result["role_accuracy"] <= 0.85
        or result["mean_position_error_m"] >= 1.5
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
