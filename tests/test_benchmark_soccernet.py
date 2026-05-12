import numpy as np

from benchmark.benchmark_soccernet import (
    best_binary_team_accuracy,
    mean_position_error_m,
    pitch_point_cm_from_annotation,
    role_accuracy,
)


def test_best_binary_team_accuracy_allows_cluster_permutation():
    predictions = ["team_b", "team_b", "team_a", "team_a"]
    ground_truth = ["left", "left", "right", "right"]

    accuracy, mapping = best_binary_team_accuracy(predictions, ground_truth)

    assert accuracy == 1.0
    assert mapping == {"team_b": "left", "team_a": "right"}


def test_role_accuracy_counts_matching_roles():
    predictions = ["player", "goalkeeper", "player", "referee"]
    ground_truth = ["player", "player", "goalkeeper", "referee"]

    assert role_accuracy(predictions, ground_truth) == 0.5


def test_pitch_point_cm_from_annotation_converts_centered_soccernet_meters():
    annotation = {
        "bbox_pitch": {
            "x_bottom_middle": 0.0,
            "y_bottom_middle": 0.0,
        }
    }

    np.testing.assert_allclose(
        pitch_point_cm_from_annotation(annotation),
        np.asarray([5250.0, 3400.0]),
    )


def test_mean_position_error_m_reports_euclidean_cm_error_as_meters():
    predicted_cm = np.asarray([[100.0, 100.0], [200.0, 200.0]])
    truth_cm = np.asarray([[100.0, 100.0], [500.0, 600.0]])

    assert mean_position_error_m(predicted_cm, truth_cm) == 2.5
