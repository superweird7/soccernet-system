import cv2
import numpy as np


def annotator_config(**overrides):
    config = {
        "team_a_color_bgr": [204, 153, 0],
        "team_b_color_bgr": [255, 255, 255],
        "team_a_gk_color_bgr": [0, 215, 255],
        "team_b_gk_color_bgr": [128, 0, 128],
        "referee_color_bgr": [0, 0, 0],
    }
    config.update(overrides)
    return config


def player(track_id, bbox, team_label="team_a", role_label="player"):
    return {
        "track_id": track_id,
        "bbox": bbox,
        "team_label": team_label,
        "role_label": role_label,
    }


def test_annotator_draws_on_copy_with_rectangle_and_feet_ellipse():
    from core.annotator import FrameAnnotator

    frame = np.full((220, 320, 3), 40, dtype=np.uint8)
    original = frame.copy()
    annotator = FrameAnnotator(annotator_config())

    annotated = annotator.annotate(frame, [player(23, (50, 60, 100, 160), "team_a")])

    np.testing.assert_array_equal(frame, original)
    assert annotated.shape == frame.shape
    assert tuple(annotated[60, 50]) == (204, 153, 0)
    assert tuple(annotated[160, 75]) != tuple(original[160, 75])


def test_annotator_formats_player_labels():
    from core.annotator import FrameAnnotator

    annotator = FrameAnnotator(annotator_config())

    assert annotator.format_player_label(player(23, (0, 0, 10, 10), "team_a")) == "A 23"
    assert annotator.format_player_label(player(7, (0, 0, 10, 10), "team_b_gk", "goalkeeper")) == "B GK 7"
    assert annotator.format_player_label(player(5, (0, 0, 10, 10), "referee", "referee")) == "REF"


def test_annotator_draws_yellow_downward_ball_triangle_above_bbox():
    from core.annotator import FrameAnnotator

    frame = np.full((220, 320, 3), 40, dtype=np.uint8)
    annotator = FrameAnnotator(annotator_config())
    ball = {"bbox": (150, 120, 170, 140), "label": "ball"}

    annotated = annotator.annotate(frame, [], ball=ball)

    assert tuple(annotated[112, 160]) == (0, 255, 255)


def test_annotator_draws_calibration_status_colors():
    from core.annotator import FrameAnnotator

    annotator = FrameAnnotator(annotator_config())
    frame = np.zeros((220, 320, 3), dtype=np.uint8)

    tvcalib = annotator.annotate(frame, [], cal_status="tvcalib")
    manual = annotator.annotate(frame, [], cal_status="manual")
    unavailable = annotator.annotate(frame, [], cal_status="unavailable")

    assert tuple(tvcalib[14, 14]) == (0, 150, 0)
    assert tuple(manual[14, 14]) == (0, 165, 255)
    assert tuple(unavailable[14, 14]) == (0, 0, 220)


def test_annotator_draws_kit_clash_warning_when_team_colors_are_close():
    from core.annotator import FrameAnnotator

    frame = np.full((220, 320, 3), 40, dtype=np.uint8)
    base = FrameAnnotator(annotator_config()).annotate(frame, [])
    clash = FrameAnnotator(
        annotator_config(
            team_a_color_bgr=[100, 100, 100],
            team_b_color_bgr=[120, 120, 120],
        )
    ).annotate(frame, [])

    assert np.count_nonzero(cv2.absdiff(base[:, 190:320], clash[:, 190:320])) > 0


def test_annotator_accepts_object_players_and_ball():
    from core.annotator import FrameAnnotator

    class Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    frame = np.full((220, 320, 3), 40, dtype=np.uint8)
    player_obj = Obj(track_id=9, bbox=(20, 60, 60, 140), team_label="team_b", role_label="player")
    ball_obj = Obj(bbox=(80, 80, 90, 90), label="ball")

    annotated = FrameAnnotator(annotator_config()).annotate(frame, [player_obj], ball=ball_obj)

    assert annotated.shape == frame.shape
    assert tuple(annotated[60, 20]) == (255, 255, 255)
    assert tuple(annotated[72, 85]) == (0, 255, 255)
