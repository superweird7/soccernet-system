import time

import numpy as np


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def extract(self, crop):
        self.calls += 1
        mean_blue = float(crop[:, :, 0].mean())
        mean_green = float(crop[:, :, 1].mean())
        embedding = np.asarray([mean_blue, mean_green], dtype=np.float32)
        return embedding, "player", 0.92


class SlowBackend(FakeBackend):
    def extract(self, crop):
        time.sleep(0.2)
        return super().extract(crop)


class LowConfidenceBackend:
    def extract(self, crop):
        return np.asarray([1.0, 0.0], dtype=np.float32), "player", 0.2


def solid_crop(bgr):
    crop = np.zeros((80, 40, 3), dtype=np.uint8)
    crop[:, :] = bgr
    return crop


def side_test_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[10:90, 10:40] = (220, 20, 20)
    frame[10:90, 150:180] = (20, 220, 20)
    return frame


def draw_player(frame, bbox, bgr):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    frame[y1:y2, x1:x2] = bgr


def test_crop_player_region_uses_top_70_percent_of_bbox():
    from core.team_classifier import PRTReidClassifier

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:90, 10:50] = (10, 20, 30)

    crop = PRTReidClassifier.crop_player_region(frame, (10, 20, 50, 90))

    assert crop.shape == (49, 40, 3)
    assert tuple(crop[0, 0]) == (10, 20, 30)


def test_default_warmup_is_120_frames_and_team_a_side_defaults_left(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {"prtreid_model_path": str(tmp_path)},
        backend=FakeBackend(),
        async_enabled=False,
    )

    assert classifier.warmup_frames == 120
    assert classifier.team_a_side == "left"


def test_warmup_cluster_team_mapping_uses_average_screen_x_position(tmp_path):
    from core.team_classifier import PRTReidClassifier

    frame = side_test_frame()
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "prtreid_warmup_frames": 4,
            "team_a_side": "left",
        },
        backend=FakeBackend(),
        async_enabled=False,
    )

    for frame_idx in [0, 5]:
        classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, frame_idx)
        classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, frame_idx)

    assert classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, 20)[0] == "team_a"
    assert classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, 20)[0] == "team_b"


def test_team_a_mapping_uses_majority_player_count_on_left_half_not_average_x(tmp_path):
    from core.team_classifier import PRTReidClassifier

    frame = np.zeros((220, 200, 3), dtype=np.uint8)
    red_bboxes = [
        (80, 10, 100, 45),
        (80, 50, 100, 85),
        (180, 10, 200, 45),
    ]
    green_bboxes = [
        (0, 100, 20, 135),
        (140, 100, 160, 135),
    ]
    for bbox in red_bboxes:
        draw_player(frame, bbox, (220, 20, 20))
    for bbox in green_bboxes:
        draw_player(frame, bbox, (20, 220, 20))

    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "prtreid_warmup_frames": 999,
            "team_a_side": "left",
        },
        backend=FakeBackend(),
        async_enabled=False,
    )
    for track_id, bbox in enumerate(red_bboxes + green_bboxes, start=1):
        classifier.predict_from_frame(frame, bbox, track_id, 0)
    classifier.finalize_warmup(force=True)

    assert classifier.predict_from_frame(frame, red_bboxes[0], 1, 10)[0] == "team_a"
    assert classifier.predict_from_frame(frame, green_bboxes[0], 4, 10)[0] == "team_b"


def test_team_a_side_right_reverses_warmup_cluster_mapping(tmp_path):
    from core.team_classifier import PRTReidClassifier

    frame = side_test_frame()
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "prtreid_warmup_frames": 4,
            "team_a_side": "right",
        },
        backend=FakeBackend(),
        async_enabled=False,
    )

    for frame_idx in [0, 5]:
        classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, frame_idx)
        classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, frame_idx)

    assert classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, 20)[0] == "team_b"
    assert classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, 20)[0] == "team_a"


def test_warmup_threshold_is_based_on_video_frames_not_embedding_samples(tmp_path):
    from core.team_classifier import PRTReidClassifier

    frame = side_test_frame()
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "prtreid_warmup_frames": 120,
            "team_a_side": "left",
        },
        backend=FakeBackend(),
        async_enabled=False,
    )

    classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, 0)
    classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, 0)
    assert not classifier.is_warmed_up

    classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, 119)
    classifier.predict_from_frame(frame, (150, 10, 180, 90), 2, 119)

    assert classifier.is_warmed_up
    assert classifier.predict_from_frame(frame, (10, 10, 40, 90), 1, 125)[0] == "team_a"


def test_warm_up_clusters_track_embeddings_into_two_teams(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {"prtreid_model_path": str(tmp_path), "prtreid_warmup_frames": 4},
        backend=FakeBackend(),
        async_enabled=False,
    )
    crops = [
        solid_crop((220, 20, 20)),
        solid_crop((210, 30, 20)),
        solid_crop((20, 220, 20)),
        solid_crop((30, 210, 20)),
    ]
    track_ids = [1, 1, 2, 2]

    classifier.warm_up(crops, track_ids)

    assert classifier.is_warmed_up
    label_1, role_1, conf_1 = classifier.predict(solid_crop((220, 20, 20)), 1, 10)
    label_2, role_2, conf_2 = classifier.predict(solid_crop((20, 220, 20)), 2, 10)

    assert {label_1, label_2} == {"team_a", "team_b"}
    assert role_1 == "player"
    assert role_2 == "player"
    assert conf_1 > 0.5
    assert conf_2 > 0.5


def test_predict_returns_cached_result_and_updates_every_five_frames(tmp_path):
    from core.team_classifier import PRTReidClassifier

    backend = FakeBackend()
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "prtreid_warmup_frames": 2,
            "prtreid_update_every_n_frames": 5,
        },
        backend=backend,
        async_enabled=False,
    )
    classifier.warm_up(
        [solid_crop((220, 20, 20)), solid_crop((20, 220, 20))],
        [1, 2],
    )
    before = backend.calls

    first = classifier.predict(solid_crop((220, 20, 20)), 1, 10)
    second = classifier.predict(solid_crop((20, 220, 20)), 1, 14)
    third = classifier.predict(solid_crop((20, 220, 20)), 1, 15)

    assert second == first
    assert third != first
    assert backend.calls == before + 2


def test_predict_schedules_background_work_without_blocking(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {"prtreid_model_path": str(tmp_path), "prtreid_warmup_frames": 2},
        backend=SlowBackend(),
        async_enabled=True,
    )
    start = time.perf_counter()

    result = classifier.predict(solid_crop((220, 20, 20)), 1, 1)
    elapsed = time.perf_counter() - start
    classifier.flush(timeout=2)

    assert result == ("unknown", "unknown", 0.0)
    assert elapsed < 0.1


def test_low_prtreid_confidence_uses_color_fallback(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "team_a_color_bgr": [220, 20, 20],
            "team_b_color_bgr": [20, 220, 20],
            "prtreid_confidence_threshold": 0.5,
        },
        backend=LowConfidenceBackend(),
        async_enabled=False,
    )

    label, role, confidence = classifier.predict(solid_crop((20, 220, 20)), 9, 1)

    assert label == "team_b"
    assert role == "player"
    assert confidence == 0.5


def test_color_fallback_ignores_green_pitch_background_for_white_kits(tmp_path):
    from core.team_classifier import PRTReidClassifier

    crop = np.zeros((80, 44, 3), dtype=np.uint8)
    crop[:, :] = (75, 150, 85)
    crop[8:56, 14:30] = (238, 236, 226)
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "team_a_color_bgr": [204, 153, 0],
            "team_b_color_bgr": [235, 233, 223],
            "prtreid_confidence_threshold": 0.5,
        },
        backend=LowConfidenceBackend(),
        async_enabled=False,
    )

    label, role, confidence = classifier.predict(crop, 12, 1)

    assert label == "team_b"
    assert role == "player"
    assert confidence == 0.5


def test_prefer_kit_color_classification_overrides_warmed_embedding_cluster(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "team_a_color_bgr": [204, 153, 0],
            "team_b_color_bgr": [235, 233, 223],
            "prefer_kit_color_classification": True,
        },
        backend=FakeBackend(),
        async_enabled=False,
    )
    classifier._centroids = np.asarray([[0.0, 0.0], [255.0, 255.0]], dtype=np.float32)
    classifier._cluster_to_team = {0: "team_a", 1: "team_b"}
    crop = np.zeros((80, 44, 3), dtype=np.uint8)
    crop[:, :] = (75, 150, 85)
    crop[8:56, 14:30] = (238, 236, 226)

    label, role, confidence = classifier.predict(crop, 99, 10)

    assert label == "team_b"
    assert role == "player"
    assert confidence == 0.5


def test_ambiguous_dark_crop_does_not_become_goalkeeper_without_decisive_special_color(tmp_path):
    from core.team_classifier import PRTReidClassifier

    crop = solid_crop((121, 116, 87))
    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "team_a_color_bgr": [204, 153, 0],
            "team_b_color_bgr": [235, 233, 223],
            "team_a_gk_color_bgr": [0, 215, 255],
            "team_b_gk_color_bgr": [128, 0, 128],
            "referee_color_bgr": [0, 0, 0],
            "prefer_kit_color_classification": True,
        },
        backend=LowConfidenceBackend(),
        async_enabled=False,
    )

    label, role, _ = classifier.predict(crop, 77, 1)

    assert label == "team_a"
    assert role == "player"


def test_close_special_kit_color_still_labels_goalkeeper(tmp_path):
    from core.team_classifier import PRTReidClassifier

    classifier = PRTReidClassifier(
        {
            "prtreid_model_path": str(tmp_path),
            "team_a_color_bgr": [204, 153, 0],
            "team_b_color_bgr": [235, 233, 223],
            "team_b_gk_color_bgr": [128, 0, 128],
            "prefer_kit_color_classification": True,
        },
        backend=LowConfidenceBackend(),
        async_enabled=False,
    )

    label, role, _ = classifier.predict(solid_crop((128, 0, 128)), 78, 1)

    assert label == "team_b_gk"
    assert role == "goalkeeper"
