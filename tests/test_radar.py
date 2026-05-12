from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def radar_config(**overrides):
    config = {
        "pitch_width_cm": 10500,
        "pitch_height_cm": 6800,
        "radar_width": 800,
        "radar_height": 520,
        "position_smoothing_frames": 3,
        "show_tracker_ids": True,
    }
    config.update(overrides)
    return config


def player(track_id, x, y, team_label="unknown", role_label="player"):
    return {
        "track_id": track_id,
        "pitch_x": x,
        "pitch_y": y,
        "team_label": team_label,
        "role_label": role_label,
    }


def test_radar_maps_pitch_coordinates_to_template_pixels_and_draws_marker_types():
    from core.radar import RadarRenderer

    renderer = RadarRenderer(PROJECT_ROOT / "assets" / "pitch_template.png", radar_config(show_tracker_ids=False))
    players = [
        player(1, 1050, 680, "team_a", "player"),
        player(2, 2100, 1360, "team_a_gk", "goalkeeper"),
        player(3, 3150, 2040, "team_b", "player"),
        player(4, 4200, 2720, "team_b_gk", "goalkeeper"),
        player(5, 5250, 3400, "referee", "referee"),
        player(6, 6300, 4080, "unknown", "player"),
    ]

    radar = renderer.render(players, ball_pos=(7350, 4760), calibrated=True)

    assert radar.shape == (520, 800, 3)
    assert tuple(radar[52, 80]) == (204, 153, 0)
    assert tuple(radar[104, 160]) == (204, 153, 0)
    assert tuple(radar[156, 240]) == (255, 255, 255)
    assert tuple(radar[208, 320]) == (128, 0, 128)
    assert tuple(radar[260, 400]) == (0, 0, 0)
    assert tuple(radar[312, 480]) == (128, 128, 128)
    assert tuple(radar[364, 560]) == (0, 255, 255)


def test_radar_uses_configured_kit_colors_for_markers():
    from core.radar import RadarRenderer

    renderer = RadarRenderer(
        PROJECT_ROOT / "assets" / "pitch_template.png",
        radar_config(
            show_tracker_ids=False,
            team_a_color_bgr=[1, 2, 3],
            team_a_gk_color_bgr=[4, 5, 6],
            team_b_color_bgr=[7, 8, 9],
            team_b_gk_color_bgr=[10, 11, 12],
            referee_color_bgr=[13, 14, 15],
        ),
    )
    players = [
        player(1, 1050, 680, "team_a", "player"),
        player(2, 2100, 1360, "team_a_gk", "goalkeeper"),
        player(3, 3150, 2040, "team_b", "player"),
        player(4, 4200, 2720, "team_b_gk", "goalkeeper"),
        player(5, 5250, 3400, "referee", "referee"),
    ]

    radar = renderer.render(players, calibrated=True)

    assert tuple(radar[52, 80]) == (1, 2, 3)
    assert tuple(radar[104, 160]) == (4, 5, 6)
    assert tuple(radar[156, 240]) == (7, 8, 9)
    assert tuple(radar[208, 320]) == (10, 11, 12)
    assert tuple(radar[260, 400]) == (13, 14, 15)


def test_radar_smooths_positions_per_track_id_with_three_frame_average():
    from core.radar import RadarRenderer

    renderer = RadarRenderer(PROJECT_ROOT / "assets" / "pitch_template.png", radar_config(show_tracker_ids=False))

    renderer.render([player(1, 0, 0, "team_a")])
    renderer.render([player(1, 1050, 0, "team_a")])
    radar = renderer.render([player(1, 2100, 0, "team_a")])

    assert tuple(radar[0, 80]) == (204, 153, 0)


def test_radar_clamps_out_of_bounds_positions_to_pitch_edge():
    from core.radar import RadarRenderer

    renderer = RadarRenderer(PROJECT_ROOT / "assets" / "pitch_template.png", radar_config(show_tracker_ids=False))

    radar = renderer.render([player(1, 12000, -300, "team_a")])

    assert tuple(radar[0, 799]) == (204, 153, 0)


def test_uncalibrated_radar_adds_red_overlay_and_text():
    from core.radar import RadarRenderer

    renderer = RadarRenderer(PROJECT_ROOT / "assets" / "pitch_template.png", radar_config(show_tracker_ids=False))

    radar = renderer.render([player(1, 5250, 3400, "team_a")], calibrated=False)

    red_pixels = (radar[:, :, 2] > 150) & (radar[:, :, 1] < 80) & (radar[:, :, 0] < 80)
    assert int(red_pixels.sum()) > 1000


def test_radar_draws_tracker_id_text_when_enabled():
    from core.radar import RadarRenderer

    visible_ids = RadarRenderer(
        PROJECT_ROOT / "assets" / "pitch_template.png",
        radar_config(show_tracker_ids=True),
    ).render([player(42, 5250, 3400, "team_a")])
    hidden_ids = RadarRenderer(
        PROJECT_ROOT / "assets" / "pitch_template.png",
        radar_config(show_tracker_ids=False),
    ).render([player(42, 5250, 3400, "team_a")])

    text_region_visible = visible_ids[250:275, 410:450]
    text_region_hidden = hidden_ids[250:275, 410:450]

    assert np.count_nonzero(cv2.absdiff(text_region_visible, text_region_hidden)) > 0


def test_radar_uses_copy_of_pitch_template_and_can_save_requested_smoke_image(tmp_path):
    from core.radar import RadarRenderer

    template = PROJECT_ROOT / "assets" / "pitch_template.png"
    before = cv2.imread(str(template), cv2.IMREAD_COLOR)
    renderer = RadarRenderer(template, radar_config(show_tracker_ids=False))

    radar = renderer.render([player(1, 5250, 3400, "team_a")], ball_pos=(5250, 3400))
    output_path = tmp_path / "test_radar.png"
    assert cv2.imwrite(str(output_path), radar)
    after = cv2.imread(str(template), cv2.IMREAD_COLOR)

    assert output_path.exists()
    assert cv2.imread(str(output_path), cv2.IMREAD_COLOR).shape == (520, 800, 3)
    np.testing.assert_array_equal(before, after)
