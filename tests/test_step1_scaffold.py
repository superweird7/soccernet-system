from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_step1_folder_scaffold_exists():
    expected_dirs = [
        "app",
        "core",
        "calibration",
        "calibration/saved",
        "stats",
        "benchmark",
        "assets",
        "models",
        "models/prtreid",
        "models/tvcalib",
        "datasets",
        "datasets/soccernet",
        "videos",
        "tests",
    ]

    missing = [
        path for path in expected_dirs if not (PROJECT_ROOT / path).is_dir()
    ]

    assert missing == []


def test_step1_root_files_exist():
    expected_files = [
        "main.py",
        "config.yaml",
        "requirements.txt",
    ]

    missing = [
        path for path in expected_files if not (PROJECT_ROOT / path).is_file()
    ]

    assert missing == []


def test_config_matches_architecture_step1_defaults():
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["device"] == "cuda"
    assert config["detector_model"] == "yolo11x"
    assert config["detector_model_path"] == "models/yolo11x.pt"
    assert config["detection_confidence_player"] == 0.35
    assert config["detection_confidence_ball"] == 0.20
    assert config["team_classifier_mode"] == "prtreid"
    assert config["prefer_kit_color_classification"] is False
    assert config["calibration_mode"] == "auto"
    assert config["pitch_width_cm"] == 10500
    assert config["pitch_height_cm"] == 6800
    assert config["pitch_template_path"] == "assets/pitch_template.png"
    assert config["radar_width"] == 800
    assert config["radar_height"] == 520
    assert config["show_tracker_ids"] is True
    assert config["team_a_name"] == "Iraq"
    assert config["team_b_name"] == "Bolivia"
    assert config["team_a_color_bgr"] == [121, 103, 46]
    assert config["team_b_color_bgr"] == [238, 236, 214]


def test_requirements_include_architecture_dependencies():
    requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8")

    expected_entries = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "supervision>=0.28.0",
        "prtreid",
        "scikit-learn>=1.3.0",
        "albumentations<2",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyQt6>=6.5.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "streamlit>=1.28.0",
    ]

    for entry in expected_entries:
        assert entry in requirements_text


def test_pitch_template_loads_with_expected_dimensions_and_markings():
    pitch_path = PROJECT_ROOT / "assets" / "pitch_template.png"

    with Image.open(pitch_path) as image:
        assert image.format == "PNG"
        assert image.size == (800, 520)
        rgb = image.convert("RGB")

    def has_white_near(x, y, radius=4):
        for px in range(x - radius, x + radius + 1):
            for py in range(y - radius, y + radius + 1):
                red, green, blue = rgb.getpixel((px, py))
                if red >= 230 and green >= 230 and blue >= 230:
                    return True
        return False

    def is_pitch_green(x, y):
        red, green, blue = rgb.getpixel((x, y))
        return green > red + 45 and green > blue + 45

    assert is_pitch_green(200, 100)
    assert has_white_near(20, 20)  # outer touchline/corner
    assert has_white_near(400, 260)  # center spot and halfway line
    assert has_white_near(460, 260)  # center circle
    assert has_white_near(150, 260)  # left penalty area
    assert has_white_near(650, 260)  # right penalty area


def test_pitch_template_generator_returns_expected_image():
    from tools.generate_pitch_template import create_pitch_template

    image = create_pitch_template()

    assert image.mode == "RGB"
    assert image.size == (800, 520)
