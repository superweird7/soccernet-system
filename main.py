from __future__ import annotations

import sys
import os
from pathlib import Path

import yaml
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow


def load_config(path: str | Path = "config.yaml") -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["project_root"] = str(config_path.resolve().parent)
    return config


def create_app(argv: list[str] | None = None) -> tuple[QApplication, MainWindow]:
    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow(load_config())
    return app, window


def main(argv: list[str] | None = None) -> int:
    app, window = create_app(argv if argv is not None else sys.argv)
    window.show()
    autoclose_ms = os.environ.get("FVP_AUTOCLOSE_MS")
    if autoclose_ms:
        QTimer.singleShot(int(autoclose_ms), app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
