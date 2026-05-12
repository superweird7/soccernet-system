# SoccerNet System

Football Vision Pro desktop analysis system for broadcast football footage. It combines YOLO detection, ByteTrack tracking, PRTReID team classification, TVCalib/manual calibration, frame annotation, and a 2D radar view in a PyQt6 application.

## What is in this repo

- `app/` - PyQt6 desktop UI panels and main window
- `core/` - detector, tracker, team classifier, homography, radar, annotator, and pipeline
- `benchmark/` - SoccerNet benchmark runner
- `tests/` - regression tests for the pipeline and UI
- `assets/` - pitch template used by the radar
- `calibration/saved/` - small manual calibration JSON files

Large local artifacts are intentionally excluded from git:

- `datasets/soccernet/`
- `videos/`
- `models/`
- `.venv/`
- `downloads/`
- `outputs/`

Those files exceed normal GitHub limits and should be restored locally.

## Setup

Use Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

TVCalib is expected as a local source checkout when automatic calibration is needed:

```powershell
git clone https://github.com/mm4spa/tvcalib.git tvcalib
pip install -e .\tvcalib
```

Place local assets at these paths:

```text
datasets/soccernet/
videos/Iraq vs. Bolivia 2026.mp4
models/yolo11x.pt
models/prtreid/
models/tvcalib/train_59.pt
```

## Run

```powershell
python main.py
```

## Test

```powershell
python -m pytest tests\ -q
```

Current local verification before publishing: `68 passed`.
