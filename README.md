# Driver Drowsiness Detection (Stable, Production-Ready)

Hybrid pipeline with Haar + ViT validation, CSRT/KCF tracking, EAR via landmarks fallback, and a robust 7-second drowsiness trigger.

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

- Press `q` to quit.
- Set `DEBUG_MODE = True` in `main.py` to visualize Haar candidates, ViT conf, and smoothed boxes.

## Requirements
- Python 3.10+
- Webcam

## Architecture
- `main.py`: Orchestrates capture, periodic Haar+ViT, tracker lifecycle, EAR smoothing, and alerts.
- `detector.py`: `run_haar_faces`, `run_nms`, `select_primary_face`, `validate_with_vit`, and eye cascade loader.
- `tracker_wrapper.py`: CSRT/KCF tracker with health checks and smoothed box output.
- `eyes.py`: Haar eye detection (top 40% ROI). Fallback to MediaPipe FaceMesh for precise EAR.
- `drowsiness.py`: EAR smoothing buffer and 7-second consecutive-close timer with alert gating.
- `utils.py`: CLAHE preprocessing, IoU, NMS, drawing helpers.

## Key Parameters
- Frame: 640x480
- Haar: scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
- NMS IoU: 0.3
- ViT confidence threshold: 0.85 (validated every 5 frames)
- Tracker: CSRT (fallback KCF)
- Box smoothing: 7 frames
- EAR smoothing: 7 frames
- EAR threshold: 0.25
- Drowsy trigger: 7.0 seconds
- Eye ROI limit: top 40% of face ROI

## Notes
- ViT runs every N frames to reduce load, tracker handles intermediate frames.
- Only the primary face is tracked; all other boxes are suppressed via NMS + selection.
- FaceMesh is used for precise EAR when available; otherwise, Haar eye boxes are used with an approximation.

## Testing
- `test_detection.py` prints face detection + ViT confidence and EAR on sample images placed in `samples/`.

## Cleanup
The repo has been refactored to just the core modules above. Remove any legacy folders left from prior versions if still present.
