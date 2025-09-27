Driver Drowsiness Detection (Python + OpenCV + dlib)

A modular, real-time driver drowsiness detection system using a webcam, OpenCV, dlib facial landmarks, and a loud alarm.

Features
- Face and eye detection via dlib 68-point facial landmarks
- Eye Aspect Ratio (EAR) to detect closed eyes across consecutive frames
- Optional yawning detection using simple Mouth Aspect Ratio (MAR)
- Visual overlays: face box, eye contours, EAR/MAR values, status text
- Loud, looping alarm when drowsiness is detected
- Clean quit with `q`

Project Structure
```
main.py          # main application
utils.py         # helper functions (EAR/MAR, model + alarm setup)
requirements.txt # dependencies
README.md        # this file
alarm/alert.wav  # alarm sound (auto-generated if missing)
models/          # shape_predictor_68_face_landmarks.dat (auto-downloaded if possible)
```

Setup (Windows / PowerShell)
1. Create and activate a virtual environment (recommended):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. First run will try to download the dlib shape predictor (~100MB). If download fails, manually download and place the file:
- File: `shape_predictor_68_face_landmarks.dat`
- Place into folder: `models/`
- Official URL: `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` (extract the `.bz2`)

Run
```powershell
python main.py
```
- Press `q` to quit cleanly.

Advanced usage
```powershell
# Common options
python main.py \
  --camera 0 \
  --width 960 \
  --ear-threshold 0.25 \
  --ear-consec-frames 20 \
  --mar-threshold 0.60 \
  --yawn-consec-frames 15 \
  --off-frames 10 \
  --show-fps

# Disable audio or overlays if desired
python main.py --no-audio --no-overlay

# Select inner- or outer-mouth landmarks for MAR
python main.py --use-inner-mouth
python main.py --use-outer-mouth

# Use a specific model path
python main.py --predictor models/shape_predictor_68_face_landmarks.dat
```

Notes on performance and accuracy
- Largest-face selection improves robustness when multiple faces are in view.
- FPS overlay (enable with `--show-fps`) helps assess real-time performance.
- Alarm hysteresis (`--off-frames`) prevents chattering when EAR hovers near the threshold.
- Inner-mouth MAR is used when available and is more consistent for yawning.

How it works
- EAR formula: `(||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)`
- If `EAR < 0.25` for `>= 20` consecutive frames → status: Drowsy → alarm plays in a loop.
- Optional MAR checks mouth openness to hint yawning.

Tuning
- Update thresholds in `main.py`:
  - `EAR_THRESHOLD` (typ. 0.20–0.30)
  - `EAR_CONSEC_FRAMES` (typ. 15–30)
  - `MAR_THRESHOLD` (typ. 0.55–0.70)
  - `YAWN_CONSEC_FRAMES` (typ. 10–20)

Notes
- Ensure only one app uses the webcam.
- If `pygame` audio fails, confirm your default output device is available.
- dlib requires C++ build tools for source install; using a prebuilt wheel is recommended.


