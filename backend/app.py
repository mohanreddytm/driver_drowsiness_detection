import cv2
import time
import threading
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import pyttsx3
import sys
import os

# Ensure project root is on sys.path to import shared modules like utils.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from simple_drowsiness_detector import SimpleDrowsinessDetector
from utils import play_alert_sound, stop_alert_sound  # default alert sound

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: Optional[SimpleDrowsinessDetector] = None
_capture: Optional[cv2.VideoCapture] = None
_running = False
_lock = threading.Lock()
_custom_alert_message: Optional[str] = None
_tts_engine: Optional[pyttsx3.Engine] = None

# Audio/TTS state
_tts_thread: Optional[threading.Thread] = None
_tts_stop_event: Optional[threading.Event] = None
_tts_active: bool = False
_audio_active: bool = False  # default beep from utils

# Initialize TTS engine (kept but not used when default alert is active)
try:
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 150)
    _tts_engine.setProperty('volume', 0.9)
    print("[INFO] TTS engine initialized successfully")
except Exception as e:
    print(f"[WARN] TTS engine initialization failed: {e}")
    _tts_engine = None


class AlertMessage(BaseModel):
    message: str


def _open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"[INFO] Camera opened with backend {backend}")
            return cap
        if cap is not None:
            cap.release()
    print("[ERROR] Failed to open camera with all backends")
    return None


def start_detection(camera_index: int = 0, width: int = 640, height: int = 480, drowsy_duration_sec: float = 7.0) -> bool:
    global _detector, _capture, _running
    with _lock:
        if _running and _capture is not None and _capture.isOpened() and _detector is not None:
            return True
        # Open camera first; only then create detector and mark running
        capture = _open_camera(camera_index, width, height)
        if capture is None or not capture.isOpened():
            _running = False
            _detector = None
            _capture = None
            return False
        _capture = capture
        _detector = SimpleDrowsinessDetector(
            drowsy_duration_sec=drowsy_duration_sec,
            detection_interval=2,
            min_face_size=80,
        )
        _running = True
        return True


def stop_tts_loop():
    global _tts_thread, _tts_stop_event, _tts_active
    if _tts_thread is not None:
        try:
            if _tts_stop_event is not None:
                _tts_stop_event.set()
            _tts_thread.join(timeout=2.0)
        except Exception:
            pass
    _tts_thread = None
    _tts_stop_event = None
    _tts_active = False
    try:
        if _tts_engine is not None:
            _tts_engine.stop()
    except Exception:
        pass


def start_tts_loop():
    # TTS kept for compatibility but not used when default audio is preferred
    return False


def stop_detection():
    global _detector, _capture, _running, _audio_active
    with _lock:
        _running = False
        if _capture is not None:
            try:
                _capture.release()
            except Exception:
                pass
            _capture = None
        # Stop any active audio
        if _audio_active:
            try:
                stop_alert_sound()
            except Exception:
                pass
            _audio_active = False
        stop_tts_loop()
        _detector = None


@app.post("/api/start")
async def api_start(camera: int = Query(0), width: int = Query(640), height: int = Query(480)):
    ok = start_detection(camera_index=camera, width=width, height=height)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to open camera. Try a different index (e.g., camera=1) and close other apps using the camera.")
    return {"ok": True}


@app.post("/api/stop")
async def api_stop():
    stop_detection()
    return {"ok": True}


@app.get("/api/health")
async def api_health():
    return {"running": _running}


@app.post("/api/set_alert_message")
async def set_alert_message(alert: AlertMessage):
    # Default beep is used; TTS/custom message disabled intentionally
    return {"message": "Default beep mode active", "current_message": None}


@app.get("/api/get_alert_message")
async def get_alert_message():
    return {"message": None}


def _ensure_capture(width: int = 640, height: int = 480):
    global _capture
    if _capture is None or not _capture.isOpened():
        _capture = _open_camera(0, width, height)


def generate_mjpeg():
    global _detector, _capture, _running, _audio_active
    # Ensure detector exists
    if _detector is None:
        if not start_detection():
            return
    while _running:
        if _detector is None:
            time.sleep(0.05)
            if not start_detection():
                continue
        if _capture is None or not _capture.isOpened():
            _ensure_capture()
            time.sleep(0.05)
            continue
        ok, frame = _capture.read()
        if not ok or frame is None:
            print("[WARN] Failed to read frame, reinitializing camera...")
            try:
                if _capture is not None:
                    _capture.release()
            except Exception:
                pass
            _capture = None
            time.sleep(0.05)
            continue
        faces, eyes = _detector.detect_faces_and_eyes(frame)
        status = "Active"
        confidence = 0.0
        ear = 0.0
        alarm_active = False
        if len(faces) > 0:
            ear = _detector.calculate_eye_aspect_ratio(eyes)
            is_drowsy, confidence, _ = _detector.classify_drowsiness(ear)
            status = "Drowsy" if is_drowsy else "Active"
            alarm_active = _detector.update_drowsiness_state(is_drowsy)
            # Default alert sound control
            if alarm_active and not _audio_active:
                try:
                    play_alert_sound()
                    _audio_active = True
                except Exception as e:
                    print(f"[AUDIO] start error: {e}")
            elif not alarm_active and _audio_active:
                try:
                    stop_alert_sound()
                except Exception as e:
                    print(f"[AUDIO] stop error: {e}")
                _audio_active = False
        else:
            # No face detected -> stop audio and reset state
            if _audio_active:
                try:
                    stop_alert_sound()
                except Exception:
                    pass
                _audio_active = False
            _detector._below_start_time = None
        frame = _detector.draw_overlays(frame, faces, eyes, status, confidence, ear, alarm_active)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


@app.get("/api/stream")
async def stream():
    if not _running or _detector is None:
        ok = start_detection()
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to open camera. Try hitting /api/start?camera=1 or close other apps using the camera.")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(generate_mjpeg(), media_type='multipart/x-mixed-replace; boundary=frame', headers=headers)


@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.receive_text()
            if _detector is None:
                await ws.send_json({"status": "Idle", "alarm": False, "elapsed": 0.0, "target": 0.0})
                continue
            elapsed = 0.0
            if _detector._below_start_time is not None:
                elapsed = max(0.0, time.time() - _detector._below_start_time)
            status = "Drowsy" if (_detector.drowsy_duration_sec > 0 and elapsed >= _detector.drowsy_duration_sec) else ("Warning" if _detector._below_start_time else "Active")
            await ws.send_json({
                "status": status,
                "elapsed": round(elapsed, 2),
                "target": _detector.drowsy_duration_sec,
                "alarm": _audio_active,
            })
    except WebSocketDisconnect:
        pass
