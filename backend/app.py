import cv2
import time
import threading
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import pyttsx3

from simple_drowsiness_detector import SimpleDrowsinessDetector

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

# TTS loop state
_tts_thread: Optional[threading.Thread] = None
_tts_stop_event: Optional[threading.Event] = None
_tts_active: bool = False

# Initialize TTS engine
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
        if _running and _capture is not None and _capture.isOpened():
            return True
        _detector = SimpleDrowsinessDetector(
            drowsy_duration_sec=drowsy_duration_sec,
            detection_interval=2,
            min_face_size=80,
        )
        _capture = _open_camera(camera_index, width, height)
        _running = True
        return _capture is not None and _capture.isOpened()


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
    global _tts_thread, _tts_stop_event, _tts_active
    if _tts_engine is None:
        return False
    if not _custom_alert_message:
        return False
    if _tts_active:
        return True
    
    # Stop any existing TTS first
    stop_tts_loop()
    
    _tts_stop_event = threading.Event()
    def _loop():
        while not _tts_stop_event.is_set():
            try:
                _tts_engine.say(_custom_alert_message)
                _tts_engine.runAndWait()
                # Small pause between repeats
                time.sleep(0.5)
            except Exception as e:
                print(f"[TTS] Error: {e}")
                time.sleep(0.5)
    _tts_thread = threading.Thread(target=_loop, daemon=True)
    _tts_thread.start()
    _tts_active = True
    print(f"[TTS] Loop started: '{_custom_alert_message}'")
    return True


def stop_detection():
    global _detector, _capture, _running
    with _lock:
        _running = False
        if _capture is not None:
            try:
                _capture.release()
            except Exception:
                pass
            _capture = None
        if _detector is not None and _detector.audio_enabled:
            try:
                _detector.trigger_alarm(False)
            except Exception:
                pass
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
    global _custom_alert_message
    _custom_alert_message = alert.message.strip() if alert.message.strip() else None
    return {"message": "Alert message updated", "current_message": _custom_alert_message}


@app.get("/api/get_alert_message")
async def get_alert_message():
    return {"message": _custom_alert_message}


def _ensure_capture(width: int = 640, height: int = 480):
    global _capture
    if _capture is None or not _capture.isOpened():
        _capture = _open_camera(0, width, height)


def generate_mjpeg():
    global _detector, _capture, _running, _tts_active
    if _detector is None:
        return
    while _running:
        if _capture is None or not _capture.isOpened():
            _ensure_capture()
            time.sleep(0.05)
            continue
        ok, frame = _capture.read()
        if not ok or frame is None:
            print("[WARN] Failed to read frame, reinitializing camera...")
            try:
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
            
            # Handle alarm activation/deactivation
            if alarm_active:
                # Start alarm if not already active
                if not (_detector.alarm_active or _tts_active):
                    if _custom_alert_message and _tts_engine is not None:
                        start_tts_loop()
                    else:
                        _detector.trigger_alarm(True)
            else:
                # Stop alarm if leaving drowsy state
                if _detector.alarm_active:
                    _detector.trigger_alarm(False)
                if _tts_active:
                    stop_tts_loop()
        else:
            # No face detected -> stop any alarms and reset state
            if _detector.alarm_active:
                _detector.trigger_alarm(False)
            if _tts_active:
                stop_tts_loop()
            # Reset drowsiness state when no face is detected
            _detector._below_start_time = None
        frame = _detector.draw_overlays(frame, faces, eyes, status, confidence, ear, alarm_active)
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


@app.get("/api/stream")
async def stream():
    if not _running:
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
                "alarm": _detector.alarm_active or _tts_active,
            })
    except WebSocketDisconnect:
        pass
