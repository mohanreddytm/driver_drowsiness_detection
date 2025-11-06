"""
utils.py

Utility helpers:
- Image preprocessing (grayscale CLAHE)
- IoU and NMS for bounding boxes
- ViT preprocessing
- Drawing helpers and simple logger
- Continuous alert sound (default beep only) with start/stop
"""

from typing import List, Tuple, Optional
from collections import deque

import cv2
import numpy as np

# Sound imports
import threading
import platform
import time
import os
import tempfile
import subprocess
from playsound import playsound

try:
	import winsound  # type: ignore
except Exception:
	winsound = None  # type: ignore

try:
	import simpleaudio as sa  # type: ignore
except Exception:
	sa = None  # type: ignore


# ---------- Preprocessing ----------

def apply_preprocessing(gray_frame: np.ndarray) -> np.ndarray:
	"""Gaussian blur + CLAHE to stabilize lighting for Haar detection."""
	blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	return clahe.apply(blurred)


def preprocess_for_vit(bgr: np.ndarray) -> np.ndarray:
	"""Prepare BGR image to RGB for ViT crop usage (crop/resize handled elsewhere)."""
	return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------- Geometry / NMS ----------

def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
	interW = max(0, xB - xA)
	interH = max(0, yB - yA)
	inter = interW * interH
	areaA = boxA[2] * boxA[3]
	areaB = boxB[2] * boxB[3]
	union = max(1e-6, float(areaA + areaB - inter))
	return float(inter) / union


def nms(boxes: List[Tuple[int, int, int, int]], scores: Optional[List[float]] = None, iou_thr: float = 0.3) -> List[Tuple[int, int, int, int]]:
	"""Greedy NMS for (x,y,w,h). If scores None, use area as score."""
	if not boxes:
		return []
	scores = scores if scores is not None else [w * h for (_, _, w, h) in boxes]
	idxs = list(range(len(boxes)))
	idxs.sort(key=lambda i: scores[i], reverse=True)
	picked: List[int] = []
	while idxs:
		current = idxs.pop(0)
		picked.append(current)
		idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_thr]
	return [boxes[i] for i in picked]


# ---------- Drawing ----------

def draw_bbox(frame: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], label: Optional[str] = None) -> None:
	x, y, w, h = box
	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	if label:
		cv2.putText(frame, label, (x, max(0, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_status_overlay(frame: np.ndarray, status: str, ear: float, vit_conf: float, closed_t: float) -> None:
	color = (0, 255, 0) if status == "Active" else (0, 0, 255)
	cv2.putText(frame, f"Status: {status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
	cv2.putText(frame, f"EAR: {ear:.3f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, f"ViT: {vit_conf:.2f}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, f"Closed: {closed_t:.1f}s", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)


def draw_drowsy_alert(frame: np.ndarray) -> None:
	h, w = frame.shape[:2]
	overlay = frame.copy()
	cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), thickness=20)
	alpha = 0.25
	cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
	cv2.putText(frame, "DROWSY ALERT", (int(0.18 * w), int(0.15 * h)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)


# ---------- Simple logger ----------

def log(msg: str) -> None:
	print(msg)


# ---------- Continuous Default Alert Sound ----------

_sound_thread: Optional[threading.Thread] = None
_stop_flag: bool = False
_default_beep_wav: Optional[str] = None
_backend: str = ""


def current_sound_backend() -> str:
	return _backend


def _ensure_default_beep_wav() -> Optional[str]:
	"""Create a short sine beep wav in temp dir and cache path (non-Windows fallback)."""
	global _default_beep_wav
	if _default_beep_wav and os.path.exists(_default_beep_wav):
		return _default_beep_wav
	try:
		import wave, struct, math
		tmpdir = tempfile.gettempdir()
		path = os.path.join(tmpdir, "drowsy_beep.wav")
		sample_rate = 22050
		duration = 0.35
		freq = 1000.0
		n_samples = int(sample_rate * duration)
		with wave.open(path, 'w') as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)
			wf.setframerate(sample_rate)
			for i in range(n_samples):
				val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
				wf.writeframesraw(struct.pack('<h', val))
			wf.writeframes(b'')
		_default_beep_wav = path
		return _default_beep_wav
	except Exception:
		return None


def _run_cmd(cmd: List[str]) -> bool:
	try:
		proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		return proc.returncode == 0
	except Exception:
		return False


def audio_self_test() -> None:
	"""Emit a brief test tone using the active backend selection logic."""
	sysname = platform.system()
	if sysname == "Windows" and winsound is not None:
		try:
			winsound.Beep(1500, 300)
			return
		except Exception:
			pass
	wav = _ensure_default_beep_wav()
	if wav and sa is not None:
		try:
			wo = sa.WaveObject.from_wave_file(wav)
			wo.play().wait_done()
			return
		except Exception:
			pass
	if wav:
		try:
			playsound(wav, block=True)
			return
		except Exception:
			pass
	# OS fallbacks
	sysname = platform.system()
	if sysname == "Darwin":
		_run_cmd(["afplay", "/System/Library/Sounds/Glass.aiff"])  # best-effort
	elif sysname == "Linux":
		_run_cmd(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"]) or \
		_run_cmd(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"]) or \
		_run_cmd(["beep", "-f", "1500", "-l", "300"]) 


def play_alert_sound() -> None:
	"""
	Start a continuous, non-blocking default alert sound.
	- Windows: Beep loop in a background thread.
	- macOS/Linux: loop generated WAV with simpleaudio if available; else playsound; else OS-level tools.
	"""
	global _sound_thread, _stop_flag, _backend

	if _sound_thread is not None and _sound_thread.is_alive():
		return

	_stop_flag = False

	def loop_sound() -> None:
		global _stop_flag, _backend
		sysname_local = platform.system()
		while not _stop_flag:
			try:
				if sysname_local == "Windows" and winsound is not None:
					_backend = "winsound.Beep(loop)"
					winsound.Beep(2000, 700)
				elif sysname_local == "Darwin":
					wav = _ensure_default_beep_wav()
					if wav and sa is not None:
						_backend = "simpleaudio(macOS)"
						wo = sa.WaveObject.from_wave_file(wav)
						wo.play().wait_done()
					elif _run_cmd(["afplay", "/System/Library/Sounds/Glass.aiff"]):
						_backend = "afplay"
					else:
						_backend = "playsound(macOS)"
						playsound(wav, block=True) if wav else time.sleep(0.7)
				elif sysname_local == "Linux":
					wav = _ensure_default_beep_wav()
					if wav and sa is not None:
						_backend = "simpleaudio(linux)"
						wo = sa.WaveObject.from_wave_file(wav)
						wo.play().wait_done()
					elif wav:
						_backend = "playsound(linux)"
						playsound(wav, block=True)
					elif _run_cmd(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"]) or \
						_run_cmd(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"]) or \
						_run_cmd(["beep", "-f", "2000", "-l", "700"]):
						_backend = "os-cmd(linux)"
					else:
						_backend = "sleep-fallback"
						time.sleep(0.7)
				time.sleep(0.3)
			except Exception as e:
				print("Sound error:", e)
				_backend = "error"
				time.sleep(1.0)

	_sound_thread = threading.Thread(target=loop_sound, daemon=True)
	_sound_thread.start()
	print("[sound] backend:", _backend or "initializing")


def stop_alert_sound() -> None:
	"""Stop any currently playing alert sound."""
	global _stop_flag, _backend
	_stop_flag = True
	_backend = ""
