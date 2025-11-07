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


def is_alert_sound_playing() -> bool:
	"""Check if alert sound is currently playing."""
	global _sound_thread
	return _sound_thread is not None and _sound_thread.is_alive()


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
	- Windows: Continuous alarm.wav loop in a background thread (NO GAPS!).
	- macOS/Linux: loop alarm.wav with playsound; else OS-level tools.
	"""
	global _sound_thread, _stop_flag, _backend

	# Always restart if thread is dead or not running
	if _sound_thread is not None:
		if _sound_thread.is_alive():
			# Thread is alive, don't restart
			print("[ALARM] Sound thread already running, not restarting")
			return
		else:
			# Thread is dead, clean up and restart
			print("[ALARM] Sound thread is dead, cleaning up and restarting")
			_sound_thread = None

	# Reset stop flag before starting
	_stop_flag = False
	print("[ALARM] Stop flag reset, starting new sound thread")

	def loop_sound() -> None:
		"""Continuous sound loop - plays sound continuously with NO GAPS!"""
		global _stop_flag, _backend
		sysname_local = platform.system()
		print(f"[ALARM] Starting continuous alarm sound loop (platform: {sysname_local})")
		
		# Try to use alarm/alert.wav file first, fallback to beep
		alarm_wav = None
		alarm_paths = ["alarm/alert.wav", "alert.wav", os.path.join(os.path.dirname(__file__), "alarm", "alert.wav")]
		for path in alarm_paths:
			if os.path.exists(path):
				alarm_wav = path
				print(f"[ALARM] Using alarm file: {alarm_wav}")
				break
		
		while not _stop_flag:
			try:
				# Try to use alarm.wav file if available (better sound!)
				if alarm_wav and os.path.exists(alarm_wav):
					_backend = f"playsound({alarm_wav})"
					try:
						# Play the WAV file - this will block until it finishes
						playsound(alarm_wav, block=True)
						# After playing, check if we should continue or stop
						if _stop_flag:
							print("[ALARM] Stop flag set, exiting loop")
							break
						# Play again immediately for continuous sound (NO GAPS!)
						# Don't sleep - just continue the loop
						continue
					except Exception as e:
						print(f"[ALARM] Error playing alarm.wav: {e}, falling back to beep")
						# Fallback to beep - don't set alarm_wav to None yet, try once more
						# If it fails again, we'll fall through to beep
						pass
				
				# Fallback to beep if alarm.wav not available or failed
				if sysname_local == "Windows" and winsound is not None:
					_backend = "winsound.Beep(continuous)"
					# Play beep continuously - NO GAPS! 
					# Use longer beeps with minimal gap for truly continuous sound
					winsound.Beep(2000, 800)  # 800ms beep (longer = less gaps)
					# Almost no sleep - just enough to allow next beep to start
					if not _stop_flag:
						time.sleep(0.01)  # TINY gap (10ms) - almost continuous!
					# Continue loop immediately for next beep
					continue
				elif sysname_local == "Darwin":
					wav = _ensure_default_beep_wav()
					_backend = "afplay|playsound(macOS)"
					if wav:
						# Try afplay first (non-blocking)
						if not _run_cmd(["afplay", wav]):
							# Fallback to playsound (blocking)
							playsound(wav, block=True)
					else:
						time.sleep(0.5)
				elif sysname_local == "Linux":
					wav = _ensure_default_beep_wav()
					_backend = "playsound|os-cmd(linux)"
					if wav:
						try:
							playsound(wav, block=True)
						except Exception:
							# Try OS-level commands
							if not (_run_cmd(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"]) or 
							        _run_cmd(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"]) or 
							        _run_cmd(["beep", "-f", "2000", "-l", "500"])):
								time.sleep(0.5)
					else:
						time.sleep(0.5)
				else:
					# Unknown platform - use WAV file
					wav = _ensure_default_beep_wav()
					if wav:
						try:
							playsound(wav, block=True)
						except Exception:
							time.sleep(0.5)
					else:
						time.sleep(0.5)
				
				# Only sleep if we're not on Windows (Windows handles it differently)
				if sysname_local != "Windows":
					if not _stop_flag:
						time.sleep(0.1)  # Small gap between sound plays
					else:
						break  # Stop flag set, exit
				else:
					# Windows - already handled with continue above
					pass
			except Exception as e:
				print(f"[ALARM] Sound error in loop: {e}")
				import traceback
				traceback.print_exc()
				_backend = "error"
				if _stop_flag:
					print("[ALARM] Stop flag set during error, exiting")
					break
				time.sleep(0.5)  # Wait before retrying
		
		print("[ALARM] Alarm sound loop stopped (stop_flag was set)")

	_sound_thread = threading.Thread(target=loop_sound, daemon=True)
	_sound_thread.start()
	print(f"[ALARM] Alarm sound thread started (thread ID: {_sound_thread.ident})")


def stop_alert_sound() -> None:
	"""Stop any currently playing alert sound and clear thread ref for future cycles."""
	global _stop_flag, _sound_thread, _backend
	_stop_flag = True
	try:
		if _sound_thread is not None:
			_sound_thread.join(timeout=1.0)
	except Exception:
		pass
	_sound_thread = None
	_backend = ""
