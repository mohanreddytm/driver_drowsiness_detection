"""
eyes.py

Eye detection and EAR calculation.
Primary: Haar eye cascade within top 40% of face ROI.
Fallback: MediaPipe FaceMesh landmarks for precise EAR.
If both fail, try contour-based approximation within boxes.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
	import mediapipe as mp
	has_mp = True
except Exception:
	has_mp = False


# MediaPipe eye landmark indices (approximate 6-point set per eye)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def _ear_from_landmarks(pts: np.ndarray) -> float:
	# pts expected shape (6,2): [p1,p2,p3,p4,p5,p6]
	p1, p2, p3, p4, p5, p6 = pts
	# Standard EAR formula using six points
	ver = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
	hor = np.linalg.norm(p1 - p4) + 1e-6
	return float(ver / (2.0 * hor))


def _approx_ear_from_box(gray: np.ndarray, box: Tuple[int, int, int, int]) -> float:
	x, y, w, h = box
	roi = gray[y : y + h, x : x + w]
	if roi.size == 0 or w <= 0:
		return 0.0
	roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
	edges = cv2.Canny(roi_blur, 30, 100)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return max(0.0, min(1.0, float(h) / float(max(1, w))))
	c = max(contours, key=cv2.contourArea)
	xc, yc, ww, hh = cv2.boundingRect(c)
	return max(0.0, min(1.0, float(hh) / float(max(1, ww))))


def detect_eyes_haar(gray_pre: np.ndarray, face_box: Tuple[int, int, int, int], eye_cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
	x, y, w, h = face_box
	roi = gray_pre[y : y + h, x : x + w]
	upper = roi[0 : max(1, int(0.4 * h)), :]
	eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
	abs_boxes: List[Tuple[int, int, int, int]] = []
	for (ex, ey, ew, eh) in eyes:
		abs_boxes.append((x + ex, y + ey, ew, eh))
	abs_boxes.sort(key=lambda b: b[2], reverse=True)
	return abs_boxes[:2]


def ear_from_facemesh(rgb: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[float]:
	if not has_mp:
		return None
	x, y, w, h = face_box
	roi = rgb[y : y + h, x : x + w]
	if roi.size == 0:
		return None
	mp_face_mesh = mp.solutions.face_mesh
	with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
		res = fm.process(roi)
		if not res.multi_face_landmarks:
			return None
		landmarks = res.multi_face_landmarks[0]
		pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark], dtype=np.float32)
		left = pts[LEFT_EYE]
		right = pts[RIGHT_EYE]
		return float((_ear_from_landmarks(left) + _ear_from_landmarks(right)) / 2.0)
