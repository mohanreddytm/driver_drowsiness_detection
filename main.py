"""
main.py

Orchestrates capture, detection, tracking, EAR, and alerts.
- Haar + NMS + ViT validation (every Nth frame)
- Single-face selection with head-region constraint
- CSRT/KCF tracking with health checks and smoothed drawing
- Eyes via Haar with FaceMesh fallback, EAR smoothing, 7-second trigger
- debug_mode overlays
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from detector import run_haar_faces, run_nms, select_primary_face, validate_with_vit, get_eye_cascade
from tracker_wrapper import TrackerWrapper
from eyes import detect_eyes_haar, ear_from_facemesh
from drowsiness import EarSmoother, DrowsinessTimer
from utils import apply_preprocessing, draw_bbox, draw_drowsy_alert, draw_status_overlay, audio_self_test, current_sound_backend


DEBUG_MODE = False
VIT_EVERY_N_FRAMES = 5


def main() -> None:
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	tracker = TrackerWrapper()
	eye_cascade = get_eye_cascade()
	ear_smoother = EarSmoother(window=7)
	drowsy = DrowsinessTimer(threshold_ear=0.25, trigger_seconds=7.0)

	vit_conf = 0.0
	tracked_box: Optional[Tuple[int, int, int, int]] = None
	frame_idx = 0

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		frame_idx += 1
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_pre = apply_preprocessing(gray)

		new_haar_box = None
		if tracked_box is None or frame_idx % VIT_EVERY_N_FRAMES == 0:
			# Run Haar + NMS periodically or when not tracking
			candidates = run_haar_faces(gray_pre)
			cand_nms = run_nms(candidates, iou_thr=0.3)
			primary = select_primary_face(cand_nms, frame.shape)
			if primary is not None:
				new_haar_box = primary
				conf = validate_with_vit(frame, primary)
				vit_conf = conf
				if conf >= 0.85:
					if tracked_box is None:
						tracker.init(frame, primary)
					tracked_box = primary
				else:
					# keep tracking but mark for revalidation
					pass

		# Update tracker if we have one
		if tracked_box is not None:
			ok, box = tracker.update(frame)
			if ok and box is not None:
				tracked_box = box
			else:
				tracked_box = None

		# Decide re-detection if tracker unhealthy
		if tracker.needs_redetect(new_haar_box):
			tracked_box = None

		# Draw and compute EAR
		cur_ear = 0.0
		if tracked_box is not None:
			draw_bbox(frame, tracked_box, (0, 255, 0), "Face")
			# Try FaceMesh first for precise EAR
			face_ear = ear_from_facemesh(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), tracked_box)
			if face_ear is None:
				# fallback to Haar eye boxes (top 40% of ROI)
				eyes = detect_eyes_haar(gray_pre, tracked_box, eye_cascade)
				if eyes:
					# approximate EAR from eye boxes by height/width median
					ears = []
					for ex, ey, ew, eh in eyes:
						if ew > 0:
							ears.append(float(eh) / float(ew))
					if ears:
						cur_ear = float(sum(ears) / len(ears))
			else:
				cur_ear = face_ear

		smoothed = ear_smoother.add(cur_ear)
		status, closed_t, alert_started = drowsy.update(smoothed, time.time())
		draw_status_overlay(frame, status, smoothed, vit_conf, closed_t)
		if status == "Drowsy":
			draw_drowsy_alert(frame)
			backend = current_sound_backend()
			if backend:
				cv2.putText(frame, f"Audio: {backend}", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

		# Debug overlays
		if DEBUG_MODE:
			if new_haar_box is not None:
				draw_bbox(frame, new_haar_box, (0, 200, 255), "Haar")

		cv2.imshow("Driver Drowsiness Detection", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('a'):
			# quick audio test
			audio_self_test()

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
