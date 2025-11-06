"""
tracker_wrapper.py

Wraps OpenCV trackers (CSRT preferred, KCF fallback) with health checks
and a small smoothing buffer of the last 7 boxes to reduce jitter.
"""

from collections import deque
from typing import Deque, Optional, Tuple

import cv2

from utils import iou


class TrackerWrapper:
	def __init__(self) -> None:
		self.tracker: Optional[cv2.Tracker] = None
		self.fail_streak: int = 0
		self.box_hist: Deque[Tuple[int, int, int, int]] = deque(maxlen=7)

	def _create(self) -> cv2.Tracker:
		if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
			return cv2.legacy.TrackerCSRT_create()
		if hasattr(cv2, "TrackerCSRT_create"):
			return cv2.TrackerCSRT_create()
		if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
			return cv2.legacy.TrackerKCF_create()
		return cv2.TrackerKCF_create()

	def init(self, frame, box: Tuple[int, int, int, int]) -> None:
		self.tracker = self._create()
		self.tracker.init(frame, tuple(map(float, box)))
		self.box_hist.clear()
		self.box_hist.append(box)
		self.fail_streak = 0

	def update(self, frame) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
		if self.tracker is None:
			return False, None
		ok, rect = self.tracker.update(frame)
		if not ok:
			self.fail_streak += 1
			return False, None
		x, y, w, h = rect
		box = (int(x), int(y), int(w), int(h))
		self.box_hist.append(box)
		self.fail_streak = 0
		return True, self.smoothed_box()

	def smoothed_box(self) -> Optional[Tuple[int, int, int, int]]:
		if not self.box_hist:
			return None
		x = int(sum(b[0] for b in self.box_hist) / len(self.box_hist))
		y = int(sum(b[1] for b in self.box_hist) / len(self.box_hist))
		w = int(sum(b[2] for b in self.box_hist) / len(self.box_hist))
		h = int(sum(b[3] for b in self.box_hist) / len(self.box_hist))
		return (x, y, w, h)

	def needs_redetect(self, new_haar_box: Optional[Tuple[int, int, int, int]]) -> bool:
		if self.fail_streak >= 3:
			return True
		if new_haar_box is None:
			return False
		cur = self.smoothed_box()
		if cur is None:
			return True
		return iou(cur, new_haar_box) < 0.3
