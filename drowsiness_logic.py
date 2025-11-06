"""
drowsiness_logic.py

Computes Eye Aspect Ratio (EAR) and manages drowsiness state with a 7-second
continuous-closure trigger. EAR is approximated using Haar eye bounding boxes
as height/width, then smoothed using a moving average window.
"""

import time
from collections import deque
from typing import Deque, Iterable, Optional, Tuple


class DrowsinessState:
	"""Tracks EAR over time and triggers drowsiness after a threshold of seconds."""

	def __init__(self, ear_threshold: float = 0.25, seconds_threshold: float = 7.0, smoothing_window: int = 10) -> None:
		self.ear_threshold = float(ear_threshold)
		self.seconds_threshold = float(seconds_threshold)
		self.ear_values: Deque[float] = deque(maxlen=int(smoothing_window))
		self.closed_start_time: Optional[float] = None

	def update(self, ear: float, now: Optional[float] = None) -> Tuple[str, float]:
		"""Update with new EAR value.

		Returns (status, closed_duration_seconds).
		"""
		now = time.time() if now is None else now
		self.ear_values.append(float(ear))
		smoothed_ear = sum(self.ear_values) / len(self.ear_values)

		if smoothed_ear < self.ear_threshold:
			if self.closed_start_time is None:
				self.closed_start_time = now
			closed_duration = now - self.closed_start_time
			status = "Drowsy" if closed_duration >= self.seconds_threshold else "Active"
			return status, closed_duration
		else:
			self.closed_start_time = None
			return "Active", 0.0


def compute_ear_from_eye_boxes(eyes: Iterable[Tuple[int, int, int, int]]) -> float:
	"""Compute EAR from eye bounding boxes as average(height / width).

	If no eyes are provided, returns 0.0.
	"""
	ears = []
	for (_, _, w, h) in eyes:
		if w > 0:
			ears.append(max(0.0, min(1.0, float(h) / float(w))))
	return float(sum(ears) / len(ears)) if ears else 0.0
