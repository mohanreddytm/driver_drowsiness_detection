"""
drowsiness.py

EAR smoothing, consecutive closed-time tracking, and drowsiness alert gating.
Starts and stops continuous default alert sound using utils.play_alert_sound/stop_alert_sound.
"""

import time
from collections import deque
from typing import Deque, Tuple

from utils import play_alert_sound, stop_alert_sound


class EarSmoother:
	def __init__(self, window: int = 7) -> None:
		self.buf: Deque[float] = deque(maxlen=window)

	def add(self, v: float) -> float:
		self.buf.append(float(v))
		return self.value

	@property
	def value(self) -> float:
		if not self.buf:
			return 0.0
		return float(sum(self.buf) / len(self.buf))


class DrowsinessTimer:
	def __init__(self, threshold_ear: float = 0.25, trigger_seconds: float = 7.0) -> None:
		self.threshold_ear = threshold_ear
		self.trigger_seconds = trigger_seconds
		self.closed_start: float = 0.0
		self.closed_time: float = 0.0
		self.alerted: bool = False

	def update(self, smoothed_ear: float, now: float) -> Tuple[str, float, bool]:
		"""Return (status, closed_time, alert_started_this_frame)."""
		alert_started = False
		if smoothed_ear < self.threshold_ear:
			if self.closed_start == 0.0:
				self.closed_start = now
			self.closed_time = now - self.closed_start
			if self.closed_time >= self.trigger_seconds and not self.alerted:
				self.alerted = True
				play_alert_sound()
				alert_started = True
			status = "Drowsy"
		else:
			self.closed_start = 0.0
			self.closed_time = 0.0
			if self.alerted:
				self.alerted = False
				stop_alert_sound()
			status = "Active"
		return status, self.closed_time, alert_started
