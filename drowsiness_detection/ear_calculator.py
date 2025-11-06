"""
Eye Aspect Ratio (EAR) calculation module.

This module handles the calculation and temporal smoothing of eye aspect ratios
for drowsiness detection.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from . import config
from . import utils

class EARCalculator:
    def __init__(self):
        """Initialize the EAR calculator with temporal smoothing."""
        self.ear_history = deque(maxlen=config.EAR_SMOOTHING_WINDOW)
        self.consecutive_frames = 0
        self.last_drowsy = False
        
    def update(
        self,
        eye_rects: List[Tuple[int, int, int, int]],
        landmarks: List[np.ndarray]
    ) -> Tuple[float, bool]:
        """
        Calculate smoothed EAR and determine drowsiness state.
        
        Args:
            eye_rects: List of eye bounding boxes
            landmarks: List of eye landmark coordinates
            
        Returns:
            Tuple of (smoothed_ear, is_drowsy)
        """
        if not eye_rects or len(eye_rects) != 2 or not landmarks:
            return 0.0, False
            
        # Calculate EAR for each eye
        ears = []
        for eye_landmarks in landmarks:
            ear = utils.calculate_eye_aspect_ratio(eye_landmarks)
            ears.append(ear)
            
        # Average EAR between both eyes
        current_ear = np.mean(ears)
        self.ear_history.append(current_ear)
        
        # Apply temporal smoothing
        smoothed_ear = np.mean(self.ear_history)
        
        # Determine drowsiness state
        if smoothed_ear < config.EAR_THRESHOLD:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            
        # Update drowsiness state
        is_drowsy = self.consecutive_frames >= config.EAR_CONSECUTIVE_FRAMES
        self.last_drowsy = is_drowsy
        
        return smoothed_ear, is_drowsy