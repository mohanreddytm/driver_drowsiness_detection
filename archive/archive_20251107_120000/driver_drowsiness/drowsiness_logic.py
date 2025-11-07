"""
Drowsiness detection logic using Eye Aspect Ratio (EAR) calculation.

This module handles eye detection, EAR calculation, and drowsiness state
management with temporal smoothing.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import time
from . import utils

class DrowsinessDetector:
    def __init__(
        self,
        eye_cascade_path: str,
        ear_threshold: float = 0.25,
        ear_frames_threshold: int = 20,
        ear_smoothing_window: int = 5
    ):
        """
        Initialize drowsiness detection components.
        
        Args:
            eye_cascade_path: Path to eye Haar cascade XML
            ear_threshold: EAR threshold for drowsiness
            ear_frames_threshold: Number of consecutive frames below threshold
            ear_smoothing_window: Window size for EAR smoothing
        """
        # Load eye cascade
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if self.eye_cascade.empty():
            raise ValueError("Error loading eye cascade classifier")
            
        # EAR parameters
        self.ear_threshold = ear_threshold
        self.ear_frames_threshold = ear_frames_threshold
        
        # State management
        self.ear_history = deque(maxlen=ear_smoothing_window)
        self.below_threshold_frames = 0
        self.last_ear = 0.0
        self.is_drowsy = False
        
        # Eye detection parameters
        self.min_eye_size = (20, 20)
        self.scale_factor = 1.1
        self.min_neighbors = 3
        
    def detect_eyes(
        self,
        frame: np.ndarray,
        face_box: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes within face region.
        
        Args:
            frame: Input frame
            face_box: Face bounding box
            
        Returns:
            List of eye bounding boxes
        """
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_eye_size
        )
        
        # Convert to absolute coordinates
        eyes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
        
        # Keep only the two largest detections
        if len(eyes) > 2:
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            # Sort left to right
            eyes.sort(key=lambda e: e[0])
            
        return eyes
        
    def calculate_ear(
        self,
        frame: np.ndarray,
        eye_boxes: List[Tuple[int, int, int, int]]
    ) -> float:
        """
        Calculate Eye Aspect Ratio from detected eyes.
        
        Args:
            frame: Input frame
            eye_boxes: List of eye bounding boxes
            
        Returns:
            Smoothed EAR value
        """
        if len(eye_boxes) != 2:
            return self.last_ear
            
        # Calculate EAR for each eye
        ears = []
        for ex, ey, ew, eh in eye_boxes:
            eye_roi = frame[ey:ey+eh, ex:ex+ew]
            ear = self._calculate_single_ear(eye_roi)
            if ear > 0:
                ears.append(ear)
                
        if not ears:
            return self.last_ear
            
        # Average EAR between eyes
        current_ear = np.mean(ears)
        
        # Apply temporal smoothing
        self.ear_history.append(current_ear)
        smoothed_ear = np.mean(self.ear_history)
        
        self.last_ear = smoothed_ear
        return smoothed_ear
        
    def _calculate_single_ear(self, eye_roi: np.ndarray) -> float:
        """
        Calculate EAR for a single eye region.
        
        Args:
            eye_roi: Eye region image
            
        Returns:
            EAR value
        """
        try:
            # Convert to grayscale if needed
            if len(eye_roi.shape) == 3:
                eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                
            # Get eye height and width
            height, width = eye_roi.shape
            
            # Simple geometric EAR approximation
            # Assuming the eye is properly cropped
            vertical_sum = np.sum(eye_roi[height//4:3*height//4, width//4:3*width//4])
            horizontal_sum = np.sum(eye_roi[height//3:2*height//3, :])
            
            if horizontal_sum == 0:
                return 0
                
            return (vertical_sum / (width * height/2)) / (horizontal_sum / width)
            
        except Exception as e:
            print(f"Error calculating EAR: {str(e)}")
            return 0
            
    def update_drowsiness_state(self, ear_value: float) -> bool:
        """
        Update drowsiness state based on current EAR.
        
        Args:
            ear_value: Current EAR value
            
        Returns:
            Current drowsiness state
        """
        if ear_value < self.ear_threshold:
            self.below_threshold_frames += 1
        else:
            self.below_threshold_frames = 0
            
        # Update drowsiness state
        self.is_drowsy = self.below_threshold_frames >= self.ear_frames_threshold
        return self.is_drowsy