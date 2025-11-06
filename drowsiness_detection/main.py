"""
Main module for drowsiness detection system.

This module orchestrates the complete drowsiness detection pipeline by combining
face detection, validation, tracking, and EAR calculation components.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional

from . import config
from .detector import HaarDetector
from .vit_validator import VisionTransformerValidator
from .tracker import SmoothTracker
from .ear_calculator import EARCalculator
from . import utils

class DrowsinessDetector:
    def __init__(self):
        """Initialize the drowsiness detection system."""
        self.detector = HaarDetector()
        self.validator = VisionTransformerValidator()
        self.tracker = SmoothTracker()
        self.ear_calculator = EARCalculator()
        
        self.last_alert_time = 0
        
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Process a single frame for drowsiness detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, is_drowsy, ear_value)
        """
        # Resize frame for processing
        frame = utils.resize_frame(frame)
        
        # Initial face detection
        face_rects = self.detector.detect_face(frame)
        
        # Variables for tracking
        validated_face = None
        validated_eyes = []
        ear_value = 0.0
        is_drowsy = False
        
        if face_rects:
            # Get the largest face
            face_rect = max(face_rects, key=lambda r: r[2] * r[3])
            
            # Validate face using ViT
            is_valid, confidence = self.validator.validate_face(frame, face_rect)
            
            if is_valid and confidence >= config.ALERT_MIN_CONFIDENCE:
                validated_face = face_rect
                
                # Detect eyes in validated face
                eye_rects = self.detector.detect_eyes(frame, face_rect)
                
                if len(eye_rects) == 2:
                    validated_eyes = eye_rects
                    
        # Update tracking
        tracked_face, tracked_eyes = self.tracker.update(validated_face, validated_eyes)
        
        if tracked_face and tracked_eyes:
            # Calculate EAR and drowsiness
            # Note: In a real implementation, we would extract actual landmarks here
            # For now, we'll use placeholder landmarks based on eye rectangles
            placeholder_landmarks = [
                np.array([[ex, ey], [ex+ew//2, ey], [ex+ew, ey],
                         [ex+ew, ey+eh], [ex+ew//2, ey+eh], [ex, ey+eh]])
                for ex, ey, ew, eh in tracked_eyes
            ]
            
            ear_value, is_drowsy = self.ear_calculator.update(
                tracked_eyes, placeholder_landmarks)
            
            # Handle drowsiness alert
            if is_drowsy:
                current_time = time.time()
                if current_time - self.last_alert_time >= config.ALERT_COOLDOWN:
                    self.last_alert_time = current_time
                    # TODO: Implement alert system
                    
            # Draw visualization
            frame = utils.draw_detection_visualization(
                frame, tracked_face, tracked_eyes, ear_value, is_drowsy)
            
        return frame, is_drowsy, ear_value
        
    def release(self):
        """Release any resources held by the detector."""
        cv2.destroyAllWindows()