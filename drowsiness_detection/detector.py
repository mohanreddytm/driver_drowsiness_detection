"""
Face and eye detection module using Haar cascade classifiers.

This module handles the initial detection of faces and eyes using OpenCV's
Haar cascade classifiers with optimized parameters for reliable detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from . import config

class HaarDetector:
    def __init__(self):
        """Initialize Haar cascade detectors for face and eyes."""
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(config.HAAR_EYE_CASCADE_PATH)
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise ValueError("Error loading Haar cascade classifiers")

    def detect_face(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the frame using Haar cascade.
        
        Args:
            frame: Input frame
            roi: Optional region of interest (x, y, w, h)
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI if provided
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_DETECTION_SCALE,
            minNeighbors=config.FACE_DETECTION_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE
        )
        
        # Adjust coordinates if ROI was used
        if roi is not None:
            faces = [(x + fx, y + fy, fw, fh) for fx, fy, fw, fh in faces]
            
        return faces

    def detect_eyes(
        self,
        frame: np.ndarray,
        face_rect: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect eyes within a face region using Haar cascade.
        
        Args:
            frame: Input frame
            face_rect: Face bounding box (x, y, w, h)
            
        Returns:
            List of eye bounding boxes (x, y, w, h)
        """
        x, y, w, h = face_rect
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=config.EYE_DETECTION_SCALE,
            minNeighbors=config.EYE_DETECTION_NEIGHBORS
        )
        
        # Adjust coordinates relative to original frame
        eyes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
        
        # Filter and sort eyes (left to right)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            
        return eyes