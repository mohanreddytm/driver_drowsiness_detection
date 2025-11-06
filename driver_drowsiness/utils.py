"""
Utility functions for the Driver Drowsiness Detection system.

This module provides helper functions for image processing, visualization,
and alert management.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import threading
from playsound import playsound
import os

# Constants for image processing
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

def resize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize frame to target dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    scaling_factor = TARGET_WIDTH / width
    dim = (TARGET_WIDTH, int(height * scaling_factor))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply preprocessing steps to the input frame.
    
    Args:
        frame: Input BGR frame
        
    Returns:
        Tuple of (preprocessed_color, preprocessed_gray)
    """
    # Resize frame
    frame = resize_frame(frame)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for lighting normalization
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )
    gray = clahe.apply(gray)
    
    # Apply slight Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return frame, gray

def smooth_bbox(
    current_box: Tuple[int, int, int, int],
    box_history: deque
) -> Tuple[int, int, int, int]:
    """
    Apply temporal smoothing to bounding box coordinates.
    
    Args:
        current_box: Current detection box (x, y, w, h)
        box_history: Deque of previous boxes
        
    Returns:
        Smoothed bounding box
    """
    box_history.append(current_box)
    if len(box_history) < 2:
        return current_box
        
    # Calculate mean coordinates
    x = int(np.mean([box[0] for box in box_history]))
    y = int(np.mean([box[1] for box in box_history]))
    w = int(np.mean([box[2] for box in box_history]))
    h = int(np.mean([box[3] for box in box_history]))
    
    return (x, y, w, h)

class AlertManager:
    def __init__(self, sound_path: str = "alert.wav"):
        """
        Initialize alert manager.
        
        Args:
            sound_path: Path to alert sound file
        """
        self.sound_path = sound_path
        self.alert_thread = None
        self.last_alert_time = 0
        self.cooldown = 3.0  # seconds
        
    def play_alert(self):
        """Play alert sound in a separate thread."""
        if self.alert_thread and self.alert_thread.is_alive():
            return
            
        self.alert_thread = threading.Thread(
            target=playsound,
            args=(self.sound_path,),
            daemon=True
        )
        self.alert_thread.start()

def draw_debug_info(
    frame: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]],
    eye_boxes: List[Tuple[int, int, int, int]],
    ear_value: float,
    is_drowsy: bool,
    vit_conf: float
) -> np.ndarray:
    """
    Draw detection visualization and debug information.
    
    Args:
        frame: Input frame
        face_box: Face bounding box
        eye_boxes: List of eye bounding boxes
        ear_value: Current EAR value
        is_drowsy: Current drowsiness state
        vit_conf: Vision Transformer confidence score
        
    Returns:
        Annotated frame
    """
    # Draw face box
    if face_box:
        x, y, w, h = face_box
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw eye boxes
        for ex, ey, ew, eh in eye_boxes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 30),
                font, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"ViT: {vit_conf:.2f}", (10, 60),
                font, 0.7, (0, 255, 255), 2)
    
    if is_drowsy:
        cv2.putText(frame, "DROWSY!", (frame.shape[1]//2 - 50, 30),
                    font, 0.8, (0, 0, 255), 2)
                    
    return frame