"""
Utility functions for the drowsiness detection system.

This module provides helper functions for image processing, geometric calculations,
visualization, and performance metrics.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """
    Resize a frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        width: Target width
        
    Returns:
        Resized frame
    """
    ratio = width / frame.shape[1]
    dim = (width, int(frame.shape[0] * ratio))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the eye aspect ratio given eye landmarks.
    
    Args:
        eye_landmarks: Array of eye landmark coordinates
        
    Returns:
        Eye aspect ratio value
    """
    # Vertical landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal landmarks
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def smooth_signal(signal: List[float], window_size: int) -> List[float]:
    """
    Apply moving average smoothing to a signal.
    
    Args:
        signal: Input signal values
        window_size: Size of smoothing window
        
    Returns:
        Smoothed signal values
    """
    if window_size <= 1:
        return signal
        
    kernel = np.ones(window_size) / window_size
    return list(np.convolve(signal, kernel, mode='valid'))

def draw_detection_visualization(
    frame: np.ndarray,
    face_rect: Tuple[int, int, int, int],
    eye_rects: List[Tuple[int, int, int, int]],
    ear_value: float,
    is_drowsy: bool
) -> np.ndarray:
    """
    Draw detection visualization on frame.
    
    Args:
        frame: Input frame
        face_rect: Face bounding box (x, y, w, h)
        eye_rects: List of eye bounding boxes
        ear_value: Current EAR value
        is_drowsy: Current drowsiness state
        
    Returns:
        Frame with visualization
    """
    # Draw face rectangle
    x, y, w, h = face_rect
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw eye rectangles
    for ex, ey, ew, eh in eye_rects:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    
    # Add EAR text
    cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame