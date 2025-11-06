"""
Smooth tracking module for face and eye regions.

This module implements temporal smoothing and prediction for stable
tracking of detected faces and eyes across video frames.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict
from . import config

class Track:
    def __init__(self, rect: Tuple[int, int, int, int]):
        """
        Initialize a new track for a detected region.
        
        Args:
            rect: Initial bounding box (x, y, w, h)
        """
        self.current_rect = rect
        self.predicted_rect = rect
        self.age = 0
        self.hits = 1
        self.misses = 0
        
        # History for smoothing
        self.history = deque([rect], maxlen=config.SMOOTHING_WINDOW)
        
    def update(self, rect: Optional[Tuple[int, int, int, int]] = None):
        """
        Update track with new detection or mark as missed.
        
        Args:
            rect: New detection rectangle or None if missed
        """
        if rect is not None:
            self.hits += 1
            self.misses = 0
            self.current_rect = rect
            self.history.append(rect)
            self.predicted_rect = self._smooth_position()
        else:
            self.misses += 1
            self.current_rect = self.predicted_rect
            
        self.age += 1
        
    def _smooth_position(self) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing to track position."""
        if len(self.history) < 2:
            return self.current_rect
            
        # Calculate smoothed coordinates
        x = np.mean([r[0] for r in self.history])
        y = np.mean([r[1] for r in self.history])
        w = np.mean([r[2] for r in self.history])
        h = np.mean([r[3] for r in self.history])
        
        return (int(x), int(y), int(w), int(h))
        
    @property
    def is_valid(self) -> bool:
        """Check if track is still valid based on age and visibility."""
        if self.age > config.MAX_TRACKING_AGE:
            return False
            
        visibility = self.hits / (self.hits + self.misses)
        return visibility >= config.MIN_TRACKING_VISIBILITY

class SmoothTracker:
    def __init__(self):
        """Initialize the smooth tracker."""
        self.face_track: Optional[Track] = None
        self.eye_tracks: List[Track] = []
        
    def update(
        self,
        face_rect: Optional[Tuple[int, int, int, int]],
        eye_rects: List[Tuple[int, int, int, int]]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """
        Update tracking with new detections.
        
        Args:
            face_rect: Detected face rectangle or None
            eye_rects: List of detected eye rectangles
            
        Returns:
            Tuple of (smoothed_face_rect, smoothed_eye_rects)
        """
        # Update face track
        if face_rect is not None:
            if self.face_track is None:
                self.face_track = Track(face_rect)
            else:
                self.face_track.update(face_rect)
        elif self.face_track is not None:
            self.face_track.update(None)
            
        # Clear invalid face track
        if self.face_track and not self.face_track.is_valid:
            self.face_track = None
            self.eye_tracks = []
            
        # Update eye tracks if face is tracked
        smoothed_eye_rects = []
        if self.face_track:
            # Match and update existing eye tracks
            if len(self.eye_tracks) == 0:
                self.eye_tracks = [Track(rect) for rect in eye_rects[:2]]
            else:
                # Update existing tracks with closest detections
                for track in self.eye_tracks:
                    closest_eye = self._find_closest_eye(track.current_rect, eye_rects)
                    track.update(closest_eye)
                    
            # Get smoothed eye positions
            smoothed_eye_rects = [
                track.predicted_rect for track in self.eye_tracks
                if track.is_valid
            ]
            
        return (
            self.face_track.predicted_rect if self.face_track else None,
            smoothed_eye_rects
        )
        
    def _find_closest_eye(
        self,
        track_rect: Tuple[int, int, int, int],
        detections: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find the closest eye detection to a tracked position."""
        if not detections:
            return None
            
        # Calculate center points
        track_center = (track_rect[0] + track_rect[2]//2,
                       track_rect[1] + track_rect[3]//2)
                       
        min_dist = float('inf')
        closest_rect = None
        
        for rect in detections:
            center = (rect[0] + rect[2]//2,
                     rect[1] + rect[3]//2)
            dist = np.sqrt((center[0] - track_center[0])**2 +
                         (center[1] - track_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_rect = rect
                
        return closest_rect