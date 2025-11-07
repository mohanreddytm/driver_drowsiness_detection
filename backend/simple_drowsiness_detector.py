"""
Simplified Driver Drowsiness Detection System
Using only OpenCV and basic computer vision techniques for maximum compatibility.

This version uses:
- Haar Cascade for face and eye detection (with optional DNN fallback)
- Eye Aspect Ratio (EAR) for drowsiness detection
- Temporal smoothing to reduce flickering in face/eye detection
- Exponential moving average for stable position tracking
- Optimized detection parameters for better stability

Improvements:
- Temporal smoothing: Reduces flickering by applying exponential moving average
- Face position tracking: Maintains detection for up to 5 frames even if temporarily lost
- Eye position matching: Matches eyes across frames for smooth tracking
- Enhanced detection parameters: Increased minNeighbors for better stability
"""

import cv2
import numpy as np
import pygame
import time
import argparse
import os
from collections import deque
import logging


class SimpleDrowsinessDetector:
    """
    Simplified drowsiness detector using only OpenCV and basic CV techniques.
    """
    
    def __init__(self, 
                 drowsy_threshold: float = 0.25,
                 consecutive_frames: int = 10,
                 alarm_sound_path: str = "alarm/alert.wav",
                 drowsy_duration_sec: float = 6.0,
                 detection_interval: int = 2,
                 min_face_size: int = 80):
        """
        Initialize the simplified drowsiness detector.
        
        Args:
            drowsy_threshold: EAR threshold for drowsiness detection
            consecutive_frames: Number of consecutive drowsy frames to trigger alarm
            alarm_sound_path: Path to alarm sound file
        """
        self.drowsy_threshold = drowsy_threshold
        self.consecutive_frames = consecutive_frames
        self.alarm_sound_path = alarm_sound_path
        self.drowsy_duration_sec = max(0.0, float(drowsy_duration_sec))
        self.detection_interval = max(1, int(detection_interval))
        self.min_face_size = max(50, int(min_face_size))
        
        # Initialize Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifiers")

        print("[INFO] Loaded Haar cascade classifiers")
        logging.basicConfig(level=logging.INFO)

        # Optional DNN-based face detector (Res10 SSD) for improved accuracy.
        self.use_dnn = True
        self.dnn_confidence = 0.5
        self.model_dir = "models"
        self.dnn_net = None
        self._load_dnn_model()
        
        # Initialize audio system
        self.audio_enabled = self._init_audio()
        
        # State tracking
        self.drowsy_frame_count = 0
        self.alarm_active = False
        self.ear_history = deque(maxlen=10)  # Keep last 10 EAR values for smoothing
        self.last_face_region = None
        self.frame_count = 0
        self._below_start_time = None  # time when EAR went below threshold
        
        # Temporal smoothing for face and eye positions (reduces flickering)
        self.smoothed_face = None  # (x, y, w, h) with exponential moving average
        self.smoothed_eyes = []  # List of smoothed eye positions
        self.face_smoothing_alpha = 0.7  # Higher = more responsive, lower = more stable (0.0-1.0)
        self.eye_smoothing_alpha = 0.6
        self.face_miss_count = 0  # Count consecutive frames without face
        self.max_face_miss = 5  # Keep smoothed face for N frames even if not detected
        
    def _init_audio(self) -> bool:
        """Initialize pygame audio system."""
        try:
            pygame.mixer.init()
            if os.path.exists(self.alarm_sound_path):
                self.alarm_sound = pygame.mixer.Sound(self.alarm_sound_path)
                print(f"[INFO] Loaded alarm sound: {self.alarm_sound_path}")
            else:
                print(f"[WARN] Alarm sound file not found: {self.alarm_sound_path}")
                self.alarm_sound = None
            return True
        except Exception as e:
            print(f"[WARN] Audio initialization failed: {e}")
            return False

    def _load_dnn_model(self):
        """Try to load the lightweight OpenCV DNN face detector model.

        The code will look for the model files in `self.model_dir`:
          - deploy.prototxt
          - res10_300x300_ssd_iter_140000.caffemodel

        If they are not present the detector will continue using Haar cascades.
        """
        if not self.use_dnn:
            logging.info("DNN face detector disabled by configuration")
            return

        proto = os.path.join(self.model_dir, "deploy.prototxt")
        model = os.path.join(self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(proto) and os.path.exists(model):
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
                # Prefer CPU backend by default (works on most Windows setups)
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logging.info(f"[INFO] Loaded DNN face detector from {self.model_dir}")
            except Exception as e:
                logging.warning(f"[WARN] Failed to load DNN model: {e}")
                self.dnn_net = None
        else:
            logging.info("[INFO] DNN model files not found in '%s' - falling back to Haar cascades", self.model_dir)
    
    def _smooth_face_position(self, new_face):
        """Apply exponential moving average to face position for stability."""
        if new_face is None:
            self.face_miss_count += 1
            if self.face_miss_count > self.max_face_miss:
                self.smoothed_face = None
            return self.smoothed_face
        
        self.face_miss_count = 0
        x, y, w, h = new_face
        
        if self.smoothed_face is None:
            self.smoothed_face = (float(x), float(y), float(w), float(h))
        else:
            sx, sy, sw, sh = self.smoothed_face
            # Exponential moving average
            self.smoothed_face = (
                self.face_smoothing_alpha * x + (1 - self.face_smoothing_alpha) * sx,
                self.face_smoothing_alpha * y + (1 - self.face_smoothing_alpha) * sy,
                self.face_smoothing_alpha * w + (1 - self.face_smoothing_alpha) * sw,
                self.face_smoothing_alpha * h + (1 - self.face_smoothing_alpha) * sh
            )
        
        # Return as integer tuple
        return (int(self.smoothed_face[0]), int(self.smoothed_face[1]), 
                int(self.smoothed_face[2]), int(self.smoothed_face[3]))
    
    def _smooth_eye_positions(self, new_eyes):
        """Apply exponential moving average to eye positions for stability."""
        if not new_eyes:
            # If no eyes detected, keep smoothed eyes for a few frames
            if len(self.smoothed_eyes) > 0:
                return self.smoothed_eyes
            return []
        
        # Match new eyes to smoothed eyes by position (simple nearest neighbor)
        smoothed = []
        for new_eye in new_eyes[:2]:  # Max 2 eyes
            nx, ny, nw, nh = new_eye
            
            # Find closest smoothed eye
            best_match = None
            best_dist = float('inf')
            for i, (sx, sy, sw, sh) in enumerate(self.smoothed_eyes):
                # Distance between centers
                dist = ((nx + nw/2) - (sx + sw/2))**2 + ((ny + nh/2) - (sy + sh/2))**2
                if dist < best_dist:
                    best_dist = dist
                    best_match = i
            
            if best_match is not None and best_dist < (nw * nh):  # Within reasonable distance
                # Update existing smoothed eye
                sx, sy, sw, sh = self.smoothed_eyes[best_match]
                smoothed.append((
                    int(self.eye_smoothing_alpha * nx + (1 - self.eye_smoothing_alpha) * sx),
                    int(self.eye_smoothing_alpha * ny + (1 - self.eye_smoothing_alpha) * sy),
                    int(self.eye_smoothing_alpha * nw + (1 - self.eye_smoothing_alpha) * sw),
                    int(self.eye_smoothing_alpha * nh + (1 - self.eye_smoothing_alpha) * sh)
                ))
            else:
                # New eye, add it
                smoothed.append((int(nx), int(ny), int(nw), int(nh)))
        
        # Keep only 2 eyes, sorted left to right
        if len(smoothed) > 2:
            smoothed = sorted(smoothed, key=lambda e: e[0])[:2]
        
        self.smoothed_eyes = smoothed
        return smoothed
    
    def detect_faces_and_eyes(self, frame: np.ndarray) -> tuple:
        """
        Detect faces and eyes in the frame with improved preprocessing and multiple detection methods.
        Uses temporal smoothing to reduce flickering.
        
        Returns:
            Tuple of (faces, eyes) where each is a list of (x, y, w, h) tuples
        """
        # Only run detection every N frames for efficiency
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0 and self.last_face_region is not None:
            # Reuse last detection result for eyes, but still apply smoothing
            x, y, w, h = self.last_face_region
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            eyes = []
            face_eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=4,  # Increased for stability
                minSize=(25, 25)
            )
            for (ex, ey, ew, eh) in face_eyes:
                eyes.append((x + ex, y + ey, ew, eh))
            
            # Apply smoothing
            smoothed_face = self._smooth_face_position(self.last_face_region)
            smoothed_eyes = self._smooth_eye_positions(eyes)
            return ([smoothed_face] if smoothed_face else []), smoothed_eyes
        
        # For performance, run detection on a resized copy if frame is large
        h, w = frame.shape[:2]
        target_max = 800
        scale = 1.0
        small = frame
        if max(w, h) > target_max:
            scale = target_max / float(max(w, h))
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Preprocess gray image for eye detection (always create enhanced version)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        gray_enhanced = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)
        gray_hist = cv2.equalizeHist(gray)

        # First try DNN face detector if available
        faces = []
        if self.dnn_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(cv2.resize(small, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                (dh, dw) = small.shape[:2]
                for i in range(0, detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < self.dnn_confidence:
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([dw, dh, dw, dh])
                    (startX, startY, endX, endY) = box.astype("int")
                    wbox = endX - startX
                    hbox = endY - startY
                    if wbox < int(self.min_face_size * scale) or hbox < int(self.min_face_size * scale):
                        continue
                    # Scale box back to original frame coordinates
                    sx = int(startX / scale)
                    sy = int(startY / scale)
                    ex = int(endX / scale)
                    ey = int(endY / scale)
                    faces.append((sx, sy, ex - sx, ey - sy))
                if len(faces) > 0:
                    logging.debug(f"[DNN] Detected {len(faces)} faces")
            except Exception as e:
                logging.warning(f"[WARN] DNN detection failed, falling back to Haar: {e}")

        # If DNN not used or found nothing, fallback to Haar cascades with enhanced preprocessing
        if len(faces) == 0:
            # Method 1: Enhanced image with CLAHE (gray_enhanced already created above)
            faces1 = self.face_cascade.detectMultiScale(
                gray_enhanced,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(int(self.min_face_size * scale), int(self.min_face_size * scale)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend([(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces1])

            # Method 2: Histogram equalized image (if no faces found)
            if len(faces) == 0:
                faces2 = self.face_cascade.detectMultiScale(
                    gray_hist,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(int(self.min_face_size * scale), int(self.min_face_size * scale)),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                faces.extend([(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces2])

            # Method 3: Original grayscale (if still no faces)
            if len(faces) == 0:
                faces3 = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(int(self.min_face_size * scale), int(self.min_face_size * scale)),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                faces.extend([(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces3])

        # Remove duplicate faces (if any)
        if len(faces) > 1:
            faces = self._remove_duplicate_faces(faces)

        # Store the largest face for reuse
        if len(faces) > 0:
            self.last_face_region = max(faces, key=lambda f: f[2] * f[3])
        else:
            self.last_face_region = None
        
        # Apply temporal smoothing to face position
        smoothed_face = self._smooth_face_position(self.last_face_region)
        
        # Use smoothed face for eye detection if available
        face_for_eye_detection = smoothed_face if smoothed_face else (self.last_face_region if self.last_face_region else None)
        
        # Detect eyes (only in face regions for efficiency)
        eyes = []
        if face_for_eye_detection:
            x, y, w, h = face_for_eye_detection
            # Convert to scaled coordinates for ROI extraction (face coords are in original frame)
            x_scaled = int(x * scale)
            y_scaled = int(y * scale)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)
            
            # Ensure coordinates are within bounds
            h_img, w_img = gray_enhanced.shape[:2]
            x_scaled = max(0, min(x_scaled, w_img - 1))
            y_scaled = max(0, min(y_scaled, h_img - 1))
            w_scaled = min(w_scaled, w_img - x_scaled)
            h_scaled = min(h_scaled, h_img - y_scaled)
            
            face_roi = gray_enhanced[0:0+1, 0:0+1]  # fallback
            try:
                # Extract face ROI from scaled image
                if w_scaled > 0 and h_scaled > 0:
                    face_roi = gray_enhanced[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
                else:
                    face_roi = gray
            except Exception:
                # Fallback to using small-scale gray if indexing fails
                try:
                    if w_scaled > 0 and h_scaled > 0:
                        face_roi = gray[y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled]
                    else:
                        face_roi = gray
                except Exception:
                    face_roi = gray

            # Restrict to upper portion of the face where eyes reside (upper 60%)
            upper_h = max(1, int(face_roi.shape[0] * 0.6))
            eye_search_roi = face_roi[0:upper_h, :]

            # Tune detection parameters to reduce false positives (increased minNeighbors for stability)
            # Calculate min eye size based on face dimensions (in scaled coordinates)
            min_eye_w = max(10, int(w_scaled * 0.08))
            min_eye_h = max(10, int(h_scaled * 0.04))
            face_eyes = self.eye_cascade.detectMultiScale(
                eye_search_roi,
                scaleFactor=1.12,
                minNeighbors=6,  # Increased from 5 for better stability
                minSize=(min_eye_w, min_eye_h)
            )

            # Convert eye coordinates back to original frame coordinates and filter
            candidates = []
            for (ex, ey, ew, eh) in face_eyes:
                # Map coords from roi back to original image (eyes are in scaled ROI coordinates)
                # ROI starts at (x_scaled, y_scaled) in scaled image, so add that offset
                mapped_x = int(x + (ex / scale))
                mapped_y = int(y + (ey / scale))
                mapped_w = int(ew / scale)
                mapped_h = int(eh / scale)

                # Basic geometric filters: eye width should be a reasonable fraction of face width
                if mapped_w < max(8, int(w * 0.06)) or mapped_w > int(w * 0.6):
                    continue
                # Aspect ratio (height/width) for an eye-like box
                ar = mapped_h / float(max(1, mapped_w))
                if ar < 0.12 or ar > 0.9:
                    continue

                candidates.append((mapped_x, mapped_y, mapped_w, mapped_h))

            # Keep up to two largest eye candidates per face (prefer larger, likely true eyes)
            candidates = sorted(candidates, key=lambda r: r[2] * r[3], reverse=True)[:2]
            for c in candidates:
                eyes.append(c)
        
        # Apply temporal smoothing to eye positions
        smoothed_eyes = self._smooth_eye_positions(eyes)
        
        # Return smoothed results
        return ([smoothed_face] if smoothed_face else []), smoothed_eyes
    
    def _remove_duplicate_faces(self, faces: list) -> list:
        """
        Remove duplicate face detections that overlap significantly.
        
        Args:
            faces: List of face bounding boxes
            
        Returns:
            List of unique face bounding boxes
        """
        if len(faces) <= 1:
            return faces
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            for unique_face in unique_faces:
                # Calculate overlap
                x1, y1, w1, h1 = face
                x2, y2, w2, h2 = unique_face
                
                # Calculate intersection
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection = x_overlap * y_overlap
                
                # Calculate union
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                # If overlap is more than 50%, consider it a duplicate
                if union > 0 and intersection / union > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def calculate_eye_aspect_ratio(self, eyes: list) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) from detected eyes.
        
        Args:
            eyes: List of eye bounding boxes
            
        Returns:
            Average EAR value
        """
        if len(eyes) < 2:
            return 0.0
        
        # Calculate EAR for each eye
        ear_values = []
        for (x, y, w, h) in eyes:
            # Simple EAR approximation using bounding box dimensions
            # In a real implementation, you'd use facial landmarks
            ear = h / w  # Height to width ratio
            ear_values.append(ear)
        
        # Return average EAR
        return np.mean(ear_values) if ear_values else 0.0
    
    def classify_drowsiness(self, ear: float) -> tuple:
        """
        Classify drowsiness based on EAR value.
        
        Args:
            ear: Eye Aspect Ratio value
            
        Returns:
            Tuple of (is_drowsy, confidence)
        """
        # Add to history for smoothing
        self.ear_history.append(ear)
        
        # Calculate smoothed EAR
        smoothed_ear = np.mean(list(self.ear_history)) if self.ear_history else ear
        
        # Classify as drowsy if EAR is below threshold
        is_drowsy = smoothed_ear < self.drowsy_threshold
        
        # Calculate confidence based on how far below threshold
        if is_drowsy:
            confidence = min(1.0, (self.drowsy_threshold - smoothed_ear) / self.drowsy_threshold)
        else:
            confidence = min(1.0, (smoothed_ear - self.drowsy_threshold) / self.drowsy_threshold)
        
        return is_drowsy, confidence, smoothed_ear
    
    def update_drowsiness_state(self, is_drowsy: bool) -> bool:
        """
        Update drowsiness state and determine if alarm should be triggered.
        
        Args:
            is_drowsy: Whether current frame shows drowsiness
            
        Returns:
            True if alarm should be active, False otherwise
        """
        now = time.time()
        if is_drowsy:
            if self._below_start_time is None:
                self._below_start_time = now
            self.drowsy_frame_count += 1
        else:
            self._below_start_time = None
            self.drowsy_frame_count = 0
        
        # Time-based alarm: when duration is configured (>0), only use time-based logic
        if self.drowsy_duration_sec > 0:
            if self._below_start_time is None:
                return False
            return (now - self._below_start_time) >= self.drowsy_duration_sec
        
        # Otherwise fallback to frame-based logic
        return self.drowsy_frame_count >= self.consecutive_frames if self.consecutive_frames > 0 else False
    
    def trigger_alarm(self, activate: bool):
        """Trigger or stop the alarm sound."""
        if not self.audio_enabled or self.alarm_sound is None:
            return
        
        if activate and not self.alarm_active:
            self.alarm_sound.play(loops=-1)
            self.alarm_active = True
            print("[ALARM] Drowsiness detected! Alarm activated.")
        elif not activate and self.alarm_active:
            self.alarm_sound.stop()
            self.alarm_active = False
            print("[ALARM] Alarm stopped.")
    
    def draw_overlays(self, frame: np.ndarray, faces: list, eyes: list, 
                     status: str, confidence: float, ear: float, alarm_active: bool) -> np.ndarray:
        """
        Draw bounding boxes and status overlays on the frame.
        """
        # Draw face bounding boxes
        for face in faces:
            try:
                # Ensure coordinates are integers and valid
                if len(face) != 4:
                    continue
                x, y, w, h = face
                x, y, w, h = int(x), int(y), int(w), int(h)
                # Validate coordinates are within frame bounds
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > frame.shape[1] or y + h > frame.shape[0]:
                    # Clamp to frame bounds
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, max(0, y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except (ValueError, TypeError, IndexError) as e:
                print(f"[WARN] Invalid face coordinates: {face}, error: {e}")
                continue
        
        # Draw eye bounding boxes
        for eye in eyes:
            try:
                # Ensure coordinates are integers and valid
                if len(eye) != 4:
                    continue
                x, y, w, h = eye
                x, y, w, h = int(x), int(y), int(w), int(h)
                # Validate coordinates are within frame bounds
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if x + w > frame.shape[1] or y + h > frame.shape[0]:
                    # Clamp to frame bounds
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Eye", (x, max(0, y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except (ValueError, TypeError, IndexError) as e:
                print(f"[WARN] Invalid eye coordinates: {eye}, error: {e}")
                continue
        
        # Determine display status and color based on current state and elapsed time
        display_status = "Active"
        status_color = (0, 255, 0)  # Green
        
        if status == "Drowsy":
            # Calculate elapsed time
            elapsed = 0.0
            if self._below_start_time is not None:
                elapsed = max(0.0, time.time() - self._below_start_time)
            
            if self.drowsy_duration_sec > 0 and elapsed < self.drowsy_duration_sec:
                display_status = "Warning"  # Intermediate state
                status_color = (0, 255, 255)  # Yellow
            else:
                display_status = "Drowsy"  # Alarm state
                status_color = (0, 0, 255)  # Red
        else:
            display_status = "Active"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, f"Status: {display_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Enhanced status display with progress bar
        cv2.putText(frame, f"EAR: {ear:.3f} (Th: {self.drowsy_threshold:.3f})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show time-under-threshold with progress bar
        elapsed = 0.0
        if self._below_start_time is not None:
            elapsed = max(0.0, time.time() - self._below_start_time)
        
        # Progress bar for drowsiness duration
        if self.drowsy_duration_sec > 0 and elapsed > 0:
            progress = min(1.0, elapsed / self.drowsy_duration_sec)
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 120
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Progress bar
            progress_width = int(bar_width * progress)
            bar_color = (0, 255, 255) if progress < 1.0 else (0, 0, 255)  # Yellow to Red
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), bar_color, -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            # Text
            cv2.putText(frame, f"Progress: {elapsed:.1f}s / {self.drowsy_duration_sec:.1f}s", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"Closed Time: {elapsed:.1f}s / {self.drowsy_duration_sec:.1f}s", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Enhanced alarm status with blinking effect
        if alarm_active:
            # Blinking effect based on time
            blink = int(time.time() * 4) % 2  # Blink every 0.25 seconds
            if blink:
                cv2.putText(frame, "🚨 ALARM ACTIVE! 🚨", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Draw red border around frame
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)
        
        return frame


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Driver Drowsiness Detection")
    
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index for video capture")
    parser.add_argument("--width", type=int, default=640, 
                       help="Frame width for processing")
    parser.add_argument("--height", type=int, default=480, 
                       help="Frame height for processing")
    parser.add_argument("--drowsy-threshold", type=float, default=0.25,
                       help="EAR threshold for drowsiness detection")
    parser.add_argument("--drowsy-duration-sec", type=float, default=6.0,
                       help="Seconds eyes must remain closed before alarm")
    parser.add_argument("--consecutive-frames", type=int, default=10,
                       help="Consecutive drowsy frames to trigger alarm")
    parser.add_argument("--detection-interval", type=int, default=2,
                       help="Run detection every N frames (higher = better performance)")
    parser.add_argument("--min-face-size", type=int, default=80,
                       help="Minimum face size for detection (pixels)")
    parser.add_argument("--alarm-sound", type=str, default="alarm/alert.wav",
                       help="Path to alarm sound file")
    parser.add_argument("--no-audio", action="store_true",
                       help="Disable audio alarm")
    parser.add_argument("--show-fps", action="store_true",
                       help="Show FPS counter")
    
    return parser.parse_args()


def main():
    """Main function to run the simplified drowsiness detection system."""
    args = parse_arguments()
    
    print("[INFO] Initializing Simple Driver Drowsiness Detection System...")
    print("[INFO] Using Haar Cascade + Eye Aspect Ratio approach")
    
    # Initialize detector
    try:
        detector = SimpleDrowsinessDetector(
            drowsy_threshold=args.drowsy_threshold,
            consecutive_frames=args.consecutive_frames,
            alarm_sound_path=args.alarm_sound if not args.no_audio else None,
            drowsy_duration_sec=args.drowsy_duration_sec,
            detection_interval=args.detection_interval,
            min_face_size=args.min_face_size
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize detector: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    print(f"[INFO] Camera initialized: {args.width}x{args.height}")
    print("[INFO] Press 'q' to quit, 'r' to reset alarm")
    
    # FPS tracking
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera")
                break
            
            # Detect faces and eyes
            faces, eyes = detector.detect_faces_and_eyes(frame)
            
            status = "No Face"
            confidence = 0.0
            ear = 0.0
            alarm_active = False
            
            # Process if faces are detected
            if len(faces) > 0:
                # Calculate Eye Aspect Ratio
                ear = detector.calculate_eye_aspect_ratio(eyes)
                
                # Classify drowsiness
                is_drowsy, confidence, smoothed_ear = detector.classify_drowsiness(ear)
                
                # Update status (keep internal logic simple)
                status = "Drowsy" if is_drowsy else "Active"
                
                # Update drowsiness state
                alarm_active = detector.update_drowsiness_state(is_drowsy)
                
                # Trigger alarm
                detector.trigger_alarm(alarm_active)
            
            # Draw overlays
            frame = detector.draw_overlays(frame, faces, eyes, status, confidence, ear, alarm_active)
            
            # Draw FPS if requested
            if args.show_fps:
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    current_fps = 30.0 / (current_time - fps_start_time)
                    fps_start_time = current_time
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                           (frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Simple Driver Drowsiness Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset alarm and drowsiness counter
                detector.trigger_alarm(False)
                detector.drowsy_frame_count = 0
                print("[INFO] Alarm reset")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        detector.trigger_alarm(False)
        cap.release()
        cv2.destroyAllWindows()
        if detector.audio_enabled:
            pygame.mixer.quit()
        
        print("[INFO] System shutdown complete")


if __name__ == "__main__":
    main()
