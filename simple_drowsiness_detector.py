"""
Simplified Driver Drowsiness Detection System
Using only OpenCV and basic computer vision techniques for maximum compatibility.

This version uses:
- Haar Cascade for face and eye detection
- Eye Aspect Ratio (EAR) for drowsiness detection
- Simple threshold-based classification
"""

import cv2
import numpy as np
import pygame
import time
import argparse
import os
from collections import deque


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
        
        # Initialize audio system
        self.audio_enabled = self._init_audio()
        
        # State tracking
        self.drowsy_frame_count = 0
        self.alarm_active = False
        self.ear_history = deque(maxlen=10)  # Keep last 10 EAR values for smoothing
        self.last_face_region = None
        self.frame_count = 0
        self._below_start_time = None  # time when EAR went below threshold
        
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
    
    def detect_faces_and_eyes(self, frame: np.ndarray) -> tuple:
        """
        Detect faces and eyes in the frame with optimized performance.
        
        Returns:
            Tuple of (faces, eyes) where each is a list of (x, y, w, h) tuples
        """
        # Only run detection every N frames for efficiency
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0 and self.last_face_region is not None:
            # Reuse last detection result
            return [self.last_face_region], []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve low-light performance using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Store the largest face for reuse
        if len(faces) > 0:
            self.last_face_region = max(faces, key=lambda f: f[2] * f[3])
        else:
            self.last_face_region = None
        
        # Detect eyes (only in face regions for efficiency)
        eyes = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(25, 25)
            )
            # Convert eye coordinates back to full frame
            for (ex, ey, ew, eh) in face_eyes:
                eyes.append((x + ex, y + ey, ew, eh))
        
        return faces, eyes
    
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
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw eye bounding boxes
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Eye", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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
