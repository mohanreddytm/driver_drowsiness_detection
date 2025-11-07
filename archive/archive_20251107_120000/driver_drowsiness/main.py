"""
Main module for the Driver Drowsiness Detection system.

This module handles video capture, orchestrates the detection pipeline,
and manages the display output.
"""

import cv2
import numpy as np
import threading
from typing import Tuple, Optional
import os
import time

from . import utils
from .detector import FaceDetector
from .drowsiness_logic import DrowsinessDetector

class DrowsinessDetectionSystem:
    def __init__(self):
        """Initialize the drowsiness detection system."""
        # Load models
        model_dir = os.path.join(os.path.dirname(__file__), 'haarcascade')
        self.face_detector = FaceDetector(
            os.path.join(model_dir, 'haarcascade_frontalface_default.xml')
        )
        self.drowsiness_detector = DrowsinessDetector(
            os.path.join(model_dir, 'haarcascade_eye.xml')
        )
        
        # Initialize video capture with threading
        self.cap = None
        self.frame_queue = []
        self.cap_thread = None
        self.running = False
        
        # Initialize alert system
        self.alert_manager = utils.AlertManager()
        
    def _capture_frames(self):
        """Continuously capture frames in a separate thread."""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Keep only the most recent frame
            self.frame_queue = [frame]
            time.sleep(0.01)  # Small delay to prevent overload
            
    def start(self, camera_id: int = 0):
        """
        Start the drowsiness detection system.
        
        Args:
            camera_id: Camera device ID
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        # Start capture thread
        self.running = True
        self.cap_thread = threading.Thread(target=self._capture_frames)
        self.cap_thread.daemon = True
        self.cap_thread.start()
        
        frame_counter = 0
        process_this_frame = True
        
        try:
            while True:
                # Get the latest frame
                if not self.frame_queue:
                    continue
                    
                frame = self.frame_queue[0]
                frame_counter += 1
                
                # Process every other frame for efficiency
                if process_this_frame:
                    # Preprocess frame
                    frame, gray = utils.preprocess_frame(frame)
                    
                    # Detect and validate face
                    face_box, vit_conf = self.face_detector.detect_face(frame, gray)
                    
                    if face_box:
                        # Detect eyes and calculate EAR
                        eye_boxes = self.drowsiness_detector.detect_eyes(frame, face_box)
                        ear_value = self.drowsiness_detector.calculate_ear(frame, eye_boxes)
                        
                        # Update drowsiness state
                        is_drowsy = self.drowsiness_detector.update_drowsiness_state(ear_value)
                        
                        # Trigger alert if drowsy
                        if is_drowsy:
                            self.alert_manager.play_alert()
                            
                        # Draw debug visualization
                        frame = utils.draw_debug_info(
                            frame, face_box, eye_boxes,
                            ear_value, is_drowsy, vit_conf
                        )
                    else:
                        # Reset drowsiness state if no face detected
                        self.drowsiness_detector.below_threshold_frames = 0
                        
                process_this_frame = not process_this_frame
                
                # Display output
                cv2.imshow('Driver Drowsiness Detection', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop()
            
    def stop(self):
        """Stop the system and release resources."""
        self.running = False
        if self.cap_thread:
            self.cap_thread.join()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point for the application."""
    try:
        system = DrowsinessDetectionSystem()
        system.start()
    except Exception as e:
        print(f"Error: {str(e)}")
        
if __name__ == '__main__':
    main()