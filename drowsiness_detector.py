"""
Driver Drowsiness Detection System
Using Haar Cascade Classifier for face/eye detection and Vision Transformer for drowsiness classification.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import pygame
import time
import argparse
import os
from collections import deque
from typing import Tuple, Optional, List


class DrowsinessDetector:
    """
    Main class for driver drowsiness detection using Haar Cascade and Vision Transformer.
    """
    
    def __init__(self, 
                 face_cascade_path: str = None,
                 eye_cascade_path: str = None,
                 vit_model_name: str = "vit_base_patch16_224",
                 device: str = "auto",
                 drowsy_threshold: float = 0.5,
                 consecutive_frames: int = 5,
                 alarm_sound_path: str = "alarm/alert.wav",
                 drowsy_duration_sec: float = 6.0):
        """
        Initialize the drowsiness detector.
        
        Args:
            face_cascade_path: Path to Haar cascade for face detection
            eye_cascade_path: Path to Haar cascade for eye detection  
            vit_model_name: Name of the Vision Transformer model to use
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            drowsy_threshold: Threshold for drowsiness classification (0-1)
            consecutive_frames: Number of consecutive drowsy frames to trigger alarm
            alarm_sound_path: Path to alarm sound file
        """
        self.drowsy_threshold = drowsy_threshold
        self.consecutive_frames = consecutive_frames
        self.alarm_sound_path = alarm_sound_path
        self.drowsy_duration_sec = max(0.0, float(drowsy_duration_sec))
        
        # Initialize device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Using device: {self.device}")
        
        # Initialize Haar cascades
        self.face_cascade = self._load_haar_cascade(face_cascade_path, "face")
        self.eye_cascade = self._load_haar_cascade(eye_cascade_path, "eye")
        
        # Initialize Vision Transformer model
        self.vit_model = self._load_vit_model(vit_model_name)
        
        # Initialize image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize audio system
        self.audio_enabled = self._init_audio()
        
        # State tracking
        self.drowsy_frame_count = 0
        self.alarm_active = False
        self.last_face_region = None
        self._below_start_time = None
        
    def _load_haar_cascade(self, cascade_path: Optional[str], cascade_type: str) -> cv2.CascadeClassifier:
        """
        Load Haar cascade classifier for face or eye detection.
        
        Args:
            cascade_path: Path to cascade file (if None, use OpenCV default)
            cascade_type: Type of cascade ('face' or 'eye')
            
        Returns:
            Loaded cascade classifier
        """
        if cascade_path and os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Use OpenCV's built-in cascades
            if cascade_type == "face":
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            elif cascade_type == "eye":
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            else:
                raise ValueError(f"Unknown cascade type: {cascade_type}")
        
        if cascade.empty():
            raise RuntimeError(f"Failed to load {cascade_type} cascade classifier")
        
        print(f"[INFO] Loaded {cascade_type} cascade classifier")
        return cascade
    
    def _load_vit_model(self, model_name: str) -> torch.nn.Module:
        """
        Load and prepare Vision Transformer model for drowsiness classification.
        
        Args:
            model_name: Name of the ViT model from timm
            
        Returns:
            Loaded and prepared ViT model
        """
        try:
            # Load pre-trained ViT model
            model = timm.create_model(model_name, pretrained=True, num_classes=2)
            model = model.to(self.device)
            model.eval()
            
            print(f"[INFO] Loaded ViT model: {model_name}")
            print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load ViT model: {e}")
            print("[INFO] Falling back to a simple CNN model...")
            
            # Fallback to a simple model if ViT fails
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> torch.nn.Module:
        """
        Create a simple fallback CNN model for drowsiness classification.
        
        Returns:
            Simple CNN model
        """
        class SimpleDrowsinessCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(256 * 7 * 7, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, 2)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = SimpleDrowsinessCNN().to(self.device)
        model.eval()
        print("[INFO] Created fallback CNN model")
        return model
    
    def _init_audio(self) -> bool:
        """
        Initialize pygame audio system for alarm sounds.
        
        Returns:
            True if audio initialization successful, False otherwise
        """
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
    
    def detect_faces_and_eyes(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detect faces and eyes in the frame using Haar cascades.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (faces, eyes) where each is a list of (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE for low-light improvement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect eyes (only in face regions for efficiency)
        eyes = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            # Convert eye coordinates back to full frame
            for (ex, ey, ew, eh) in face_eyes:
                eyes.append((x + ex, y + ey, ew, eh))
        
        return faces, eyes
    
    def preprocess_face_for_vit(self, frame: np.ndarray, face_bbox: Tuple) -> torch.Tensor:
        """
        Preprocess face region for Vision Transformer input.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Preprocessed tensor ready for ViT model
        """
        x, y, w, h = face_bbox
        
        # Extract face region with some padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_region = frame[y1:y2, x1:x2]
        
        # Convert BGR to RGB
        face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        face_tensor = self.transform(face_region)
        
        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        return face_tensor
    
    def classify_drowsiness(self, face_tensor: torch.Tensor) -> Tuple[float, str]:
        """
        Classify drowsiness using Vision Transformer.
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            Tuple of (confidence, status) where status is 'Active' or 'Drowsy'
        """
        with torch.no_grad():
            try:
                # Get model predictions
                outputs = self.vit_model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get drowsiness probability (assuming class 1 is drowsy)
                drowsy_prob = probabilities[0][1].item()
                
                # Determine status
                if drowsy_prob > self.drowsy_threshold:
                    status = "Drowsy"
                else:
                    status = "Active"
                
                return drowsy_prob, status
                
            except Exception as e:
                print(f"[ERROR] Classification failed: {e}")
                # Return default values on error
                return 0.0, "Active"
    
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
        """
        Trigger or stop the alarm sound.
        
        Args:
            activate: True to start alarm, False to stop
        """
        if not self.audio_enabled or self.alarm_sound is None:
            return
        
        if activate and not self.alarm_active:
            self.alarm_sound.play(loops=-1)  # Loop indefinitely
            self.alarm_active = True
            print("[ALARM] Drowsiness detected! Alarm activated.")
        elif not activate and self.alarm_active:
            self.alarm_sound.stop()
            self.alarm_active = False
            print("[ALARM] Alarm stopped.")
    
    def draw_overlays(self, frame: np.ndarray, faces: List[Tuple], eyes: List[Tuple], 
                     status: str, confidence: float, alarm_active: bool) -> np.ndarray:
        """
        Draw bounding boxes and status overlays on the frame.
        
        Args:
            frame: Input frame
            faces: List of face bounding boxes
            eyes: List of eye bounding boxes
            status: Current status ('Active' or 'Drowsy')
            confidence: Drowsiness confidence score
            alarm_active: Whether alarm is currently active
            
        Returns:
            Frame with overlays drawn
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
        
        # Draw status information
        status_color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Drowsy Frames: {self.drowsy_frame_count}/{self.consecutive_frames}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw alarm status
        if alarm_active:
            cv2.putText(frame, "ALARM ACTIVE!", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection with Haar Cascade and ViT")
    
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index for video capture")
    parser.add_argument("--width", type=int, default=640, 
                       help="Frame width for processing")
    parser.add_argument("--height", type=int, default=480, 
                       help="Frame height for processing")
    parser.add_argument("--face-cascade", type=str, default=None,
                       help="Path to face cascade file")
    parser.add_argument("--eye-cascade", type=str, default=None,
                       help="Path to eye cascade file")
    parser.add_argument("--vit-model", type=str, default="vit_base_patch16_224",
                       help="Vision Transformer model name")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device for model inference")
    parser.add_argument("--drowsy-threshold", type=float, default=0.5,
                       help="Threshold for drowsiness classification")
    parser.add_argument("--consecutive-frames", type=int, default=5,
                       help="Consecutive drowsy frames to trigger alarm")
    parser.add_argument("--drowsy-duration-sec", type=float, default=6.0,
                       help="Seconds in drowsy state required before alarm")
    parser.add_argument("--alarm-sound", type=str, default="alarm/alert.wav",
                       help="Path to alarm sound file")
    parser.add_argument("--no-audio", action="store_true",
                       help="Disable audio alarm")
    parser.add_argument("--show-fps", action="store_true",
                       help="Show FPS counter")
    
    return parser.parse_args()


def main():
    """Main function to run the drowsiness detection system."""
    args = parse_arguments()
    
    print("[INFO] Initializing Driver Drowsiness Detection System...")
    print(f"[INFO] Using Haar Cascade + Vision Transformer approach")
    
    # Initialize detector
    try:
        detector = DrowsinessDetector(
            face_cascade_path=args.face_cascade,
            eye_cascade_path=args.eye_cascade,
            vit_model_name=args.vit_model,
            device=args.device,
            drowsy_threshold=args.drowsy_threshold,
            consecutive_frames=args.consecutive_frames,
            alarm_sound_path=args.alarm_sound if not args.no_audio else None,
            drowsy_duration_sec=args.drowsy_duration_sec
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
            alarm_active = False
            
            # Process the largest face if detected
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                
                # Preprocess face for ViT
                face_tensor = detector.preprocess_face_for_vit(frame, largest_face)
                
                # Classify drowsiness
                confidence, status = detector.classify_drowsiness(face_tensor)
                
                # Update drowsiness state
                is_drowsy = status == "Drowsy"
                alarm_active = detector.update_drowsiness_state(is_drowsy)
                
                # Trigger alarm
                detector.trigger_alarm(alarm_active)
            
            # Draw overlays
            frame = detector.draw_overlays(frame, faces, eyes, status, confidence, alarm_active)
            
            # Draw FPS if requested
            if args.show_fps:
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    current_time = time.time()
                    current_fps = 30.0 / (current_time - fps_start_time)
                    fps_start_time = current_time
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                           (frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Driver Drowsiness Detection", frame)
            
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
