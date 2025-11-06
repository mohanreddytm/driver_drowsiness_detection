"""
Face detection and tracking module using Haar Cascade and Vision Transformer.

This module handles face detection using Haar Cascade, validation using a Vision
Transformer model, and smooth tracking using the KCF tracker.
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
from . import utils

class FaceDetector:
    def __init__(
        self,
        haar_cascade_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize face detection and tracking components.
        
        Args:
            haar_cascade_path: Path to Haar cascade XML file
            device: Device to run ViT inference on
        """
        # Load Haar cascade
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Error loading Haar cascade classifier")
            
        # Initialize ViT
        self.device = device
        self.vit_model = self._load_vit_model()
        self.transform = self._create_transform()
        
        # Initialize tracker
        self.tracker = None
        self.tracking_box = None
        self.box_history = deque(maxlen=5)
        
        # Detection parameters
        self.min_face_size = (60, 60)
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.vit_threshold = 0.85
        self.last_confidence = 0.0
        
    def _load_vit_model(self) -> nn.Module:
        """Load and prepare Vision Transformer model."""
        model = torch.hub.load(
            'facebookresearch/deit:main',
            'deit_base_patch16_224',
            pretrained=True
        )
        model.to(self.device)
        model.eval()
        return model
        
    def _create_transform(self) -> T.Compose:
        """Create image transformation pipeline for ViT."""
        return T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
    def detect_face(
        self,
        frame: np.ndarray,
        gray: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Detect and validate face in frame.
        
        Args:
            frame: Input color frame
            gray: Preprocessed grayscale frame
            
        Returns:
            Tuple of (face_box, confidence)
        """
        # Try tracking first if initialized
        if self.tracker is not None:
            tracking_success, bbox = self.tracker.update(frame)
            
            if tracking_success:
                # Convert from float bbox to int tuple
                bbox = tuple(map(int, bbox))
                x, y, w, h = bbox
                
                # Validate tracked face using ViT
                is_valid, conf = self._validate_face(frame, (x, y, w, h))
                
                if is_valid:
                    self.tracking_box = bbox
                    self.last_confidence = conf
                    return utils.smooth_bbox(bbox, self.box_history), conf
            
            # Tracking failed, reset tracker
            self.tracker = None
        
        # Detect faces using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )
        
        if len(faces) == 0:
            return None, 0.0
            
        # Get largest face
        face_box = max(faces, key=lambda b: b[2] * b[3])
        
        # Validate using ViT
        is_valid, conf = self._validate_face(frame, face_box)
        
        if not is_valid:
            return None, conf
            
        # Initialize new tracker
        x, y, w, h = face_box
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, (x, y, w, h))
        
        self.tracking_box = face_box
        self.last_confidence = conf
        
        return utils.smooth_bbox(face_box, self.box_history), conf
        
    def _validate_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, float]:
        """
        Validate face detection using Vision Transformer.
        
        Args:
            frame: Input frame
            bbox: Face bounding box
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        try:
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Prepare input
            input_tensor = self.transform(face_roi)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.vit_model(input_batch)
                probs = torch.nn.functional.softmax(output, dim=1)
                
                # Use "person" class confidence (index may vary by model)
                conf = float(probs[0, 0].cpu())
                
                return conf >= self.vit_threshold, conf
                
        except Exception as e:
            print(f"Error in face validation: {str(e)}")
            return False, 0.0
            
    def reset_tracker(self):
        """Reset the face tracker."""
        self.tracker = None
        self.tracking_box = None
        self.box_history.clear()