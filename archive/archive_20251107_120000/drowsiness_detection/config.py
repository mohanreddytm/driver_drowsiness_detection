"""
Configuration settings for the drowsiness detection system.

This module contains all configurable parameters for the detection pipeline including
model paths, detection thresholds, and performance settings.
"""

# Path configurations
HAAR_CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
HAAR_EYE_CASCADE_PATH = "models/haarcascade_eye.xml"
VIT_MODEL_PATH = "models/vit_face_validator.pth"

# Detection parameters
MIN_FACE_SIZE = (30, 30)
FACE_DETECTION_SCALE = 1.1
FACE_DETECTION_NEIGHBORS = 5
EYE_DETECTION_SCALE = 1.1
EYE_DETECTION_NEIGHBORS = 3

# Vision Transformer settings
VIT_IMAGE_SIZE = 224
VIT_CONFIDENCE_THRESHOLD = 0.85

# Tracking parameters
MAX_TRACKING_AGE = 30  # frames
MIN_TRACKING_VISIBILITY = 0.6
SMOOTHING_WINDOW = 5

# EAR parameters
EAR_THRESHOLD = 0.25
EAR_CONSECUTIVE_FRAMES = 20
EAR_SMOOTHING_WINDOW = 3

# Alert settings
ALERT_COOLDOWN = 5.0  # seconds
ALERT_MIN_CONFIDENCE = 0.8