# Libraries Used in Driver Drowsiness Detection System

This document lists all the libraries and technologies used in this project, along with their purposes and why they were chosen.

## Core Computer Vision Libraries

### OpenCV (cv2)
- **Version**: >= 4.8.0
- **Purpose**: Primary computer vision library for image processing, face detection, and video capture
- **Why Used**: 
  - Industry standard for computer vision tasks
  - Excellent performance and cross-platform support
  - Built-in Haar Cascade classifiers for face and eye detection
  - DNN (Deep Neural Network) support for advanced face detection
  - Comprehensive image preprocessing capabilities (CLAHE, Gaussian blur, histogram equalization)

### NumPy
- **Version**: >= 1.24.0
- **Purpose**: Numerical computing and array operations
- **Why Used**:
  - Essential for image data manipulation
  - Efficient array operations for face/eye coordinate calculations
  - Required dependency for OpenCV

## Face Detection Methods

### 1. Haar Cascade Classifiers (Primary)
- **Library**: OpenCV built-in
- **Files**: 
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye.xml`
- **Why Used**:
  - Fast and lightweight
  - Works well in good lighting conditions
  - No external model files required (built into OpenCV)
  - Good balance between speed and accuracy
- **Improvements Made**:
  - Added temporal smoothing to reduce flickering
  - Multiple preprocessing techniques (CLAHE, histogram equalization)
  - Optimized detection parameters for stability

### 2. DNN Face Detector (Optional, Better Accuracy)
- **Library**: OpenCV DNN module
- **Model**: Res10 SSD (300x300)
- **Files Required**:
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`
- **Why Used**:
  - More accurate than Haar cascades
  - Better performance in challenging lighting conditions
  - Modern deep learning-based approach
- **Status**: Falls back to Haar if model files not found

### 3. MediaPipe Face Mesh (Optional, Best Quality)
- **Library**: MediaPipe
- **Purpose**: Advanced face landmark detection for precise eye tracking
- **Why Used**:
  - Most accurate face and eye detection
  - Provides 468 facial landmarks
  - Excellent for precise EAR (Eye Aspect Ratio) calculation
  - Better stability and less flickering
- **Status**: Available as alternative implementation (`backend/mediapipe_drowsiness.py`)
- **Note**: Requires `pip install mediapipe`

## Backend Framework

### FastAPI
- **Version**: >= 0.115.0
- **Purpose**: Modern, fast web framework for building the API
- **Why Used**:
  - High performance (async support)
  - Automatic API documentation
  - Type hints and validation
  - WebSocket support for real-time status updates

### Uvicorn
- **Version**: >= 0.30.0
- **Purpose**: ASGI server for FastAPI
- **Why Used**:
  - Fast and efficient
  - Supports async operations
  - Production-ready

## Audio Libraries

### pyttsx3
- **Version**: >= 2.90
- **Purpose**: Text-to-Speech engine for custom alert messages
- **Why Used**:
  - Cross-platform TTS support
  - No external dependencies
  - Works offline
  - Supports custom voice messages

### pygame
- **Version**: >= 2.5.0
- **Purpose**: Audio playback for default alarm sounds
- **Why Used**:
  - Reliable audio playback
  - Cross-platform support
  - Good for continuous sound loops

## Detection Improvements

### Temporal Smoothing
- **Technique**: Exponential Moving Average (EMA)
- **Purpose**: Reduces flickering in face and eye detection
- **Implementation**:
  - Face position smoothing: alpha = 0.7
  - Eye position smoothing: alpha = 0.6
  - Maintains detection for up to 5 frames even if face temporarily not detected

### Detection Parameters
- **Face Detection**:
  - `scaleFactor`: 1.05-1.1 (balance between speed and accuracy)
  - `minNeighbors`: 4-5 (reduces false positives)
  - `minSize`: 80x80 pixels (filters small detections)
- **Eye Detection**:
  - `scaleFactor`: 1.12 (more conservative for stability)
  - `minNeighbors`: 6 (increased for better stability)
  - Search restricted to upper 60% of face region

## Why These Libraries?

1. **OpenCV + Haar Cascades**: 
   - Primary choice for reliability and speed
   - Works out of the box without additional downloads
   - Good performance on most systems

2. **Temporal Smoothing**:
   - Custom implementation to reduce flickering
   - Uses exponential moving average for smooth transitions
   - Maintains detection continuity between frames

3. **Multiple Detection Methods**:
   - Haar Cascade: Fast, always available
   - DNN: Better accuracy when model files available
   - MediaPipe: Best quality, optional enhancement

## Performance Optimizations

1. **Frame Skipping**: Detection runs every N frames (configurable)
2. **Image Resizing**: Large frames resized to 800px max for processing
3. **ROI Processing**: Eye detection limited to face region only
4. **Temporal Smoothing**: Reduces computational overhead from constant re-detection

## Future Enhancements

- MediaPipe integration as primary detector (when available)
- GPU acceleration for DNN models
- Additional face detection backends (MTCNN, RetinaFace)

---

**Last Updated**: 2024
**Maintained By**: Driver Drowsiness Detection Project

