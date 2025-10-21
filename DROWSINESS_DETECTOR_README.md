# Driver Drowsiness Detection System

A modern driver drowsiness detection system using **Haar Cascade Classifiers** for face/eye detection and **Vision Transformer (ViT)** for drowsiness classification.

## 🚀 Features

- **Real-time Detection**: Live video stream processing with low latency
- **Haar Cascade Detection**: Robust face and eye detection using OpenCV's built-in cascades
- **Vision Transformer Classification**: State-of-the-art drowsiness classification using pre-trained ViT models
- **Audio Alerts**: Configurable alarm system for drowsiness warnings
- **Visual Overlays**: Real-time bounding boxes and status information
- **Modular Design**: Well-structured, commented code with separate functions for each component
- **Efficient Processing**: Optimized for real-time performance

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Webcam or video input device
- Windows/Linux/macOS

### Python Dependencies
```
opencv-python>=4.8.0
numpy>=1.24.0
pygame>=2.5.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0
Pillow>=9.0.0
```

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation**:
   ```bash
   python test_drowsiness_detector.py
   ```

## 🎯 Usage

### Basic Usage
```bash
python drowsiness_detector.py
```

### Advanced Usage with Options
```bash
python drowsiness_detector.py \
    --camera 0 \
    --width 640 \
    --height 480 \
    --vit-model vit_base_patch16_224 \
    --drowsy-threshold 0.5 \
    --consecutive-frames 5 \
    --show-fps
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera` | 0 | Camera index for video capture |
| `--width` | 640 | Frame width for processing |
| `--height` | 480 | Frame height for processing |
| `--face-cascade` | None | Path to custom face cascade file |
| `--eye-cascade` | None | Path to custom eye cascade file |
| `--vit-model` | vit_base_patch16_224 | Vision Transformer model name |
| `--device` | auto | Device for inference (auto/cpu/cuda) |
| `--drowsy-threshold` | 0.5 | Threshold for drowsiness classification |
| `--consecutive-frames` | 5 | Consecutive drowsy frames to trigger alarm |
| `--alarm-sound` | alarm/alert.wav | Path to alarm sound file |
| `--no-audio` | False | Disable audio alarm |
| `--show-fps` | False | Show FPS counter |

## 🏗️ Architecture

### System Components

1. **Face Detection Module**
   - Uses Haar Cascade Classifier for robust face detection
   - Detects multiple faces and selects the largest one
   - Provides face bounding boxes for further processing

2. **Eye Detection Module**
   - Detects eyes within detected face regions
   - Uses Haar Cascade for eye detection
   - Provides eye bounding boxes for visualization

3. **Vision Transformer Module**
   - Loads pre-trained ViT models from TIMM library
   - Preprocesses face regions for model input
   - Performs drowsiness classification
   - Returns confidence scores and status

4. **Alert System**
   - Tracks consecutive drowsy frames
   - Triggers audio alarm when threshold is reached
   - Provides visual feedback and status updates

5. **User Interface**
   - Real-time video display with overlays
   - Bounding boxes for faces and eyes
   - Status information and confidence scores
   - FPS counter (optional)

### Data Flow

```
Video Stream → Face Detection → Eye Detection → Face Cropping → 
ViT Classification → Drowsiness State → Alert System → UI Display
```

## 🔧 Configuration

### Model Selection
The system supports various Vision Transformer models from TIMM:

- `vit_tiny_patch16_224` - Fastest, lower accuracy
- `vit_small_patch16_224` - Balanced speed/accuracy
- `vit_base_patch16_224` - Default, good accuracy
- `vit_large_patch16_224` - Highest accuracy, slower

### Threshold Tuning
- **drowsy_threshold**: Adjust sensitivity (0.0-1.0)
  - Lower values = more sensitive
  - Higher values = less sensitive
- **consecutive_frames**: Reduce false positives
  - Higher values = fewer false alarms
  - Lower values = faster response

## 🎮 Controls

- **'q'**: Quit the application
- **'r'**: Reset alarm and drowsiness counter

## 📊 Performance

### Expected Performance
- **CPU**: 10-20 FPS (depending on model size)
- **GPU**: 20-40 FPS (with CUDA support)
- **Memory**: 2-4 GB RAM usage
- **Latency**: <100ms end-to-end

### Optimization Tips
1. Use smaller ViT models for better performance
2. Reduce frame resolution for faster processing
3. Enable GPU acceleration with CUDA
4. Adjust detection intervals for efficiency

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure camera is not used by other applications

2. **Model loading fails**
   - Check internet connection for model download
   - Verify PyTorch installation
   - Try different model names

3. **Audio not working**
   - Check if alarm sound file exists
   - Verify pygame installation
   - Use `--no-audio` flag to disable audio

4. **Poor performance**
   - Use smaller ViT models
   - Reduce frame resolution
   - Enable GPU acceleration
   - Close other applications

### Testing
Run the test script to diagnose issues:
```bash
python test_drowsiness_detector.py
```

## 🔬 Technical Details

### Haar Cascade Detection
- **Face Detection**: Uses `haarcascade_frontalface_default.xml`
- **Eye Detection**: Uses `haarcascade_eye.xml`
- **Parameters**: Optimized for real-time performance
- **Fallback**: Built-in OpenCV cascades if custom ones fail

### Vision Transformer Classification
- **Model**: Pre-trained ViT from TIMM library
- **Input**: 224x224 RGB face crops
- **Preprocessing**: Standard ImageNet normalization
- **Output**: Binary classification (Active/Drowsy)
- **Fallback**: Simple CNN if ViT fails to load

### State Management
- **Frame Counting**: Tracks consecutive drowsy frames
- **Alarm Logic**: Triggers after threshold frames
- **Reset Mechanism**: Manual and automatic reset options

## 📁 File Structure

```
DriverDrowsinessDetection/
├── drowsiness_detector.py      # Main application
├── test_drowsiness_detector.py # Test suite
├── requirements.txt            # Dependencies
├── DROWSINESS_DETECTOR_README.md # This file
└── alarm/
    └── alert.wav              # Alarm sound file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- OpenCV team for Haar Cascade implementations
- TIMM library for Vision Transformer models
- PyTorch team for deep learning framework
- Pygame team for audio support

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script
3. Review the error messages
4. Check system requirements

---

**Note**: This system is for educational and research purposes. Always ensure driver safety and follow local regulations when using in vehicles.
