"""
Test script for the Driver Drowsiness Detection System.
This script helps verify that all components are working correctly.
"""

import sys
import os
import cv2
import torch
import numpy as np

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import timm
        print(f"✓ TIMM version: {timm.__version__}")
    except ImportError:
        print("✗ TIMM not found")
        return False
    
    try:
        import pygame
        print(f"✓ Pygame available")
    except ImportError:
        print("✗ Pygame not found")
        return False
    
    return True

def test_haar_cascades():
    """Test if Haar cascade classifiers can be loaded."""
    print("\nTesting Haar cascades...")
    
    try:
        # Test face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("✗ Face cascade failed to load")
            return False
        print("✓ Face cascade loaded successfully")
        
        # Test eye cascade
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if eye_cascade.empty():
            print("✗ Eye cascade failed to load")
            return False
        print("✓ Eye cascade loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Haar cascade test failed: {e}")
        return False

def test_camera():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print("✗ Could not read from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_vit_model():
    """Test if Vision Transformer model can be loaded."""
    print("\nTesting Vision Transformer model...")
    
    try:
        import timm
        
        # Try to create a small ViT model
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=2)
        print("✓ ViT model created successfully")
        
        # Test inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Model inference successful - Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ ViT model test failed: {e}")
        return False

def test_audio():
    """Test if audio system works."""
    print("\nTesting audio system...")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✓ Pygame mixer initialized")
        
        # Check if alarm sound file exists
        alarm_path = "alarm/alert.wav"
        if os.path.exists(alarm_path):
            print(f"✓ Alarm sound file found: {alarm_path}")
        else:
            print(f"⚠ Alarm sound file not found: {alarm_path}")
            print("  The system will work but without audio alerts")
        
        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f"✗ Audio test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Driver Drowsiness Detection System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_haar_cascades,
        test_camera,
        test_vit_model,
        test_audio
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The system should work correctly.")
        print("\nTo run the drowsiness detector:")
        print("  python drowsiness_detector.py")
        print("\nFor help with command line options:")
        print("  python drowsiness_detector.py --help")
    else:
        print("⚠ Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nIf problems persist, check the error messages above.")

if __name__ == "__main__":
    main()
