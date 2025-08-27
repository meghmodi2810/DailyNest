#!/usr/bin/env python3
"""
Test face emotion inference
"""

import os
import sys
import django
import numpy as np
from PIL import Image

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from DailyNest.ml_models_fixed import get_emotion_detector

def create_test_image():
    """Create a simple test image"""
    # Create a 48x48 grayscale image
    img_array = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img_array

def test_face_emotion_detection():
    """Test face emotion detection"""
    print("Testing face emotion detection...")
    
    try:
        # Get emotion detector
        detector = get_emotion_detector()
        
        # Create test image
        test_image = create_test_image()
        
        # Test detection
        emotion, confidence = detector.detect_face_emotion(test_image)
        
        # Validate results
        expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        
        if emotion in expected_emotions:
            print(f"✓ Face emotion detection passed: {emotion} (confidence: {confidence:.2f})")
            return True
        else:
            print(f"✗ Face emotion detection failed: unexpected emotion '{emotion}'")
            return False
            
    except Exception as e:
        print(f"✗ Face emotion detection error: {e}")
        return False

if __name__ == "__main__":
    success = test_face_emotion_detection()
    sys.exit(0 if success else 1) 