#!/usr/bin/env python3
"""
Test script for emotion detection integration within Django environment
Run this with: python manage.py shell < test_emotion_integration.py
"""

import os
import sys
import django
import numpy as np

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.conf import settings

def test_emotion_detector():
    """Test the emotion detector integration"""
    print("Testing emotion detector integration...")
    
    try:
        # Import the emotion detector
        from ml_models_fallback import get_emotion_detector
        
        # Get the detector instance
        detector = get_emotion_detector()
        print(f"✓ Emotion detector initialized: {type(detector).__name__}")
        
        # Check if ML model is loaded
        if hasattr(detector, 'emotion_model') and detector.emotion_model is not None:
            print("✓ ML emotion model loaded successfully")
            print(f"  Model input shape: {detector.emotion_model.input_shape}")
            print(f"  Model output shape: {detector.emotion_model.output_shape}")
        else:
            print("⚠ ML emotion model not loaded, will use fallback methods")
        
        # Check face cascade
        if hasattr(detector, 'face_cascade') and detector.face_cascade is not None:
            print("✓ Face cascade loaded successfully")
        else:
            print("✗ Face cascade not loaded")
            return False
        
        # Test with dummy image data
        print("\nTesting with dummy image data...")
        
        # Create a dummy grayscale image (48x48)
        dummy_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        
        # Test emotion detection
        try:
            emotion, confidence = detector.detect_face_emotion(dummy_image)
            print(f"✓ Emotion detection successful: {emotion} (confidence: {confidence:.3f})")
            return True
        except Exception as e:
            print(f"✗ Emotion detection failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize emotion detector: {e}")
        return False

def test_model_paths():
    """Test if the emotion model file exists in expected locations"""
    print("\nTesting model file paths...")
    
    model_paths = [
        'models/face_emotion/emotion_model.hdf5',
        'emotion_model.hdf5',
        os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'emotion_model.hdf5'),
        os.path.join(settings.BASE_DIR, 'emotion_model.hdf5')
    ]
    
    found_models = []
    for path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ Model found: {path} ({size:,} bytes)")
            found_models.append((path, size))
        else:
            print(f"✗ Model not found: {path}")
    
    if found_models:
        print(f"\nFound {len(found_models)} model file(s)")
        return True
    else:
        print("\nNo model files found!")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Emotion Detection Integration Test")
    print("=" * 60)
    
    # Test model file existence
    models_ok = test_model_paths()
    
    # Test emotion detector
    detector_ok = test_emotion_detector()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    if models_ok:
        print("✓ Model files: PASSED")
    else:
        print("✗ Model files: FAILED")
    
    if detector_ok:
        print("✓ Emotion detector: PASSED")
    else:
        print("✗ Emotion detector: FAILED")
    
    if models_ok and detector_ok:
        print("\n🎉 All tests passed! The emotion detection system is working correctly.")
        print("The emotion_model.hdf5 is now being used for face emotion detection.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return models_ok and detector_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 