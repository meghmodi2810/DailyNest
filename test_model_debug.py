#!/usr/bin/env python3
"""
Debug script to test emotion model loading and basic functionality
"""

import os
import sys
import numpy as np

def test_tensorflow():
    """Test TensorFlow installation and version"""
    print("Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False

def test_model_loading():
    """Test model loading with detailed error reporting"""
    print("\nTesting model loading...")
    
    try:
        import tensorflow as tf
        
        # Test model paths
        model_paths = [
            'models/face_emotion/emotion_model.hdf5',
            'emotion_model.hdf5'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"✓ Model file found: {model_path}")
                print(f"  File size: {os.path.getsize(model_path):,} bytes")
                
                try:
                    # Try to load the model
                    print(f"  Attempting to load model...")
                    model = tf.keras.models.load_model(model_path, compile=False)
                    print(f"  ✓ Model loaded successfully!")
                    print(f"  Input shape: {model.input_shape}")
                    print(f"  Output shape: {model.output_shape}")
                    
                    # Test with dummy data
                    print(f"  Testing model prediction...")
                    input_shape = model.input_shape
                    if len(input_shape) == 4:
                        dummy_input = np.random.random((1, input_shape[1], input_shape[2], input_shape[3]))
                        prediction = model.predict(dummy_input, verbose=0)
                        print(f"  ✓ Prediction successful!")
                        print(f"  Output shape: {prediction.shape}")
                        print(f"  Sample output: {prediction[0][:3]}")
                        return True
                    else:
                        print(f"  ✗ Unexpected input shape: {input_shape}")
                        
                except Exception as e:
                    print(f"  ✗ Model loading/prediction failed: {e}")
                    print(f"  Error type: {type(e).__name__}")
                    
                    # Try to get more details
                    if hasattr(e, '__cause__') and e.__cause__:
                        print(f"  Caused by: {e.__cause__}")
                    
                    continue
            else:
                print(f"✗ Model file not found: {model_path}")
        
        return False
        
    except Exception as e:
        print(f"✗ General error: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print("\nTesting OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} imported successfully")
        
        # Test face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                print("✗ Face cascade failed to load")
                return False
            else:
                print("✓ Face cascade loaded successfully")
                return True
        else:
            print(f"✗ Face cascade file not found at: {cascade_path}")
            return False
            
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Emotion Model Debug Test")
    print("=" * 60)
    
    # Test TensorFlow
    tf_ok = test_tensorflow()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test OpenCV
    cv_ok = test_opencv()
    
    print("\n" + "=" * 60)
    print("Debug Results Summary")
    print("=" * 60)
    
    print(f"TensorFlow: {'✓ PASSED' if tf_ok else '✗ FAILED'}")
    print(f"Model Loading: {'✓ PASSED' if model_ok else '✗ FAILED'}")
    print(f"OpenCV: {'✓ PASSED' if cv_ok else '✗ FAILED'}")
    
    if tf_ok and model_ok and cv_ok:
        print("\n🎉 All tests passed! The emotion detection system should work correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        
        if not model_ok:
            print("\n🔍 Model Loading Issues:")
            print("- Check if the model file exists and is not corrupted")
            print("- Verify TensorFlow version compatibility")
            print("- Check if the model format is supported")
            print("- Try loading the model in a simple Python script first")
    
    return tf_ok and model_ok and cv_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 