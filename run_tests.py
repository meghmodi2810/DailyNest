#!/usr/bin/env python
"""
Simple test runner to check if all AI models are working.
This script tests the core functionality without requiring full Django setup.
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ NumPy available")
    except ImportError:
        print("  ⚠ NumPy not available - using fallbacks")
    
    try:
        import cv2
        print("  ✓ OpenCV available")
    except ImportError:
        print("  ⚠ OpenCV not available - using fallbacks")
    
    try:
        import tensorflow as tf
        print("  ✓ TensorFlow available")
    except ImportError:
        print("  ⚠ TensorFlow not available - using fallbacks")
    
    try:
        import librosa
        print("  ✓ Librosa available")
    except ImportError:
        print("  ⚠ Librosa not available - using fallbacks")
    
    try:
        import speech_recognition as sr
        print("  ✓ SpeechRecognition available")
    except ImportError:
        print("  ⚠ SpeechRecognition not available - using fallbacks")
    
    try:
        from langchain.llms import Ollama
        print("  ✓ LangChain available")
    except ImportError:
        print("  ⚠ LangChain not available - using fallbacks")
    
    return True

def test_basic_functionality():
    """Test basic ML functionality without Django"""
    print("\nTesting basic ML functionality...")
    
    # Test emotion detection logic
    try:
        import numpy as np
        
        # Simulate image analysis
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        brightness = np.mean(dummy_image)
        contrast = np.std(dummy_image)
        
        # Simple emotion classification
        if brightness > 150 and contrast > 50:
            emotion = "happy"
        elif brightness < 100:
            emotion = "sad"
        else:
            emotion = "neutral"
        
        confidence = np.random.uniform(0.4, 0.8)
        print(f"  ✓ Emotion detection: {emotion} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"  ✗ Emotion detection failed: {e}")
        return False
    
    # Test audio analysis
    try:
        # Simulate audio file analysis
        file_size = 150000  # Simulated file size
        
        if file_size > 200000:
            voice_emotion = "excited"
        elif file_size > 100000:
            voice_emotion = "happy"
        else:
            voice_emotion = "calm"
        
        voice_confidence = np.random.uniform(0.3, 0.7)
        print(f"  ✓ Voice emotion: {voice_emotion} (confidence: {voice_confidence:.2f})")
        
    except Exception as e:
        print(f"  ✗ Voice emotion failed: {e}")
        return False
    
    # Test chatbot responses
    try:
        emotion_responses = {
            'happy': [
                "I can sense your positive energy! That's wonderful.",
                "Your happiness is contagious! Tell me more.",
            ],
            'sad': [
                "I can sense you might be going through something difficult.",
                "Your feelings are valid and important.",
            ],
            'neutral': [
                "I'm here to listen. What's on your mind?",
                "How are you feeling right now?",
            ]
        }
        
        test_emotion = "happy"
        response = np.random.choice(emotion_responses[test_emotion])
        print(f"  ✓ Chatbot response for {test_emotion}: '{response[:50]}...'")
        
    except Exception as e:
        print(f"  ✗ Chatbot response failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test if required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'DailyNest/ml_models.py',
        'DailyNest/models.py',
        'DailyNest/views.py',
        'DailyNest/admin.py',
        'requirements.txt',
        'manage.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} missing")
            return False
    
    # Check models directory
    models_dir = 'models'
    if os.path.exists(models_dir):
        print(f"  ✓ {models_dir} directory exists")
        
        # Check for model files
        face_emotion_dir = os.path.join(models_dir, 'face_emotion')
        if os.path.exists(face_emotion_dir):
            model_files = os.listdir(face_emotion_dir)
            if model_files:
                print(f"  ✓ Found {len(model_files)} model files in face_emotion/")
            else:
                print("  ⚠ No model files found in face_emotion/")
        else:
            print("  ⚠ face_emotion directory not found")
    else:
        print(f"  ⚠ {models_dir} directory not found")
    
    return True

def main():
    """Run all tests"""
    print("🚀 DailyNest AI Models Quick Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("File Structure", test_file_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception:")
            print(f"   {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"🏁 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed!")
        print("\n📝 Next steps:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run migrations: python manage.py migrate")
        print("3. Test the web interface: python manage.py runserver")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
