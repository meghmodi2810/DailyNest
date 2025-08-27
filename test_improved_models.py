#!/usr/bin/env python3
"""
Test improved models functionality
"""

import os
import sys
import django
import numpy as np
import tempfile
import wave

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_face_emotion_detection():
    """Test improved face emotion detection"""
    print("Testing improved face emotion detection...")
    
    try:
        from DailyNest.ml_models_fallback import get_emotion_detector
        
        # Get emotion detector
        detector = get_emotion_detector()
        print("  ✓ Emotion detector loaded successfully")
        
        # Create test image (simulate face)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        emotion, confidence = detector.detect_face_emotion(test_image)
        
        # Validate results
        expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        
        if emotion in expected_emotions:
            print(f"  ✓ Face emotion detection passed: {emotion} (confidence: {confidence:.3f})")
            return True
        else:
            print(f"  ✗ Face emotion detection failed: unexpected emotion '{emotion}'")
            return False
            
    except Exception as e:
        print(f"  ✗ Face emotion detection error: {e}")
        return False

def test_voice_emotion_detection():
    """Test improved voice emotion detection with Whisper"""
    print("Testing improved voice emotion detection...")
    
    try:
        from DailyNest.ml_models_fallback import get_speech_processor
        
        # Get speech processor
        processor = get_speech_processor()
        print("  ✓ Speech processor loaded successfully")
        
        # Create test audio
        test_audio_path = create_test_audio()
        
        try:
            # Test processing
            speech_text, emotion, confidence = processor.process_audio_file(test_audio_path)
            
            # Validate results
            expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
            
            if emotion in expected_emotions:
                print(f"  ✓ Voice emotion detection passed: {emotion} (confidence: {confidence:.3f})")
                if speech_text:
                    print(f"    Whisper transcription: {speech_text[:50]}...")
                return True
            else:
                print(f"  ✗ Voice emotion detection failed: unexpected emotion '{emotion}'")
                return False
                
        finally:
            # Clean up
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
            
    except Exception as e:
        print(f"  ✗ Voice emotion detection error: {e}")
        return False

def create_test_audio():
    """Create a test audio file"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Create a simple sine wave
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name

def test_enhanced_chatbot():
    """Test enhanced chatbot"""
    print("Testing enhanced chatbot...")
    
    try:
        from DailyNest.chatbot_enhanced import get_enhanced_chatbot
        
        # Get chatbot
        chatbot = get_enhanced_chatbot("gemma:2b")
        print("  ✓ Enhanced chatbot loaded successfully")
        
        # Test basic functionality
        stats = chatbot.get_stats()
        print(f"    Model: {stats['model']}")
        print(f"    Connection tested: {stats['connection_tested']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Enhanced chatbot error: {e}")
        return False

def test_model_files():
    """Test that required model files exist"""
    print("Testing model files...")
    
    try:
        from django.conf import settings
        
        # Check face emotion model
        face_model_path = os.path.join(settings.MODELS_DIR, 'face_emotion', 'best_mobilenet_model.h5')
        if os.path.exists(face_model_path):
            size_mb = os.path.getsize(face_model_path) / (1024 * 1024)
            print(f"  ✓ Face emotion model found: {size_mb:.1f} MB")
        else:
            print(f"  ✗ Face emotion model not found: {face_model_path}")
            return False
        
        # Check other model files
        model_files = [
            'models/face_emotion/fer.h5',
            'models/face_emotion/final_mobilenet_model.h5'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"  ✓ {model_file}: {size_mb:.1f} MB")
            else:
                print(f"  ⚠ {model_file}: not found")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model files test error: {e}")
        return False

def main():
    """Run all improved model tests"""
    print("DailyNest Improved Models Test")
    print("=" * 50)
    
    # Test model files
    files_ok = test_model_files()
    print()
    
    # Test face emotion detection
    face_ok = test_face_emotion_detection()
    print()
    
    # Test voice emotion detection
    voice_ok = test_voice_emotion_detection()
    print()
    
    # Test enhanced chatbot
    chatbot_ok = test_enhanced_chatbot()
    print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"  Model Files: {'✓' if files_ok else '✗'}")
    print(f"  Face Emotion: {'✓' if face_ok else '✗'}")
    print(f"  Voice Emotion: {'✓' if voice_ok else '✗'}")
    print(f"  Enhanced Chatbot: {'✓' if chatbot_ok else '✗'}")
    
    all_passed = all([files_ok, face_ok, voice_ok, chatbot_ok])
    
    if all_passed:
        print("\n🎉 All improved model tests passed!")
        print("\nKey improvements:")
        print("  ✓ Using best_mobilenet_model.h5 for face emotion")
        print("  ✓ OpenAI Whisper base for voice transcription")
        print("  ✓ Enhanced chatbot with better memory")
        print("  ✓ Proper error handling and fallbacks")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 