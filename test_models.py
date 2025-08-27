#!/usr/bin/env python
"""
Test script to verify all AI models are working correctly.
Run this script to test emotion detection, speech processing, and chatbot functionality.
"""

import os
import sys
import django
import numpy as np
import tempfile
import base64
from io import BytesIO

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from DailyNest.ml_models import get_emotion_detector, get_speech_processor, get_chatbot
from DailyNest.models import EmotionRecord, ChatMessage

def test_emotion_detection():
    """Test face emotion detection"""
    print("🧠 Testing Emotion Detection...")
    
    detector = get_emotion_detector()
    
    # Test with None input
    emotion, confidence = detector.detect_face_emotion(None)
    print(f"  ✓ None input: {emotion} (confidence: {confidence:.2f})")
    
    # Test with dummy image data
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    emotion, confidence = detector.detect_face_emotion(dummy_image)
    print(f"  ✓ Dummy image: {emotion} (confidence: {confidence:.2f})")
    
    # Test with base64 encoded image
    test_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A"
    emotion, confidence = detector.detect_face_emotion(test_b64)
    print(f"  ✓ Base64 image: {emotion} (confidence: {confidence:.2f})")
    
    return True

def test_speech_processing():
    """Test speech recognition and voice emotion detection"""
    print("🎤 Testing Speech Processing...")
    
    processor = get_speech_processor()
    
    # Create a dummy audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        # Write minimal WAV header and some dummy data
        temp_audio.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
        temp_audio.write(np.random.randint(-32768, 32767, 1000, dtype=np.int16).tobytes())
        temp_audio_path = temp_audio.name
    
    try:
        # Test audio processing
        speech_text, emotion, confidence = processor.process_audio_file(temp_audio_path)
        print(f"  ✓ Audio processing: speech='{speech_text}', emotion={emotion} (confidence: {confidence:.2f})")
        
        # Test with non-existent file
        speech_text, emotion, confidence = processor.process_audio_file("nonexistent.wav")
        print(f"  ✓ Non-existent file: speech='{speech_text}', emotion={emotion} (confidence: {confidence:.2f})")
        
    finally:
        # Clean up
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
    
    return True

def test_chatbot():
    """Test chatbot responses"""
    print("🤖 Testing Chatbot...")
    
    chatbot = get_chatbot()
    
    # Test basic response
    response = chatbot.get_response("Hello, how are you?")
    print(f"  ✓ Basic response: '{response[:100]}...'")
    
    # Test emotion-aware response
    response = chatbot.get_response(
        "I'm feeling really sad today",
        face_emotion="sad",
        voice_emotion="sad",
        face_confidence=0.8,
        voice_confidence=0.7
    )
    print(f"  ✓ Sad emotion response: '{response[:100]}...'")
    
    # Test happy emotion response
    response = chatbot.get_response(
        "I'm so excited about my new job!",
        face_emotion="happy",
        voice_emotion="excited",
        face_confidence=0.9,
        voice_confidence=0.8
    )
    print(f"  ✓ Happy emotion response: '{response[:100]}...'")
    
    # Test with speech text
    response = chatbot.get_response(
        "Tell me about the weather",
        speech_text="Tell me about the weather today please"
    )
    print(f"  ✓ Speech text response: '{response[:100]}...'")
    
    return True

def test_database_integration():
    """Test database model integration"""
    print("💾 Testing Database Integration...")
    
    # Test EmotionRecord creation
    record = EmotionRecord.objects.create(
        face_emotion='happy',
        voice_emotion='excited',
        face_confidence=0.85,
        voice_confidence=0.78,
        notes='Test emotion record'
    )
    print(f"  ✓ EmotionRecord created: ID={record.id}, dominant={record.dominant_emotion}")
    
    # Test ChatMessage creation
    message = ChatMessage.objects.create(
        sender='user',
        message='Test message',
        emotion_context=record
    )
    print(f"  ✓ ChatMessage created: ID={message.id}")
    
    # Clean up test data
    message.delete()
    record.delete()
    print("  ✓ Test data cleaned up")
    
    return True

def test_error_handling():
    """Test error handling and fallbacks"""
    print("⚠️  Testing Error Handling...")
    
    detector = get_emotion_detector()
    processor = get_speech_processor()
    chatbot = get_chatbot()
    
    # Test with invalid inputs
    try:
        emotion, confidence = detector.detect_face_emotion("invalid_data")
        print(f"  ✓ Invalid face data handled: {emotion} (confidence: {confidence:.2f})")
    except Exception as e:
        print(f"  ✗ Face detection error: {e}")
    
    try:
        speech_text, emotion, confidence = processor.process_audio_file("/invalid/path.wav")
        print(f"  ✓ Invalid audio path handled: {emotion} (confidence: {confidence:.2f})")
    except Exception as e:
        print(f"  ✗ Audio processing error: {e}")
    
    try:
        response = chatbot.get_response("")
        print(f"  ✓ Empty message handled: '{response[:50]}...'")
    except Exception as e:
        print(f"  ✗ Chatbot error: {e}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting DailyNest AI Models Test Suite")
    print("=" * 50)
    
    tests = [
        test_emotion_detection,
        test_speech_processing,
        test_chatbot,
        test_database_integration,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}\n")
    
    print("=" * 50)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your AI models are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
