#!/usr/bin/env python
"""
Production test script for DailyNest AI models.
Tests all components with comprehensive error handling and reporting.
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

from DailyNest.ml_models_unified import get_emotion_detector, get_speech_processor, get_chatbot
from DailyNest.models import EmotionRecord, ChatMessage

def test_emotion_detection():
    """Test face emotion detection with various inputs"""
    print("Testing Production Emotion Detection...")
    
    detector = get_emotion_detector()
    
    # Test 1: None input
    emotion, confidence = detector.detect_face_emotion(None)
    print(f"  ✓ None input: {emotion} (confidence: {confidence:.2f})")
    assert emotion == "neutral"
    
    # Test 2: Random image array
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    emotion, confidence = detector.detect_face_emotion(dummy_image)
    print(f"  ✓ Random image: {emotion} (confidence: {confidence:.2f})")
    assert emotion in detector.emotion_labels
    assert 0.0 <= confidence <= 1.0
    
    # Test 3: Base64 image
    test_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A"
    emotion, confidence = detector.detect_face_emotion(test_b64)
    print(f"  ✓ Base64 image: {emotion} (confidence: {confidence:.2f})")
    
    return True

def test_speech_processing():
    """Test speech recognition and voice emotion detection"""
    print("Testing Production Speech Processing...")
    
    processor = get_speech_processor()
    
    # Create dummy WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        # Write minimal WAV header
        temp_audio.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x22\x56\x00\x00\x44\xac\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
        temp_audio.write(np.random.randint(-32768, 32767, 1000, dtype=np.int16).tobytes())
        temp_audio_path = temp_audio.name
    
    try:
        # Test audio processing
        speech_text, emotion, confidence = processor.process_audio_file(temp_audio_path)
        print(f"  ✓ Audio processing: speech='{speech_text}', emotion={emotion} (confidence: {confidence:.2f})")
        assert 0.0 <= confidence <= 1.0
        
        # Test non-existent file
        speech_text, emotion, confidence = processor.process_audio_file("nonexistent.wav")
        print(f"  ✓ Non-existent file: emotion={emotion} (confidence: {confidence:.2f})")
        assert emotion == "neutral"
        
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
    
    return True

def test_chatbot():
    """Test chatbot responses with emotion awareness"""
    print("Testing Production Chatbot...")
    
    chatbot = get_chatbot()
    
    # Test basic response
    response = chatbot.get_response("Hello, how are you?")
    print(f"  ✓ Basic response: '{response[:60]}...'")
    assert len(response) > 10
    
    # Test emotion-aware responses
    test_cases = [
        ("I'm feeling really sad today", "sad", 0.8),
        ("I'm so excited about my new job!", "happy", 0.9),
        ("I'm worried about my health", "fear", 0.7),
        ("This is so frustrating!", "angry", 0.8),
    ]
    
    for message, emotion, confidence in test_cases:
        response = chatbot.get_response(
            message, 
            face_emotion=emotion, 
            face_confidence=confidence
        )
        print(f"  ✓ {emotion.title()} response: '{response[:60]}...'")
        assert len(response) > 10
    
    return True

def test_database_integration():
    """Test database model integration"""
    print("Testing Database Integration...")
    
    # Test EmotionRecord creation
    record = EmotionRecord.objects.create(
        face_emotion='happy',
        voice_emotion='excited',
        face_confidence=0.85,
        voice_confidence=0.78,
        notes='Production test record'
    )
    print(f"  ✓ EmotionRecord created: ID={record.id}, dominant={record.dominant_emotion}")
    
    # Test ChatMessage creation
    message = ChatMessage.objects.create(
        sender='user',
        message='Production test message',
        emotion_context=record
    )
    print(f"  ✓ ChatMessage created: ID={message.id}")
    
    # Test queries
    recent_emotions = EmotionRecord.objects.order_by('-timestamp')[:5]
    print(f"  ✓ Recent emotions query: {len(recent_emotions)} records")
    
    recent_messages = ChatMessage.objects.order_by('-timestamp')[:5]
    print(f"  ✓ Recent messages query: {len(recent_messages)} messages")
    
    # Clean up
    message.delete()
    record.delete()
    print("  ✓ Test data cleaned up")
    
    return True

def test_error_handling():
    """Test comprehensive error handling"""
    print("Testing Error Handling...")
    
    detector = get_emotion_detector()
    processor = get_speech_processor()
    chatbot = get_chatbot()
    
    # Test invalid inputs
    test_cases = [
        ("Invalid image data", lambda: detector.detect_face_emotion("invalid_data")),
        ("Invalid audio path", lambda: processor.process_audio_file("/invalid/path.wav")),
        ("Empty message", lambda: chatbot.get_response("")),
        ("None message", lambda: chatbot.get_response(None)),
    ]
    
    for test_name, test_func in test_cases:
        try:
            result = test_func()
            print(f"  ✓ {test_name} handled gracefully: {result}")
        except Exception as e:
            print(f"  ✗ {test_name} failed: {e}")
            return False
    
    return True

def test_performance():
    """Test performance and response times"""
    print("Testing Performance...")
    
    import time
    
    detector = get_emotion_detector()
    chatbot = get_chatbot()
    
    # Test emotion detection speed
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    start_time = time.time()
    for _ in range(5):
        detector.detect_face_emotion(dummy_image)
    avg_time = (time.time() - start_time) / 5
    print(f"  ✓ Emotion detection avg time: {avg_time:.3f}s")
    
    # Test chatbot response speed
    start_time = time.time()
    for _ in range(5):
        chatbot.get_response("Hello")
    avg_time = (time.time() - start_time) / 5
    print(f"  ✓ Chatbot response avg time: {avg_time:.3f}s")
    
    return True

def main():
    """Run all production tests"""
    print("Starting DailyNest Production Test Suite")
    print("=" * 60)
    
    tests = [
        ("Emotion Detection", test_emotion_detection),
        ("Speech Processing", test_speech_processing),
        ("Chatbot", test_chatbot),
        ("Database Integration", test_database_integration),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} Tests...")
        try:
            if test_func():
                passed += 1
                print(f"PASSED: {test_name}")
            else:
                failed += 1
                print(f"FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"FAILED: {test_name} with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Production Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! Your DailyNest AI is production-ready.")
    else:
        print("Some tests failed. Review the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
