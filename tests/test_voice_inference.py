#!/usr/bin/env python3
"""
Test voice emotion inference
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

from DailyNest.ml_models_fixed import get_speech_processor

def create_test_audio():
    """Create a simple test audio file"""
    # Create a temporary WAV file with random audio data
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Create a simple sine wave
        sample_rate = 16000
        duration = 1.0  # 1 second
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

def test_voice_emotion_detection():
    """Test voice emotion detection"""
    print("Testing voice emotion detection...")
    
    try:
        # Get speech processor
        processor = get_speech_processor()
        
        # Create test audio
        test_audio_path = create_test_audio()
        
        try:
            # Test processing
            speech_text, emotion, confidence = processor.process_audio_file(test_audio_path)
            
            # Validate results
            expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
            
            if emotion in expected_emotions:
                print(f"✓ Voice emotion detection passed: {emotion} (confidence: {confidence:.2f})")
                if speech_text:
                    print(f"  Speech text: {speech_text[:50]}...")
                return True
            else:
                print(f"✗ Voice emotion detection failed: unexpected emotion '{emotion}'")
                return False
                
        finally:
            # Clean up
            if os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
            
    except Exception as e:
        print(f"✗ Voice emotion detection error: {e}")
        return False

if __name__ == "__main__":
    success = test_voice_emotion_detection()
    sys.exit(0 if success else 1) 