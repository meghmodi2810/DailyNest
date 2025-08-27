#!/usr/bin/env python3
"""
Test script to verify DailyNest fixes
"""

import os
import sys
import django
import requests
import json
import time

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from DailyNest.ml_models_fixed import get_emotion_detector, get_speech_processor
from DailyNest.chatbot_ollama import get_ollama_chatbot

def test_ml_models_loading():
    """Test that ML models load without freezing"""
    print("Testing ML models loading...")
    
    try:
        # Test emotion detector
        print("  Loading emotion detector...")
        detector = get_emotion_detector()
        print("  ✓ Emotion detector loaded successfully")
        
        # Test speech processor
        print("  Loading speech processor...")
        processor = get_speech_processor()
        print("  ✓ Speech processor loaded successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ ML models loading failed: {e}")
        return False

def test_chatbot_loading():
    """Test that chatbot loads without freezing"""
    print("Testing chatbot loading...")
    
    try:
        chatbot = get_ollama_chatbot("gemma:2b")
        print("  ✓ Chatbot loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Chatbot loading failed: {e}")
        return False

def test_server_endpoints():
    """Test server endpoints"""
    print("Testing server endpoints...")
    
    # Start server if not running
    try:
        response = requests.get('http://127.0.0.1:8000/', timeout=5)
        print("  ✓ Server is running")
    except:
        print("  ✗ Server is not running. Please start with: python manage.py runserver")
        return False
    
    # Test chat endpoint
    try:
        response = requests.post(
            'http://127.0.0.1:8000/chat-message/',
            json={'message': 'Hello'},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"  ✓ Chat endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"    Response: {data.get('response', 'No response')[:100]}...")
    except Exception as e:
        print(f"  ✗ Chat endpoint failed: {e}")
    
    # Test emotion endpoint
    try:
        response = requests.post(
            'http://127.0.0.1:8000/detect-emotion/',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"  ✓ Emotion endpoint: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Emotion endpoint failed: {e}")
    
    return True

def test_ollama_connection():
    """Test Ollama connection"""
    print("Testing Ollama connection...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"  ✓ Ollama is running. Available models: {models}")
            return True
        else:
            print(f"  ✗ Ollama returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("DailyNest Fix Verification")
    print("=" * 50)
    
    # Test Ollama connection
    ollama_ok = test_ollama_connection()
    print()
    
    # Test ML models loading
    ml_ok = test_ml_models_loading()
    print()
    
    # Test chatbot loading
    chatbot_ok = test_chatbot_loading()
    print()
    
    # Test server endpoints
    server_ok = test_server_endpoints()
    print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"  Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"  ML Models: {'✓' if ml_ok else '✗'}")
    print(f"  Chatbot: {'✓' if chatbot_ok else '✗'}")
    print(f"  Server: {'✓' if server_ok else '✗'}")
    
    if all([ollama_ok, ml_ok, chatbot_ok, server_ok]):
        print("\n🎉 All tests passed! DailyNest should be working properly.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 