#!/usr/bin/env python3
"""
Test endpoint functionality
"""

import requests
import json

def test_chat_endpoint():
    """Test chat endpoint"""
    try:
        print("Testing chat endpoint...")
        response = requests.post(
            'http://127.0.0.1:8000/chat-message/',
            json={'message': 'Hello'},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_emotion_endpoint():
    """Test emotion endpoint"""
    try:
        print("\nTesting emotion endpoint...")
        response = requests.post(
            'http://127.0.0.1:8000/detect-emotion/',
            json={},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("DailyNest Endpoint Tests")
    print("=" * 40)
    
    chat_ok = test_chat_endpoint()
    emotion_ok = test_emotion_endpoint()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  Chat: {'✓' if chat_ok else '✗'}")
    print(f"  Emotion: {'✓' if emotion_ok else '✗'}")
    
    if chat_ok and emotion_ok:
        print("\n🎉 All endpoints working!")
    else:
        print("\n⚠️  Some endpoints failed.") 