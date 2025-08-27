#!/usr/bin/env python3
"""
Test script to check DailyNest endpoints and identify issues
"""

import requests
import json
import sys

def test_chat_endpoint():
    """Test the chat message endpoint"""
    try:
        response = requests.post(
            'http://127.0.0.1:8000/chat-message/',
            json={'message': 'Hello'},
            headers={'Content-Type': 'application/json'}
        )
        print(f"Chat endpoint status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return False

def test_emotion_endpoint():
    """Test the emotion detection endpoint"""
    try:
        # Test with minimal data
        response = requests.post(
            'http://127.0.0.1:8000/detect-emotion/',
            json={},
            headers={'Content-Type': 'application/json'}
        )
        print(f"Emotion endpoint status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Emotion endpoint error: {e}")
        return False

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get('http://127.0.0.1:8000/')
        print(f"Server health status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Server health error: {e}")
        return False

if __name__ == "__main__":
    print("Testing DailyNest endpoints...")
    print("=" * 50)
    
    # Test server health
    if not test_server_health():
        print("Server is not running. Please start with: python manage.py runserver")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Test chat endpoint
    test_chat_endpoint()
    
    print("\n" + "=" * 50)
    
    # Test emotion endpoint
    test_emotion_endpoint() 