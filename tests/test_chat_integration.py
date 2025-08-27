#!/usr/bin/env python3
"""
Test chat integration
"""

import os
import sys
import django
import requests
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_chat_endpoint():
    """Test chat endpoint"""
    print("Testing chat endpoint...")
    
    try:
        # Test chat message
        response = requests.post(
            'http://127.0.0.1:8000/chat-message/',
            json={'message': 'Hello, how are you?'},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                response_text = data.get('response', '')
                if response_text and len(response_text) > 10:
                    print(f"✓ Chat endpoint passed: {response_text[:100]}...")
                    return True
                else:
                    print("✗ Chat endpoint returned empty or too short response")
                    return False
            else:
                print(f"✗ Chat endpoint failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Chat endpoint returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Chat endpoint connection failed - server may not be running")
        return False
    except Exception as e:
        print(f"✗ Chat endpoint error: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("Testing Ollama connection...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if 'gemma:2b' in models:
                print("✓ Ollama connection passed with gemma:2b available")
                return True
            else:
                print("✗ Ollama connection passed but gemma:2b not available")
                return False
        else:
            print(f"✗ Ollama connection failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ollama connection error: {e}")
        return False

if __name__ == "__main__":
    print("Chat Integration Tests")
    print("=" * 40)
    
    # Test Ollama connection
    ollama_ok = test_ollama_connection()
    print()
    
    # Test chat endpoint
    chat_ok = test_chat_endpoint()
    print()
    
    # Summary
    print("=" * 40)
    print("SUMMARY:")
    print(f"  Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"  Chat: {'✓' if chat_ok else '✗'}")
    
    success = ollama_ok and chat_ok
    if success:
        print("\n🎉 All chat tests passed!")
    else:
        print("\n⚠️  Some chat tests failed.")
    
    sys.exit(0 if success else 1) 