#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import os
import sys
import django
import requests

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_ollama_connection():
    """Test Ollama connection"""
    print("Testing Ollama connection...")
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            if 'gemma:2b' in models:
                print("✓ Ollama is running with gemma:2b available")
                return True
            else:
                print("✗ Ollama is running but gemma:2b not available")
                return False
        else:
            print(f"✗ Ollama returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False

def test_ml_models_import():
    """Test ML models import without loading"""
    print("Testing ML models import...")
    
    try:
        # Test import without loading models
        from DailyNest.ml_models_fixed import get_emotion_detector, get_speech_processor
        print("✓ ML models import successful")
        return True
    except Exception as e:
        print(f"✗ ML models import failed: {e}")
        return False

def test_server_health():
    """Test if server is running"""
    print("Testing server health...")
    
    try:
        response = requests.get('http://127.0.0.1:8000/', timeout=5)
        print(f"✓ Server is running (status: {response.status_code})")
        return True
    except Exception as e:
        print(f"✗ Server is not running: {e}")
        return False

def main():
    """Run simple tests"""
    print("DailyNest Simple Tests")
    print("=" * 40)
    
    # Test Ollama
    ollama_ok = test_ollama_connection()
    print()
    
    # Test ML models import
    ml_ok = test_ml_models_import()
    print()
    
    # Test server
    server_ok = test_server_health()
    print()
    
    # Summary
    print("=" * 40)
    print("SUMMARY:")
    print(f"  Ollama: {'✓' if ollama_ok else '✗'}")
    print(f"  ML Models: {'✓' if ml_ok else '✗'}")
    print(f"  Server: {'✓' if server_ok else '✗'}")
    
    if all([ollama_ok, ml_ok, server_ok]):
        print("\n🎉 Basic tests passed! DailyNest should be working.")
        print("\nNext steps:")
        print("1. Start Django server: python manage.py runserver")
        print("2. Test chat endpoint manually")
        print("3. Test emotion detection manually")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 