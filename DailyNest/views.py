from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import base64
import tempfile
import os
from .models import EmotionRecord, ChatMessage, UserPreference
from .ml_models_fallback import get_emotion_detector, get_speech_processor

# Import enhanced chatbot
from .chatbot_enhanced import get_enhanced_chatbot
import logging

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

def dashboard(request):
    """Dashboard page view - main landing page"""
    return render(request, 'dashboard.html')

def home(request):
    """Home page view"""
    return render(request, 'home.html')

def emotion(request):
    """Emotion detection page view"""
    # Get or create preferences for the current user (or default user if anonymous)
    if request.user.is_authenticated:
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
    else:
        # For anonymous users, try to get the first superuser or create default preferences
        from django.contrib.auth.models import User
        default_user = User.objects.filter(is_superuser=True).first()
        if default_user:
            preferences, created = UserPreference.objects.get_or_create(user=default_user)
        else:
            # Create a minimal context without preferences
            preferences = None
    
    context = {
        'preferences': preferences,
    }
    return render(request, 'emotion.html', context)

def chat(request):
    """Chat page view"""
    # Get or create preferences for the current user (or default user if anonymous)
    if request.user.is_authenticated:
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
    else:
        # For anonymous users, try to get the first superuser or create default preferences
        from django.contrib.auth.models import User
        default_user = User.objects.filter(is_superuser=True).first()
        if default_user:
            preferences, created = UserPreference.objects.get_or_create(user=default_user)
        else:
            # Create a minimal context without preferences
            preferences = None
    
    # Get recent chat messages
    recent_messages = ChatMessage.objects.order_by('-timestamp')[:20]
    context = {
        'preferences': preferences,
        'recent_messages': reversed(recent_messages),
    }
    return render(request, 'chat.html', context)

@csrf_exempt
def detect_emotion(request):
    """Enhanced emotion detection endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            face_emotion = None
            voice_emotion = None
            face_confidence = 0.0
            voice_confidence = 0.0
            speech_text = None
            
            detector = get_emotion_detector()
            speech_processor = get_speech_processor()

            # Process face image
            if 'image' in data:
                try:
                    face_emotion, face_confidence = detector.detect_face_emotion(data['image'])
                except Exception as e:
                    logger.error(f"Face processing error: {str(e)}")
                    face_emotion, face_confidence = "neutral", 0.0

            # Process audio with speech recognition
            if 'audio' in data:
                try:
                    # Decode base64 audio
                    audio_data = base64.b64decode(data['audio'].split(',')[1])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio.write(audio_data)
                        temp_audio_path = temp_audio.name
                    
                    # Process audio for speech and emotion
                    speech_text, voice_emotion, voice_confidence = speech_processor.process_audio_file(temp_audio_path)
                    
                    # Clean up
                    os.unlink(temp_audio_path)
                    
                except Exception as e:
                    logger.error(f"Audio processing error: {str(e)}")
                    voice_emotion, voice_confidence = "neutral", 0.0
                    speech_text = None

            # Save emotion record with confidence scores
            record = EmotionRecord.objects.create(
                face_emotion=face_emotion if face_emotion and not face_emotion.startswith('Error') else None,
                voice_emotion=voice_emotion if voice_emotion and not voice_emotion.startswith('Error') else None,
                face_confidence=face_confidence,
                voice_confidence=voice_confidence,
                notes=f"Face: {face_emotion or 'None'} ({face_confidence:.2f}), Voice: {voice_emotion or 'None'} ({voice_confidence:.2f}), Speech: {speech_text or 'None'}"
            )

            return JsonResponse({
                'success': True,
                'face_emotion': face_emotion or "No face detected",
                'voice_emotion': voice_emotion or "No voice detected",
                'speech_text': speech_text or "No speech detected",
                'face_confidence': face_confidence,
                'voice_confidence': voice_confidence,
                'record_id': record.id,
                'confidence': 'high' if (face_confidence > 0.6 or voice_confidence > 0.6) else 'low'
            })

        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def chat_message(request):
    """Enhanced chat message handler with LangChain integration"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({'error': 'Empty message'}, status=400)
            
            # Get the latest emotion record for context
            latest_emotion = EmotionRecord.objects.order_by('-timestamp').first()
            
            # Extract emotions
            face_emotion = latest_emotion.face_emotion if latest_emotion else 'neutral'
            voice_emotion = latest_emotion.voice_emotion if latest_emotion else 'neutral'
            
            # Clean emotion strings (remove error messages for chatbot)
            if face_emotion and ('Error' in face_emotion or 'No face' in face_emotion):
                face_emotion = 'neutral'
            if voice_emotion and ('Error' in voice_emotion or 'No voice' in voice_emotion):
                voice_emotion = 'neutral'
            
            # Save user message
            user_chat_message = ChatMessage.objects.create(
                sender='user',
                message=user_message,
                emotion_context=latest_emotion
            )
            
            # Get chatbot response using enhanced Ollama integration
            try:
                chatbot = get_enhanced_chatbot(model_name="gemma:2b")  # Using your installed gemma:2b model
                bot_response = chatbot.get_response(
                    message=user_message,
                    face_emotion=face_emotion,
                    voice_emotion=voice_emotion,
                    face_confidence=getattr(latest_emotion, 'face_confidence', 0.0),
                    voice_confidence=getattr(latest_emotion, 'voice_confidence', 0.0),
                    speech_text=getattr(latest_emotion, 'notes', '').split('Speech: ')[-1] if latest_emotion and 'Speech: ' in getattr(latest_emotion, 'notes', '') else None
                )
            except Exception as e:
                print(f"Chatbot error: {e}")
                bot_response = "I'm having trouble connecting to the local AI. Please make sure Ollama is running."
            
            # Save bot response
            bot_chat_message = ChatMessage.objects.create(
                sender='bot',
                message=bot_response,
                emotion_context=latest_emotion
            )
            
            return JsonResponse({
                'success': True,
                'response': bot_response,
                'emotion_context': {
                    'face_emotion': face_emotion,
                    'voice_emotion': voice_emotion
                },
                'message_id': bot_chat_message.id
            })
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': 'Sorry, I had trouble processing your message. Please try again.',
                'response': 'I apologize, but I encountered an issue. Could you please try rephrasing your message?'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def update_preferences(request):
    """Update user accessibility preferences"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get or create preferences for the current user (or default user if anonymous)
            if request.user.is_authenticated:
                preferences, created = UserPreference.objects.get_or_create(user=request.user)
            else:
                # For anonymous users, try to get the first superuser
                from django.contrib.auth.models import User
                default_user = User.objects.filter(is_superuser=True).first()
                if default_user:
                    preferences, created = UserPreference.objects.get_or_create(user=default_user)
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'No user available for preferences'
                    }, status=400)
            
            # Update preferences
            if 'theme' in data:
                preferences.theme = data['theme']
            if 'font_size' in data:
                preferences.font_size = data['font_size']
            if 'reduce_animations' in data:
                preferences.reduce_animations = data['reduce_animations']
            if 'high_contrast_mode' in data:
                preferences.high_contrast_mode = data['high_contrast_mode']
            if 'text_to_speech' in data:
                preferences.text_to_speech = data['text_to_speech']
                
            preferences.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Preferences updated successfully'
            })
            
        except Exception as e:
            logger.error(f"Preferences update error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def emotion_history(request):
    """Get recent emotion detection history"""
    return get_emotion_history(request)

@csrf_exempt
def get_emotion_history(request):
    """Get recent emotion detection history"""
    if request.method == 'GET':
        try:
            # Get last 10 emotion records
            records = EmotionRecord.objects.order_by('-timestamp')[:10]
            
            history = []
            for record in records:
                history.append({
                    'id': record.id,
                    'timestamp': record.timestamp.isoformat(),
                    'face_emotion': record.face_emotion,
                    'voice_emotion': record.voice_emotion,
                    'notes': record.notes
                })
            
            return JsonResponse({
                'success': True,
                'emotions': history
            })
            
        except Exception as e:
            logger.error(f"History retrieval error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def clear_chat(request):
    """Clear chat message history"""
    return clear_chat_history(request)

@csrf_exempt
def clear_chat_history(request):
    """Clear chat message history"""
    if request.method == 'POST':
        try:
            ChatMessage.objects.all().delete()
            return JsonResponse({
                'success': True,
                'message': 'Chat history cleared'
            })
        except Exception as e:
            logger.error(f"Clear history error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)