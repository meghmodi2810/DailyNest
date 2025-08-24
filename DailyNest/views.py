from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import base64
import tempfile
import os
from .models import EmotionRecord, ChatMessage, UserPreference
from .utils import get_emotion_detector, get_chatbot
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
    preferences, created = UserPreference.objects.get_or_create(id=1)
    context = {
        'preferences': preferences,
    }
    return render(request, 'emotion.html', context)

def chat(request):
    """Chat page view"""
    preferences, created = UserPreference.objects.get_or_create(id=1)
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
            detector = get_emotion_detector()

            # Process face image
            if 'image' in data:
                try:
                    if NUMPY_AVAILABLE and CV2_AVAILABLE:
                        # Decode base64 image
                        img_data = base64.b64decode(data['image'].split(',')[1])
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            face_emotion = detector.detect_face_emotion(img)
                        else:
                            face_emotion = "Error: Invalid image"
                    else:
                        face_emotion = detector.detect_face_emotion(None)
                        
                except Exception as e:
                    logger.error(f"Face processing error: {str(e)}")
                    face_emotion = "neutral"

            # Process audio
            if 'audio' in data:
                try:
                    # Decode base64 audio
                    audio_data = base64.b64decode(data['audio'].split(',')[1])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio.write(audio_data)
                        temp_audio_path = temp_audio.name
                    
                    # Process audio
                    voice_emotion = detector.detect_voice_emotion(temp_audio_path)
                    
                    # Clean up
                    os.unlink(temp_audio_path)
                    
                except Exception as e:
                    logger.error(f"Audio processing error: {str(e)}")
                    voice_emotion = f"Error: {str(e)}"

            # Save emotion record
            record = EmotionRecord.objects.create(
                face_emotion=face_emotion,
                voice_emotion=voice_emotion,
                notes=f"Face: {face_emotion}, Voice: {voice_emotion}"
            )

            return JsonResponse({
                'success': True,
                'face_emotion': face_emotion or "No face detected",
                'voice_emotion': voice_emotion or "No voice detected",
                'record_id': record.id,
                'confidence': 'high' if face_emotion and 'Error' not in face_emotion else 'low'
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
            
            # Get chatbot response
            chatbot = get_chatbot()
            bot_response = chatbot.get_response(
                message=user_message,
                face_emotion=face_emotion,
                voice_emotion=voice_emotion
            )
            
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
            preferences, created = UserPreference.objects.get_or_create(id=1)
            
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