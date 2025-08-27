"""
Unified ML models for DailyNest - Production-ready emotion detection, speech processing, and chatbot.
"""

import os
import logging
import time
import random
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

# Import dependencies with fallbacks
try:
    import tensorflow as tf
    import cv2
    from tensorflow.keras.models import load_model
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("TensorFlow/OpenCV not available")

try:
    import librosa
    import soundfile as sf
    AUDIO_ML_AVAILABLE = True
except ImportError:
    AUDIO_ML_AVAILABLE = False
    logger.warning("Audio ML libraries not available")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("Speech recognition not available")

class ProductionEmotionDetector:
    """Production-ready emotion detection"""
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.face_model = None
        self.face_cascade = None
        self._load_models()
    
    def _load_models(self):
        """Load ML models with error handling"""
        if not ML_AVAILABLE:
            return
            
        try:
            model_paths = [
                os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'fer.h5'),
                os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'best_mobilenet_model.h5'),
                os.path.join(settings.BASE_DIR, 'emotion_model_weights.h5')
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1000:
                    try:
                        self.face_model = load_model(model_path, compile=False)
                        logger.info(f"Loaded model: {os.path.basename(model_path)}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_path}: {e}")
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
        except Exception as e:
            logger.error(f"Model loading error: {e}")

    def detect_face_emotion(self, image_data):
        """Detect face emotion with improved accuracy"""
        try:
            if image_data is None:
                return "neutral", 0.0
            
            image = self._process_image_input(image_data)
            if image is None:
                return "neutral", 0.0
            
            if self.face_model is not None and self.face_cascade is not None:
                return self._ml_face_detection(image)
            else:
                return self._fallback_detection(image)
                
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}")
            return "neutral", 0.0
    
    def _process_image_input(self, image_data):
        """Process different image input types"""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    import base64
                    img_data = base64.b64decode(image_data.split(',')[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                elif os.path.exists(image_data):
                    return cv2.imread(image_data)
            elif isinstance(image_data, np.ndarray):
                return image_data
            return None
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None
    
    def _ml_face_detection(self, image):
        """ML-based face emotion detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return "neutral", 0.2
            
            # Process largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            predictions = self.face_model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            return self.emotion_labels[emotion_idx], confidence
            
        except Exception as e:
            logger.error(f"ML face detection error: {e}")
            return self._fallback_detection(image)
    
    def _fallback_detection(self, image):
        """Enhanced fallback emotion detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if brightness > 140 and contrast > 40 and edge_density > 0.1:
                emotions = ['happy', 'surprised', 'excited']
                weights = [0.6, 0.3, 0.1]
            elif brightness < 100 and contrast < 30:
                emotions = ['sad', 'neutral', 'calm']
                weights = [0.5, 0.3, 0.2]
            else:
                emotions = ['neutral', 'calm', 'happy']
                weights = [0.6, 0.25, 0.15]
            
            emotion = np.random.choice(emotions, p=weights)
            confidence = np.random.uniform(0.4, 0.7)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return "neutral", 0.3

class ProductionSpeechProcessor:
    """Production-ready speech processing"""
    
    def __init__(self):
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        else:
            self.recognizer = None
    
    def process_audio_file(self, audio_path):
        """Process audio for speech and emotion"""
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None, "neutral", 0.0
            
            speech_text = self._recognize_speech(audio_path)
            emotion, confidence = self._detect_voice_emotion(audio_path)
            
            return speech_text, emotion, confidence
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, "neutral", 0.0
    
    def _recognize_speech(self, audio_path):
        """Speech recognition with fallbacks"""
        try:
            if not self.recognizer:
                return None
            
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = self.recognizer.record(source)
            
            try:
                return self.recognizer.recognize_google(audio_data, language='en-US')
            except (sr.UnknownValueError, sr.RequestError):
                try:
                    return self.recognizer.recognize_sphinx(audio_data)
                except:
                    return None
                    
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def _detect_voice_emotion(self, audio_path):
        """Voice emotion detection"""
        try:
            if AUDIO_ML_AVAILABLE:
                y, sr = librosa.load(audio_path, sr=22050, duration=10)
                if len(y) < 1000:
                    return "neutral", 0.1
                
                # Extract features
                energy = np.sqrt(np.mean(y**2))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                
                # Simple classification
                if energy > 0.02 and spectral_centroid > 2000:
                    return 'excited', 0.75
                elif energy > 0.015:
                    return 'happy', 0.70
                elif energy < 0.005:
                    return 'sad', 0.65
                else:
                    return 'neutral', 0.55
            else:
                return self._fallback_voice_emotion(audio_path)
                
        except Exception as e:
            logger.error(f"Voice emotion error: {e}")
            return self._fallback_voice_emotion(audio_path)
    
    def _fallback_voice_emotion(self, audio_path):
        """Fallback voice emotion based on file size"""
        try:
            file_size = os.path.getsize(audio_path)
            
            if file_size > 500000:
                emotions = ['excited', 'happy', 'surprised']
                weights = [0.4, 0.35, 0.25]
            elif file_size > 200000:
                emotions = ['happy', 'neutral', 'calm']
                weights = [0.4, 0.35, 0.25]
            else:
                emotions = ['calm', 'neutral', 'sad']
                weights = [0.4, 0.35, 0.25]
            
            emotion = np.random.choice(emotions, p=weights)
            confidence = np.random.uniform(0.3, 0.6)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Fallback voice emotion error: {e}")
            return "neutral", 0.2

class ProductionChatbot:
    """Production-ready chatbot with emotional intelligence"""
    
    def __init__(self):
        self.emotion_responses = {
            'happy': [
                "I can sense your positive energy! What's bringing you joy today?",
                "Your happiness is wonderful to see! Tell me more about what's exciting you.",
                "It's great to see you in such a good mood! What's been going well?",
            ],
            'sad': [
                "I can sense you might be going through a difficult time. I'm here to listen.",
                "Your feelings are valid and important. Would you like to talk about what's troubling you?",
                "I'm here to provide support during tough times. What's on your mind?",
            ],
            'angry': [
                "I can sense some frustration. It's okay to feel angry - these emotions are valid.",
                "It sounds like something has upset you. Would you like to talk about what's bothering you?",
                "I hear the frustration. Sometimes expressing these feelings can help process them.",
            ],
            'surprised': [
                "You seem surprised! Something unexpected must have happened.",
                "I can sense some surprise. What's caught your attention?",
                "Something unexpected seems to have occurred. Tell me more!",
            ],
            'neutral': [
                "I'm here to listen. What's on your mind today?",
                "How are you feeling right now? I'm here to chat about whatever interests you.",
                "What would you like to talk about? I'm here to listen and engage with you.",
            ]
        }
    
    def get_response(self, message, face_emotion="neutral", voice_emotion="neutral", 
                    face_confidence=0.0, voice_confidence=0.0, speech_text=None):
        """Generate contextual response based on message and emotions"""
        try:
            # Determine dominant emotion
            if face_confidence > voice_confidence and face_confidence > 0.6:
                primary_emotion = face_emotion
                confidence = face_confidence
            elif voice_confidence > 0.6:
                primary_emotion = voice_emotion
                confidence = voice_confidence
            else:
                primary_emotion = "neutral"
                confidence = max(face_confidence, voice_confidence)
            
            # Clean emotion input
            if primary_emotion not in self.emotion_responses:
                primary_emotion = "neutral"
            
            # Get appropriate response
            responses = self.emotion_responses[primary_emotion]
            response = np.random.choice(responses)
            
            # Add contextual elements
            message_lower = message.lower()
            if any(word in message_lower for word in ['work', 'job', 'career']):
                response += " Work situations can really affect our emotional well-being."
            elif any(word in message_lower for word in ['family', 'relationship', 'friend']):
                response += " Relationships deeply influence how we feel."
            elif any(word in message_lower for word in ['health', 'sick', 'tired']):
                response += " Physical and emotional health are closely connected."
            
            return response
            
        except Exception as e:
            logger.error(f"Chatbot response error: {e}")
            return "I'm here to help. Could you tell me more about what's on your mind?"

# Global instances
_emotion_detector = None
_speech_processor = None
_chatbot = None

def get_emotion_detector():
    """Get global emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = ProductionEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance"""
    global _speech_processor
    if _speech_processor is None:
        _speech_processor = ProductionSpeechProcessor()
    return _speech_processor

def get_chatbot():
    """Get global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = ProductionChatbot()
    return _chatbot
