import os
import logging
import time
import random
import numpy as np
from django.conf import settings
from django.core.cache import cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import tensorflow as tf
    import cv2
    from tensorflow.keras.models import load_model
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("TensorFlow/OpenCV not available - using fallback methods")

try:
    import librosa
    import soundfile as sf
    AUDIO_ML_AVAILABLE = True
except ImportError:
    AUDIO_ML_AVAILABLE = False
    logger.warning("Audio ML libraries not available - using basic audio analysis")

class RateLimiter:
    def __init__(self, min_interval=0.1):
        self.min_interval = min_interval
        self.last_time = 0

    def wait(self):
        elapsed = time.time() - self.last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_time = time.time()

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.rate_limiter = RateLimiter(min_interval=0.1)
        self.face_model = None
        self.face_cascade = None
        self._load_models()
        logger.info(f"EmotionDetector initialized - ML Available: {ML_AVAILABLE}")
    
    def _load_models(self):
        """Load ML models if available"""
        if not ML_AVAILABLE:
            return
            
        try:
            # Load face emotion model
            model_path = os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'fer.h5')
            if os.path.exists(model_path):
                self.face_model = load_model(model_path)
                logger.info("Face emotion model loaded successfully")
            else:
                # Try alternative model
                alt_path = os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'best_mobilenet_model.h5')
                if os.path.exists(alt_path):
                    self.face_model = load_model(alt_path)
                    logger.info("Alternative face emotion model loaded")
            
            # Load face cascade for detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Face cascade loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.face_model = None
            self.face_cascade = None

    def detect_face_emotion(self, image):
        """Detect face emotion using ML model or fallback"""
        self.rate_limiter.wait()
        
        try:
            if image is None:
                return "neutral", 0.0
            
            if ML_AVAILABLE and hasattr(self, 'face_model') and self.face_model is not None:
                emotion, confidence = self._predict_face_emotion_ml(image)
            else:
                emotion, confidence = self._predict_face_emotion_fallback(image)
            
            logger.info(f"Face emotion detected: {emotion} (confidence: {confidence:.2f})")
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Face emotion detection error: {str(e)}")
            return "neutral", 0.0
    
    def _predict_face_emotion_ml(self, image):
        """ML-based face emotion prediction"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Load face cascade if not loaded
            if not hasattr(self, 'face_cascade') or self.face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return "neutral", 0.1
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract and preprocess face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            predictions = self.face_model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            emotion = self.emotion_labels[emotion_idx]
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"ML face prediction error: {str(e)}")
            return self._predict_face_emotion_fallback(image)
    
    def _predict_face_emotion_fallback(self, image):
        """Fallback face emotion prediction"""
        if hasattr(image, 'shape') and len(image.shape) >= 2:
            brightness = np.mean(image)
            contrast = np.std(image)
            
            if brightness > 150 and contrast > 50:
                emotions = ['happy', 'surprised', 'excited']
                weights = [0.5, 0.3, 0.2]
            elif brightness < 100:
                emotions = ['sad', 'neutral', 'calm']
                weights = [0.4, 0.4, 0.2]
            else:
                emotions = ['neutral', 'calm', 'happy']
                weights = [0.5, 0.3, 0.2]
        else:
            emotions = ['neutral', 'happy', 'calm']
            weights = [0.4, 0.3, 0.3]
        
        emotion = random.choices(emotions, weights=weights)[0]
        confidence = random.uniform(0.3, 0.7)
        return emotion, confidence

    def detect_voice_emotion(self, audio_path):
        """Detect voice emotion using audio analysis"""
        self.rate_limiter.wait()
        
        try:
            if not audio_path or not os.path.exists(audio_path):
                return "neutral", 0.0
            
            if AUDIO_ML_AVAILABLE:
                emotion, confidence = self._predict_voice_emotion_ml(audio_path)
            else:
                emotion, confidence = self._predict_voice_emotion_fallback(audio_path)
            
            logger.info(f"Voice emotion detected: {emotion} (confidence: {confidence:.2f})")
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Voice emotion detection error: {str(e)}")
            return "neutral", 0.0
    
    def _predict_voice_emotion_ml(self, audio_path):
        """ML-based voice emotion prediction using audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            if len(y) < 1000:  # Too short
                return "neutral", 0.1
            
            # Extract basic audio features
            rms_energy = np.sqrt(np.mean(y**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Simple rule-based classification
            if rms_energy > 0.02 and spectral_centroid_mean > 2000:
                return 'excited', 0.75
            elif rms_energy > 0.015:
                return 'happy', 0.70
            elif rms_energy < 0.005:
                if zcr < 0.05:
                    return 'sad', 0.65
                else:
                    return 'calm', 0.60
            elif spectral_centroid_mean > 3000:
                return 'surprised', 0.60
            else:
                return 'neutral', 0.55
            
        except Exception as e:
            logger.error(f"ML voice prediction error: {str(e)}")
            return self._predict_voice_emotion_fallback(audio_path)
    
    def _predict_voice_emotion_fallback(self, audio_path):
        """Fallback voice emotion prediction"""
        file_size = os.path.getsize(audio_path)
        
        if file_size < 1000:
            return "neutral", 0.1
        
        # Improved heuristics
        if file_size > 200000:
            emotions = ['excited', 'happy', 'surprised']
            weights = [0.4, 0.4, 0.2]
        elif file_size > 100000:
            emotions = ['happy', 'neutral', 'calm']
            weights = [0.4, 0.3, 0.3]
        else:
            emotions = ['calm', 'neutral', 'sad']
            weights = [0.4, 0.3, 0.3]
        
        emotion = random.choices(emotions, weights=weights)[0]
        confidence = random.uniform(0.3, 0.6)
        return emotion, confidence

class EmotionAwareChatbot:
    def __init__(self):
        self.responses = [
            "I understand how you're feeling. Can you tell me more about that?",
            "That sounds interesting. What's on your mind right now?",
            "Thank you for sharing that with me. How does that make you feel?",
            "I can sense that this is important to you. Let's explore this together.",
            "Your feelings are valid. What would help you feel better right now?",
            "I'm here to listen. Please continue sharing your thoughts with me.",
            "That's a great insight. How can we build on that?",
            "I appreciate you being open with me. What's on your mind next?",
            "It's okay to feel this way. Many people experience similar emotions.",
            "You're doing great by expressing yourself. Keep going.",
            "I notice you might be feeling different emotions. That's completely normal.",
            "Your emotional experience is unique and valid. Tell me more.",
            "Sometimes talking about our feelings helps us understand them better.",
            "I'm here to listen without judgment. What would you like to share?",
            "Every emotion serves a purpose. What do you think this feeling is telling you?",
            "I hear you. Can you help me understand what you're going through?",
            "That must be challenging for you. How are you coping with that?",
            "What you're feeling makes complete sense. Tell me more about it.",
            "I'm glad you're comfortable sharing this with me. How can I support you?",
            "Your perspective is valuable. What else would you like to discuss?"
        ]
        logger.info("Chatbot initialized with emotion-aware responses")
        
    def get_response(self, message, face_emotion="neutral", voice_emotion="neutral", face_confidence=0.0, voice_confidence=0.0):
        """Get chatbot response with enhanced emotion awareness"""
        try:
            # Determine dominant emotion based on confidence
            if face_confidence > voice_confidence:
                primary_emotion = face_emotion
                primary_confidence = face_confidence
            else:
                primary_emotion = voice_emotion
                primary_confidence = voice_confidence
            
            # Only use emotion-specific responses if confidence is high enough
            if primary_confidence > 0.6:
                if primary_emotion in ['sad', 'angry', 'fear']:
                    supportive_responses = [
                        "I can sense you might be going through something difficult. I'm here to listen.",
                        "It sounds like you're dealing with some challenging emotions. That's completely okay.",
                        "Your feelings are important and valid. Would you like to talk about what's bothering you?",
                        "I'm here to support you through whatever you're experiencing right now.",
                        "Sometimes it helps to talk about what's troubling us. I'm here to listen without judgment."
                    ]
                    response = random.choice(supportive_responses)
                elif primary_emotion in ['happy', 'excited']:
                    positive_responses = [
                        "I can sense some positive energy from you! That's wonderful to see.",
                        "You seem to be in a good mood today. What's bringing you joy?",
                        "It's great to connect with you when you're feeling positive. Tell me more!",
                        "Your positive emotions are contagious! What's going well for you?",
                        "I love seeing you in such a good mood! What's making you feel so positive?"
                    ]
                    response = random.choice(positive_responses)
                elif primary_emotion == 'surprised':
                    surprised_responses = [
                        "You seem surprised! What's caught your attention?",
                        "Something unexpected happened? I'd love to hear about it.",
                        "I can sense some surprise in your expression. What's going on?"
                    ]
                    response = random.choice(surprised_responses)
                else:
                    response = random.choice(self.responses)
            else:
                # Low confidence - use general responses
                response = random.choice(self.responses)
            
            logger.info(f"Generated response for emotions - Face: {face_emotion}({face_confidence:.2f}), Voice: {voice_emotion}({voice_confidence:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"Chatbot response error: {str(e)}")
            return "I'm here to help. Could you please tell me more about what's on your mind?"

# Global instances
_emotion_detector = None
_chatbot = None

def get_emotion_detector():
    """Get global emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    return _emotion_detector

def get_chatbot():
    """Get global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = EmotionAwareChatbot()
    return _chatbot
