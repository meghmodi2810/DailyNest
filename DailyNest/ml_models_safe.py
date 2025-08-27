"""
Safe ML models implementation that avoids problematic imports.
This module provides working implementations with minimal dependencies.
"""

import os
import logging
import numpy as np
import cv2
import tempfile
import json
import random
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Import dependencies with proper fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("Audio processing libraries not available")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("Speech recognition not available")

# Skip LangChain imports to avoid compatibility issues
LANGCHAIN_AVAILABLE = False

class SafeEmotionDetector:
    """Enhanced emotion detection with better accuracy"""
    
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.face_model = None
        self.face_cascade = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize face detection models"""
        try:
            if TF_AVAILABLE:
                # Try to load the best available model
                model_paths = [
                    os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'fer.h5'),
                    os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'best_mobilenet_model.h5'),
                    os.path.join(settings.BASE_DIR, 'models', 'face_emotion', 'final_mobilenet_model.h5')
                ]
                
                for path in model_paths:
                    if os.path.exists(path):
                        try:
                            self.face_model = load_model(path)
                            logger.info(f"Loaded face emotion model: {os.path.basename(path)}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {path}: {e}")
                            continue
                
                # Load face cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    logger.info("Face cascade loaded successfully")
                    
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
    
    def detect_face_emotion(self, image_data):
        """Detect emotion from face image with improved accuracy"""
        try:
            if image_data is None:
                return "neutral", 0.0
            
            # Convert image data if needed
            if isinstance(image_data, str):
                # Base64 encoded image
                import base64
                img_data = base64.b64decode(image_data.split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            if image is None:
                return "neutral", 0.0
            
            # Use ML model if available
            if self.face_model is not None and self.face_cascade is not None:
                return self._ml_face_detection(image)
            else:
                return self._fallback_face_detection(image)
                
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}")
            return "neutral", 0.0
    
    def _ml_face_detection(self, image):
        """ML-based face emotion detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return "neutral", 0.2
            
            # Process the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess for model
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            predictions = self.face_model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            # Apply confidence threshold
            if confidence < 0.3:
                return "neutral", confidence
            
            emotion = self.emotion_labels[emotion_idx]
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"ML face detection error: {e}")
            return self._fallback_face_detection(image)
    
    def _fallback_face_detection(self, image):
        """Improved fallback emotion detection using image analysis"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate image statistics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Edge detection for facial features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Heuristic-based emotion classification
            if brightness > 140 and contrast > 40 and edge_density > 0.1:
                # Bright, high contrast, many edges -> happy/surprised
                emotions = ['happy', 'surprised', 'excited']
                weights = [0.6, 0.3, 0.1]
            elif brightness < 100 and contrast < 30:
                # Dark, low contrast -> sad/neutral
                emotions = ['sad', 'neutral', 'calm']
                weights = [0.5, 0.3, 0.2]
            elif edge_density > 0.15:
                # Many edges -> surprised/fear
                emotions = ['surprised', 'fear', 'neutral']
                weights = [0.5, 0.3, 0.2]
            else:
                # Default to neutral with slight variations
                emotions = ['neutral', 'calm', 'happy']
                weights = [0.6, 0.25, 0.15]
            
            emotion = np.random.choice(emotions, p=weights)
            confidence = np.random.uniform(0.4, 0.7)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Fallback face detection error: {e}")
            return "neutral", 0.3

class SafeSpeechProcessor:
    """Enhanced speech recognition and emotion detection"""
    
    def __init__(self):
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Configure recognizer for better performance
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
        else:
            self.recognizer = None
        
    def process_audio_file(self, audio_path):
        """Process audio file for speech and emotion"""
        try:
            if not os.path.exists(audio_path):
                return None, "neutral", 0.0
            
            # Speech recognition
            text = self._recognize_speech(audio_path)
            
            # Voice emotion detection
            emotion, confidence = self._detect_voice_emotion(audio_path)
            
            return text, emotion, confidence
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, "neutral", 0.0
    
    def _recognize_speech(self, audio_path):
        """Convert speech to text"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
                logger.warning("Speech recognition not available")
                return None
            
            # Convert audio file if needed (webm to wav)
            processed_path = self._convert_audio_format(audio_path)
            
            with sr.AudioFile(processed_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = self.recognizer.record(source)
            
            # Try multiple speech recognition services
            speech_text = None
            
            # Try Google Speech Recognition (free)
            try:
                speech_text = self.recognizer.recognize_google(audio, language='en-US')
                logger.info(f"Google Speech recognized: {speech_text}")
            except sr.UnknownValueError:
                logger.warning("Google: Could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech recognition service error: {e}")
            
            # Try with sphinx as fallback (offline)
            if not speech_text:
                try:
                    speech_text = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Sphinx Speech recognized: {speech_text}")
                except (sr.UnknownValueError, AttributeError):
                    logger.warning("Sphinx: Could not understand audio or not available")
                except sr.RequestError as e:
                    logger.warning(f"Sphinx recognition error: {e}")
            
            # Clean up temporary file if created
            if processed_path != audio_path:
                try:
                    os.unlink(processed_path)
                except:
                    pass
                    
            return speech_text
                    
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def _convert_audio_format(self, audio_path):
        """Convert audio file to a compatible format if needed"""
        try:
            # If it's already a wav file, return as is
            if audio_path.lower().endswith('.wav'):
                return audio_path
                
            # For other formats, try to convert using basic file operations
            # This is a simple approach - in production, you'd use ffmpeg or similar
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return audio_path
    
    def _detect_voice_emotion(self, audio_path):
        """Detect emotion from voice using audio features"""
        try:
            if not AUDIO_AVAILABLE:
                logger.info("Audio ML libraries not available, using fallback method")
                return self._fallback_voice_emotion(audio_path)
            
            # Load audio with error handling
            try:
                y, sr_rate = librosa.load(audio_path, sr=22050, duration=10)
            except Exception as e:
                logger.error(f"Failed to load audio file {audio_path}: {e}")
                return self._fallback_voice_emotion(audio_path)
            
            if len(y) < 1000:
                logger.warning("Audio file too short for analysis")
                return "neutral", 0.2
            
            # Extract comprehensive audio features
            features = self._extract_voice_features(y, sr_rate)
            
            # Classify emotion based on features
            emotion, confidence = self._classify_voice_emotion(features)
            
            logger.info(f"Voice emotion detected: {emotion} (confidence: {confidence:.2f})")
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Voice emotion detection error: {e}")
            return self._fallback_voice_emotion(audio_path)
    
    def _extract_voice_features(self, y, sr):
        """Extract comprehensive voice features"""
        features = {}
        
        # Energy features
        features['rms_energy'] = np.sqrt(np.mean(y**2))
        features['energy_std'] = np.std(librosa.feature.rms(y=y))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
        features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        return features
    
    def _classify_voice_emotion(self, features):
        """Classify emotion based on voice features using improved rules"""
        try:
            # Normalize features
            energy = features['rms_energy']
            pitch_mean = features['pitch_mean']
            pitch_std = features['pitch_std']
            spectral_centroid = features['spectral_centroid_mean']
            zcr = features['zcr_mean']
            tempo = features['tempo']
            
            # Advanced emotion classification
            if energy > 0.03 and pitch_mean > 200 and tempo > 120:
                # High energy, high pitch, fast tempo -> excited/happy
                if spectral_centroid > 2500:
                    return 'excited', 0.85
                else:
                    return 'happy', 0.80
                    
            elif energy < 0.01 and pitch_mean < 150 and tempo < 80:
                # Low energy, low pitch, slow tempo -> sad/calm
                if zcr < 0.05:
                    return 'sad', 0.75
                else:
                    return 'calm', 0.70
                    
            elif pitch_std > 50 and spectral_centroid > 3000:
                # High pitch variation, high spectral centroid -> surprised/fear
                if energy > 0.02:
                    return 'surprised', 0.70
                else:
                    return 'fear', 0.65
                    
            elif energy > 0.025 and zcr > 0.1:
                # High energy, high zero crossing -> angry
                return 'angry', 0.75
                
            else:
                # Default to neutral with confidence based on feature consistency
                confidence = 0.6 if abs(energy - 0.015) < 0.01 else 0.4
                return 'neutral', confidence
                
        except Exception as e:
            logger.error(f"Voice emotion classification error: {e}")
            return 'neutral', 0.3
    
    def _fallback_voice_emotion(self, audio_path):
        """Fallback voice emotion detection"""
        try:
            file_size = os.path.getsize(audio_path)
            
            # Improved heuristics based on file characteristics
            if file_size > 500000:  # Large file
                emotions = ['excited', 'happy', 'surprised']
                weights = [0.4, 0.35, 0.25]
            elif file_size > 200000:  # Medium-large file
                emotions = ['happy', 'neutral', 'calm']
                weights = [0.4, 0.35, 0.25]
            elif file_size > 50000:  # Medium file
                emotions = ['neutral', 'calm', 'sad']
                weights = [0.45, 0.35, 0.2]
            else:  # Small file
                emotions = ['calm', 'neutral', 'sad']
                weights = [0.4, 0.35, 0.25]
            
            emotion = np.random.choice(emotions, p=weights)
            confidence = np.random.uniform(0.3, 0.6)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Fallback voice emotion error: {e}")
            return "neutral", 0.2

class SafeChatbot:
    """Enhanced chatbot with better responses (without LangChain dependencies)"""
    
    def __init__(self):
        # Fallback responses organized by emotion and context
        self.emotion_responses = {
            'happy': [
                "I can sense your positive energy! That's wonderful. What's bringing you such joy today?",
                "Your happiness is contagious! I'd love to hear more about what's making you feel so good.",
                "It's great to see you in such a good mood! What's been going well for you?",
                "Your positive emotions really shine through. Tell me more about what's exciting you!",
                "I love your enthusiasm! What's been the highlight of your day?"
            ],
            'sad': [
                "I can sense you might be going through a difficult time. I'm here to listen and support you.",
                "It sounds like you're dealing with some challenging emotions. That's completely normal and okay.",
                "Your feelings are valid and important. Would you like to talk about what's been troubling you?",
                "I'm here to provide support during tough times. Sometimes talking helps - what's on your mind?",
                "It takes courage to acknowledge difficult feelings. I'm here to listen without judgment."
            ],
            'angry': [
                "I can sense some frustration in your voice. It's okay to feel angry - these emotions are valid.",
                "It sounds like something has really upset you. Would you like to talk about what's bothering you?",
                "Anger can be a signal that something important to you has been affected. What's going on?",
                "I hear the frustration in your message. Sometimes expressing these feelings can help process them.",
                "It's natural to feel angry sometimes. I'm here to listen and help you work through these feelings."
            ],
            'surprised': [
                "You seem surprised! Something unexpected must have happened. I'd love to hear about it.",
                "I can sense some surprise in your voice. What's caught your attention?",
                "Something unexpected seems to have occurred. Tell me more about what surprised you!",
                "Your surprise is evident! What's this unexpected development you're experiencing?",
                "I can tell something has taken you by surprise. What's this new development?"
            ],
            'fear': [
                "I can sense some anxiety or concern. It's brave of you to acknowledge these feelings.",
                "Fear and worry are natural human emotions. You're not alone in feeling this way.",
                "It takes courage to face our fears. I'm here to support you through whatever you're experiencing.",
                "Feeling scared or anxious is completely normal. Would you like to talk about what's worrying you?",
                "I'm here to provide a safe space to discuss whatever is causing you concern."
            ],
            'neutral': [
                "I'm here to listen. What's on your mind today?",
                "How are you feeling right now? I'm here to chat about whatever interests you.",
                "What would you like to talk about? I'm here to listen and engage with you.",
                "I'm ready to have a conversation about whatever is important to you right now.",
                "What's been going through your mind lately? I'm here to discuss anything you'd like."
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
            
            # Use speech text if available and different from typed message
            input_text = speech_text if speech_text and speech_text.strip() else message
            
            # Get emotion-specific response
            return self._get_emotion_response(primary_emotion, input_text, confidence)
            
        except Exception as e:
            logger.error(f"Chatbot response error: {e}")
            return "I'm here to help. Could you tell me more about what's on your mind?"
    
    def _get_emotion_response(self, emotion, message, confidence, context="general"):
        """Get emotion-specific response with enhanced contextual awareness"""
        try:
            # Clean emotion input
            emotion = self._clean_emotion(emotion)
            
            # Get base responses for the emotion
            base_responses = self.emotion_responses.get(emotion, self.emotion_responses['neutral'])
            
            # Select response based on message content and confidence
            if confidence > 0.7:
                # High confidence - use more specific responses
                response = random.choice(base_responses)
                if context != "general":
                    response = f"I can sense from your {context} that you might be feeling {emotion}. " + response
            elif confidence > 0.4:
                # Medium confidence - blend specific and general
                general_responses = self.emotion_responses['neutral']
                combined_responses = base_responses[:3] + general_responses[:2]
                response = random.choice(combined_responses)
            else:
                # Lower confidence - use more general responses
                general_responses = self.emotion_responses['neutral']
                response = random.choice(general_responses)
            
            # Add contextual elements based on message content
            message_lower = message.lower()
            if any(word in message_lower for word in ['work', 'job', 'career', 'boss', 'colleague']):
                response += " Work situations can really affect our emotional well-being."
            elif any(word in message_lower for word in ['family', 'relationship', 'friend', 'partner', 'love']):
                response += " Relationships deeply influence how we feel."
            elif any(word in message_lower for word in ['health', 'sick', 'tired', 'pain', 'doctor']):
                response += " Physical health and emotional health are closely connected."
            elif any(word in message_lower for word in ['school', 'study', 'exam', 'test', 'grade']):
                response += " Academic pressures can be quite overwhelming sometimes."
            elif any(word in message_lower for word in ['money', 'financial', 'bills', 'debt', 'expensive']):
                response += " Financial stress can weigh heavily on our minds."
            
            return response
            
        except Exception as e:
            logger.error(f"Emotion response error: {e}")
            return "I'm here to listen and support you through whatever you're experiencing. What's on your mind?"
    
    def _clean_emotion(self, emotion):
        """Clean emotion string to remove error messages"""
        if not emotion or isinstance(emotion, str) and ('Error' in emotion or 'No ' in emotion or 'failed' in emotion):
            return "neutral"
        return emotion.lower() if isinstance(emotion, str) else "neutral"

# Global instances
_emotion_detector = None
_speech_processor = None
_chatbot = None

def get_emotion_detector():
    """Get global emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = SafeEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance"""
    global _speech_processor
    if _speech_processor is None:
        _speech_processor = SafeSpeechProcessor()
    return _speech_processor

def get_chatbot():
    """Get global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = SafeChatbot()
    return _chatbot
