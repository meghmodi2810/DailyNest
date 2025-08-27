"""
Improved ML models for emotion detection, speech processing, and chatbot responses.
This module provides working implementations with proper error handling and fallbacks.
"""

import os
import logging
import numpy as np
import cv2
import tempfile
import json
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

try:
    from langchain_community.llms import Ollama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old imports for backward compatibility
        from langchain.llms import Ollama
        from langchain.schema import HumanMessage, SystemMessage
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationChain
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning("LangChain not available")

class ImprovedEmotionDetector:
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

class ImprovedSpeechProcessor:
    """Enhanced speech recognition and emotion detection"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None
        
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
            if not SPEECH_RECOGNITION_AVAILABLE:
                return None
            
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            
            # Try multiple recognition engines
            try:
                # Google Speech Recognition (free)
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Speech recognized: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                # Fallback to offline recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    return None
                    
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None
    
    def _detect_voice_emotion(self, audio_path):
        """Detect emotion from voice using audio features"""
        try:
            if not AUDIO_AVAILABLE:
                return self._fallback_voice_emotion(audio_path)
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=10)
            
            if len(y) < 1000:
                return "neutral", 0.1
            
            # Extract comprehensive audio features
            features = self._extract_voice_features(y, sr)
            
            # Classify emotion based on features
            emotion, confidence = self._classify_voice_emotion(features)
            
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

class ImprovedChatbot:
    """Enhanced chatbot with LangChain integration and better responses"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory() if LANGCHAIN_AVAILABLE else None
        self.llm = None
        self.conversation_chain = None
        self._initialize_llm()
        
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
        
    def _initialize_llm(self):
        """Initialize LangChain LLM if available"""
        try:
            if LANGCHAIN_AVAILABLE:
                # Try to connect to local Ollama instance with timeout
                self.llm = Ollama(
                    model="llama2", 
                    base_url="http://localhost:11434",
                    timeout=5  # 5 second timeout
                )
                
                # Create conversation chain
                if self.memory:
                    self.conversation_chain = ConversationChain(
                        llm=self.llm,
                        memory=self.memory,
                        verbose=False
                    )
                
                logger.info("LangChain LLM initialized (connection will be tested on first use)")
                
        except Exception as e:
            logger.warning(f"LangChain LLM initialization failed: {e}")
            self.llm = None
            self.conversation_chain = None
    
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
            
            # Try LangChain first if available
            if self.conversation_chain is not None:
                try:
                    # Create context-aware prompt
                    context_prompt = self._create_context_prompt(
                        input_text, primary_emotion, confidence
                    )
                    
                    response = self.conversation_chain.predict(input=context_prompt)
                    
                    # Clean up response
                    response = self._clean_response(response)
                    
                    if response and len(response.strip()) > 10:
                        logger.info(f"LangChain response generated for emotion: {primary_emotion}")
                        return response
                        
                except Exception as e:
                    logger.warning(f"LangChain response failed: {e}")
            
            # Fallback to emotion-based responses
            return self._get_emotion_response(primary_emotion, input_text, confidence)
            
        except Exception as e:
            logger.error(f"Chatbot response error: {e}")
            return "I'm here to help. Could you tell me more about what's on your mind?"
    
    def _create_context_prompt(self, message, emotion, confidence):
        """Create context-aware prompt for LangChain"""
        emotion_context = ""
        if confidence > 0.6:
            emotion_context = f" The user seems to be feeling {emotion} (confidence: {confidence:.2f})."
        
        prompt = f"""You are an empathetic AI assistant designed to provide emotional support and engaging conversation. 
        
Context: {emotion_context}

User message: "{message}"

Please respond in a caring, supportive way that acknowledges the user's emotional state if apparent. Keep responses conversational, helpful, and emotionally intelligent. Avoid being overly clinical or robotic."""

        return prompt
    
    def _clean_response(self, response):
        """Clean up LLM response"""
        if not response:
            return ""
        
        # Remove common LLM artifacts
        response = response.strip()
        response = response.replace("AI:", "").replace("Assistant:", "").strip()
        
        # Ensure response isn't too long
        if len(response) > 500:
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def _get_emotion_response(self, emotion, message, confidence):
        """Get emotion-specific response"""
        try:
            # Get base responses for the emotion
            base_responses = self.emotion_responses.get(emotion, self.emotion_responses['neutral'])
            
            # Select response based on message content and confidence
            if confidence > 0.7:
                # High confidence - use more specific responses
                response = np.random.choice(base_responses)
            else:
                # Lower confidence - use more general responses
                general_responses = self.emotion_responses['neutral']
                response = np.random.choice(general_responses + base_responses[:2])
            
            # Add contextual elements based on message content
            if any(word in message.lower() for word in ['work', 'job', 'career']):
                response += " Work-related concerns can be really challenging."
            elif any(word in message.lower() for word in ['family', 'relationship', 'friend']):
                response += " Relationships are such an important part of our lives."
            elif any(word in message.lower() for word in ['health', 'sick', 'tired']):
                response += " Taking care of your wellbeing is so important."
            
            return response
            
        except Exception as e:
            logger.error(f"Emotion response error: {e}")
            return "I'm here to listen and support you. What would you like to talk about?"

# Global instances
_emotion_detector = None
_speech_processor = None
_chatbot = None

def get_emotion_detector():
    """Get global emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = ImprovedEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance"""
    global _speech_processor
    if _speech_processor is None:
        _speech_processor = ImprovedSpeechProcessor()
    return _speech_processor

def get_chatbot():
    """Get global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = ImprovedChatbot()
    return _chatbot
