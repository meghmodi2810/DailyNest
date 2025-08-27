"""
Improved ML models for DailyNest - Fixed emotion detection with proper model integration
"""

import os
import logging
import numpy as np
import tempfile
from django.conf import settings
import threading

logger = logging.getLogger(__name__)

# Global instances with lazy loading
_emotion_detector = None
_speech_processor = None
_emotion_detector_lock = threading.Lock()
_speech_processor_lock = threading.Lock()

class ImprovedEmotionDetector:
    """Improved emotion detection using the specified MobileNet model"""
    
    def __init__(self):
        self.face_model = None
        self.face_cascade = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self._initialized = False
        self._initialization_lock = threading.Lock()
    
    def _ensure_initialized(self):
        """Ensure models are loaded (lazy loading)"""
        if not self._initialized:
            with self._initialization_lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
    
    def _initialize(self):
        """Initialize face detection and emotion model"""
        try:
            import cv2
            import tensorflow as tf
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Load the specified MobileNet emotion model
            model_path = os.path.join(settings.MODELS_DIR, 'face_emotion', 'best_mobilenet_model.h5')
            if os.path.exists(model_path):
                try:
                    self.face_model = tf.keras.models.load_model(model_path, compile=False)
                    logger.info("MobileNet emotion model loaded successfully")
                except Exception as model_error:
                    logger.error(f"Error loading model: {model_error}")
                    # Try alternative model
                    alt_model_path = os.path.join(settings.MODELS_DIR, 'face_emotion', 'final_mobilenet_model.h5')
                    if os.path.exists(alt_model_path):
                        try:
                            self.face_model = tf.keras.models.load_model(alt_model_path, compile=False)
                            logger.info("Alternative MobileNet model loaded successfully")
                        except Exception as alt_error:
                            logger.error(f"Error loading alternative model: {alt_error}")
                            raise
                    else:
                        raise FileNotFoundError(f"No usable model found")
            else:
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info("Face emotion detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Face emotion detector initialization error: {e}")
            raise
    
    def detect_face_emotion(self, image_data):
        """Detect face emotion using the MobileNet model"""
        self._ensure_initialized()
        
        try:
            if image_data is None:
                return "neutral", 0.0
            
            image = self._process_image_input(image_data)
            if image is None:
                return "neutral", 0.0
            
            # Detect faces
            faces = self._detect_faces(image)
            if not faces:
                return "neutral", 0.1
            
            # Process the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            emotion, confidence = self._analyze_face_emotion(image, largest_face)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}")
            return "neutral", 0.0
    
    def _process_image_input(self, image_data):
        """Process different image input types"""
        try:
            import cv2
            
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
    
    def _detect_faces(self, image):
        """Detect faces in the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _analyze_face_emotion(self, image, face_rect):
        """Analyze emotion for a specific face region"""
        try:
            x, y, w, h = face_rect
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Resize to model input size (160x160 for this MobileNet model)
            resized_face = cv2.resize(face_roi, (160, 160))
            
            # Convert to RGB (MobileNet expects RGB)
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized_face = rgb_face.astype('float32') / 255.0
            
            # Add batch dimension
            input_tensor = np.expand_dims(normalized_face, axis=0)
            
            # Predict emotion
            predictions = self.face_model.predict(input_tensor, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            emotion = self.emotion_labels[emotion_idx]
            
            logger.info(f"Emotion detected: {emotion} (confidence: {confidence:.3f})")
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Face emotion analysis error: {e}")
            return "neutral", 0.0

class ImprovedSpeechProcessor:
    """Improved speech processing using OpenAI Whisper base only"""
    
    def __init__(self):
        self.whisper_model = None
        self._initialized = False
        self._initialization_lock = threading.Lock()
    
    def _ensure_initialized(self):
        """Ensure Whisper model is loaded (lazy loading)"""
        if not self._initialized:
            with self._initialization_lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
    
    def _initialize(self):
        """Initialize OpenAI Whisper base model"""
        try:
            import whisper
            
            # Load Whisper base model
            self.whisper_model = whisper.load_model("base")
            logger.info("OpenAI Whisper base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Whisper initialization error: {e}")
            raise
    
    def process_audio_file(self, audio_path):
        """Process audio using Whisper for transcription and basic emotion analysis"""
        self._ensure_initialized()
        
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None, "neutral", 0.0
            
            # Transcribe speech using Whisper
            speech_text = self._whisper_transcribe(audio_path)
            
            # Analyze emotion from speech text
            emotion, confidence = self._analyze_speech_emotion(speech_text)
            
            return speech_text, emotion, confidence
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, "neutral", 0.0
    
    def _whisper_transcribe(self, audio_path):
        """Transcribe audio using OpenAI Whisper base"""
        try:
            if not self.whisper_model:
                return None
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_path)
            text = result["text"].strip() if result["text"] else None
            
            logger.info(f"Whisper transcription: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    def _analyze_speech_emotion(self, speech_text):
        """Analyze emotion from speech text using keyword analysis"""
        try:
            if not speech_text:
                return "neutral", 0.4
            
            text_lower = speech_text.lower()
            
            # Enhanced emotion keywords with weights
            emotion_keywords = {
                'happy': {
                    'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'fantastic', 'awesome', 'brilliant'],
                    'weight': 1.0
                },
                'sad': {
                    'keywords': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'disappointed', 'miserable', 'unhappy'],
                    'weight': 1.0
                },
                'angry': {
                    'keywords': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate', 'irritated', 'rage', 'outraged'],
                    'weight': 1.0
                },
                'surprised': {
                    'keywords': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable', 'astonished'],
                    'weight': 1.0
                },
                'fear': {
                    'keywords': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'frightened', 'panicked'],
                    'weight': 1.0
                }
            }
            
            # Calculate emotion scores
            emotion_scores = {}
            for emotion, config in emotion_keywords.items():
                score = 0
                for keyword in config['keywords']:
                    if keyword in text_lower:
                        score += config['weight']
                if score > 0:
                    emotion_scores[emotion] = score
            
            # Determine best emotion
            if emotion_scores:
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = min(emotion_scores[best_emotion] / 3.0, 0.9)  # Normalize confidence
                return best_emotion, confidence
            else:
                # Analyze text sentiment for neutral cases
                positive_words = ['good', 'nice', 'okay', 'fine', 'alright', 'well']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst']
                
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    return 'happy', 0.6
                elif negative_count > positive_count:
                    return 'sad', 0.6
                else:
                    return 'neutral', 0.5
                
        except Exception as e:
            logger.error(f"Speech emotion analysis error: {e}")
            return 'neutral', 0.3

# Global getter functions with lazy loading
def get_emotion_detector():
    """Get global emotion detector instance with lazy loading"""
    global _emotion_detector
    if _emotion_detector is None:
        with _emotion_detector_lock:
            if _emotion_detector is None:
                _emotion_detector = ImprovedEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance with lazy loading"""
    global _speech_processor
    if _speech_processor is None:
        with _speech_processor_lock:
            if _speech_processor is None:
                _speech_processor = ImprovedSpeechProcessor()
    return _speech_processor
