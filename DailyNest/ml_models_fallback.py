"""
Fallback ML models for DailyNest - Reliable emotion detection with fallbacks
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

class FallbackEmotionDetector:
    """Fallback emotion detection using OpenCV and basic ML"""
    
    def __init__(self):
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
        """Initialize face detection"""
        try:
            import cv2
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            logger.info("Fallback emotion detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Fallback emotion detector initialization error: {e}")
            raise
    
    def detect_face_emotion(self, image_data):
        """Detect face emotion using OpenCV-based analysis"""
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
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _analyze_face_emotion(self, image, face_rect):
        """Analyze emotion for a specific face region using OpenCV features"""
        try:
            import cv2
            x, y, w, h = face_rect
            
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Analyze facial features
            features = self._extract_facial_features(gray_face)
            
            # Classify emotion based on features
            emotion, confidence = self._classify_emotion(features)
            
            logger.info(f"Emotion detected: {emotion} (confidence: {confidence:.3f})")
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Face emotion analysis error: {e}")
            return "neutral", 0.0
    
    def _extract_facial_features(self, gray_face):
        """Extract facial features for emotion analysis"""
        try:
            import cv2
            
            # Resize for consistent analysis
            resized_face = cv2.resize(gray_face, (64, 64))
            
            # Calculate basic features
            features = {}
            
            # Brightness and contrast
            features['brightness'] = float(np.mean(resized_face))
            features['contrast'] = float(np.std(resized_face))
            
            # Mouth region analysis (lower third of face)
            mouth_region = resized_face[40:64, 16:48]
            features['mouth_brightness'] = float(np.mean(mouth_region))
            features['mouth_contrast'] = float(np.std(mouth_region))
            
            # Eye region analysis (upper third of face)
            eye_region = resized_face[8:32, 16:48]
            features['eye_brightness'] = float(np.mean(eye_region))
            features['eye_contrast'] = float(np.std(eye_region))
            
            # Edge density (smile detection)
            edges = cv2.Canny(resized_face, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def _classify_emotion(self, features):
        """Classify emotion based on extracted features"""
        try:
            if not features:
                return "neutral", 0.5
            
            # Extract features with proper type conversion
            mouth_brightness = float(features.get('mouth_brightness', 128))
            mouth_contrast = float(features.get('mouth_contrast', 30))
            edge_density = float(features.get('edge_density', 0.1))
            brightness = float(features.get('brightness', 128))
            contrast = float(features.get('contrast', 30))
            eye_brightness = float(features.get('eye_brightness', 128))
            eye_contrast = float(features.get('eye_contrast', 30))
            
            # Enhanced rule-based classification with better thresholds
            emotion_scores = {}
            
            # Happy: bright mouth, high edge density (smile), good contrast
            happy_score = 0
            if mouth_brightness > 135: happy_score += 0.3
            if edge_density > 0.12: happy_score += 0.3
            if contrast > 25: happy_score += 0.2
            if brightness > 120: happy_score += 0.2
            emotion_scores['happy'] = happy_score
            
            # Sad: low brightness, low contrast, dark features
            sad_score = 0
            if brightness < 105: sad_score += 0.4
            if contrast < 25: sad_score += 0.3
            if mouth_brightness < 120: sad_score += 0.3
            emotion_scores['sad'] = sad_score
            
            # Angry: high contrast, dark features, tense
            angry_score = 0
            if contrast > 45: angry_score += 0.4
            if brightness < 115: angry_score += 0.3
            if edge_density > 0.15: angry_score += 0.3
            emotion_scores['angry'] = angry_score
            
            # Surprised: high contrast, bright eyes, high edge density
            surprised_score = 0
            if contrast > 40: surprised_score += 0.3
            if eye_brightness > 145: surprised_score += 0.3
            if edge_density > 0.18: surprised_score += 0.4
            emotion_scores['surprised'] = surprised_score
            
            # Fear: low brightness, high edge density (tension), dark eyes
            fear_score = 0
            if brightness < 110: fear_score += 0.3
            if edge_density > 0.14: fear_score += 0.4
            if eye_brightness < 125: fear_score += 0.3
            emotion_scores['fear'] = fear_score
            
            # Disgust: medium brightness, low mouth contrast, moderate edge density
            disgust_score = 0
            if 100 < brightness < 130: disgust_score += 0.3
            if mouth_contrast < 22: disgust_score += 0.4
            if 0.08 < edge_density < 0.15: disgust_score += 0.3
            emotion_scores['disgust'] = disgust_score
            
            # Neutral: balanced features
            neutral_score = 0
            if 110 < brightness < 140: neutral_score += 0.3
            if 20 < contrast < 40: neutral_score += 0.3
            if 0.08 < edge_density < 0.12: neutral_score += 0.2
            if 120 < mouth_brightness < 140: neutral_score += 0.2
            emotion_scores['neutral'] = neutral_score
            
            # Find the emotion with highest score
            if emotion_scores:
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = min(emotion_scores[best_emotion], 0.9)
                
                # Ensure minimum confidence
                if confidence < 0.3:
                    return "neutral", 0.5
                
                return best_emotion, confidence
            else:
                return "neutral", 0.5
                
        except Exception as e:
            logger.error(f"Emotion classification error: {e}")
            return "neutral", 0.5

class FallbackSpeechProcessor:
    """Fallback speech processing using OpenAI Whisper base only"""
    
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
        """Process audio using Whisper for transcription and emotion analysis"""
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
        """Analyze emotion from speech text using enhanced keyword analysis"""
        try:
            if not speech_text:
                return "neutral", 0.4
            
            text_lower = speech_text.lower()
            
            # Enhanced emotion keywords with weights and context
            emotion_keywords = {
                'happy': {
                    'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'fantastic', 'awesome', 'brilliant', 'excellent', 'perfect', 'good', 'nice', 'beautiful', 'wonderful', 'delighted', 'pleased', 'thrilled'],
                    'weight': 1.0,
                    'context_words': ['feel', 'am', 'is', 'was', 'being', 'feeling']
                },
                'sad': {
                    'keywords': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'disappointed', 'miserable', 'unhappy', 'lonely', 'hopeless', 'sadness', 'grief', 'sorrow', 'melancholy', 'blue', 'gloomy'],
                    'weight': 1.0,
                    'context_words': ['feel', 'am', 'is', 'was', 'being', 'feeling']
                },
                'angry': {
                    'keywords': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate', 'irritated', 'rage', 'outraged', 'furious', 'livid', 'enraged', 'fuming', 'seething', 'hostile', 'aggressive'],
                    'weight': 1.0,
                    'context_words': ['feel', 'am', 'is', 'was', 'being', 'feeling']
                },
                'surprised': {
                    'keywords': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable', 'astonished', 'stunned', 'startled', 'bewildered', 'confused', 'what', 'how', 'why'],
                    'weight': 1.0,
                    'context_words': ['feel', 'am', 'is', 'was', 'being', 'feeling']
                },
                'fear': {
                    'keywords': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'frightened', 'panicked', 'fearful', 'terrified', 'horrified', 'dread', 'panic', 'stress', 'tension'],
                    'weight': 1.0,
                    'context_words': ['feel', 'am', 'is', 'was', 'being', 'feeling']
                }
            }
            
            # Calculate emotion scores with context
            emotion_scores = {}
            for emotion, config in emotion_keywords.items():
                score = 0
                
                # Check for direct emotion keywords
                for keyword in config['keywords']:
                    if keyword in text_lower:
                        score += config['weight']
                
                # Check for context (e.g., "I feel happy", "I am sad")
                for context_word in config['context_words']:
                    for keyword in config['keywords']:
                        if f"{context_word} {keyword}" in text_lower:
                            score += config['weight'] * 1.5  # Boost for context
                
                # Check for question patterns that might indicate surprise
                if emotion == 'surprised':
                    if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where']):
                        score += 0.5
                
                # Check for confidence-related words
                if 'confident' in text_lower or 'confidence' in text_lower:
                    if emotion == 'happy':
                        score += 1.0
                    elif emotion == 'fear':
                        score -= 0.5
                
                if score > 0:
                    emotion_scores[emotion] = score
            
            # Determine best emotion
            if emotion_scores:
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = min(emotion_scores[best_emotion] / 4.0, 0.9)  # Normalize confidence
                
                # Ensure minimum confidence
                if confidence < 0.3:
                    return "neutral", 0.5
                
                return best_emotion, confidence
            else:
                # Analyze text sentiment for neutral cases
                positive_words = ['good', 'nice', 'okay', 'fine', 'alright', 'well', 'better', 'great', 'confident', 'sure', 'certain']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'terrible', 'awful', 'unsure', 'uncertain', 'doubt']
                
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
                _emotion_detector = FallbackEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance with lazy loading"""
    global _speech_processor
    if _speech_processor is None:
        with _speech_processor_lock:
            if _speech_processor is None:
                _speech_processor = FallbackSpeechProcessor()
    return _speech_processor 