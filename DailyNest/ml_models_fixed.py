"""
Fixed ML models for DailyNest - Lazy loading with OpenAI Whisper base
"""

import os
import logging
import numpy as np
from django.conf import settings
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

# Global instances with lazy loading
_emotion_detector = None
_speech_processor = None
_emotion_detector_lock = threading.Lock()
_speech_processor_lock = threading.Lock()

# Thread pool for model loading
_executor = ThreadPoolExecutor(max_workers=2)

class LazyEmotionDetector:
    """Lazy-loading emotion detector using MediaPipe and FER-2013 model"""
    
    def __init__(self):
        self.face_mesh = None
        self.face_detection = None
        self.fer_model = None
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
        """Initialize MediaPipe components and FER model"""
        try:
            # Import MediaPipe
            import cv2
            import mediapipe as mp
            
            mp_face_mesh = mp.solutions.face_mesh
            mp_face_detection = mp.solutions.face_detection
            
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            
            # Load FER-2013 model
            self._load_fer_model()
            
            logger.info("Emotion detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Emotion detector initialization error: {e}")
            # Fallback to basic OpenCV
            self._initialize_opencv_fallback()
    
    def _load_fer_model(self):
        """Load FER-2013 emotion recognition model"""
        try:
            import tensorflow as tf
            
            # Try to load the FER model
            model_path = os.path.join(settings.MODELS_DIR, 'face_emotion', 'fer.h5')
            if os.path.exists(model_path):
                self.fer_model = tf.keras.models.load_model(model_path)
                logger.info("FER-2013 model loaded successfully")
            else:
                logger.warning("FER-2013 model not found, using fallback")
                
        except Exception as e:
            logger.error(f"FER model loading error: {e}")
    
    def _initialize_opencv_fallback(self):
        """Initialize OpenCV fallback"""
        try:
            import cv2
            self.cv2_available = True
        except ImportError:
            self.cv2_available = False
            logger.warning("OpenCV not available for fallback")
    
    def detect_face_emotion(self, image_data):
        """Detect face emotion using FER-2013 model or MediaPipe landmarks"""
        self._ensure_initialized()
        
        try:
            if image_data is None:
                return "neutral", 0.0
            
            image = self._process_image_input(image_data)
            if image is None:
                return "neutral", 0.0
            
            # Try FER-2013 model first
            if self.fer_model:
                return self._fer_detection(image)
            # Fallback to MediaPipe
            elif self.face_mesh and self.face_detection:
                return self._mediapipe_detection(image)
            # Final fallback to OpenCV
            else:
                return self._opencv_fallback(image)
                
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}")
            return "neutral", 0.0
    
    def _fer_detection(self, image):
        """Detect emotion using FER-2013 model"""
        try:
            import cv2
            
            # Convert to grayscale and resize to 48x48 (FER-2013 format)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            
            # Normalize to [0, 1]
            normalized = resized.astype('float32') / 255.0
            
            # Add batch and channel dimensions
            input_tensor = np.expand_dims(normalized, axis=(0, -1))  # (1, 48, 48, 1)
            
            # Predict
            predictions = self.fer_model.predict(input_tensor, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            emotion = self.emotion_labels[emotion_idx]
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"FER detection error: {e}")
            return self._mediapipe_detection(image)
    
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
    
    def _mediapipe_detection(self, image):
        """MediaPipe-based emotion detection using facial landmarks"""
        try:
            import cv2
            import mediapipe as mp
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face
            face_results = self.face_detection.process(rgb_image)
            if not face_results.detections:
                return "neutral", 0.2
            
            # Get facial landmarks
            mesh_results = self.face_mesh.process(rgb_image)
            if not mesh_results.multi_face_landmarks:
                return "neutral", 0.2
            
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Extract key facial features
            emotion, confidence = self._analyze_landmarks(landmarks, image.shape)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return self._opencv_fallback(image)
    
    def _analyze_landmarks(self, landmarks, image_shape):
        """Analyze facial landmarks for emotion detection"""
        try:
            h, w = image_shape[:2]
            
            # Key landmark indices for emotion analysis
            mouth_left = landmarks.landmark[61]
            mouth_right = landmarks.landmark[291]
            mouth_center = landmarks.landmark[13]
            
            left_eyebrow = landmarks.landmark[70]
            right_eyebrow = landmarks.landmark[107]
            
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            
            # Calculate relative positions
            mouth_curve = (mouth_left.y + mouth_right.y) / 2 - mouth_center.y
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
            eye_openness = abs(left_eye.y - right_eye.y)
            
            # Simple emotion classification based on facial geometry
            if mouth_curve < -0.01:  # Mouth curves up
                if eyebrow_height < 0.4:  # Raised eyebrows
                    return 'surprised', 0.75
                else:
                    return 'happy', 0.80
            elif mouth_curve > 0.01:  # Mouth curves down
                if eyebrow_height > 0.45:  # Lowered eyebrows
                    return 'angry', 0.70
                else:
                    return 'sad', 0.75
            elif eyebrow_height < 0.35:  # Very raised eyebrows
                return 'surprised', 0.65
            elif eyebrow_height > 0.5:  # Very lowered eyebrows
                return 'angry', 0.60
            else:
                return 'neutral', 0.60
                
        except Exception as e:
            logger.error(f"Landmark analysis error: {e}")
            return 'neutral', 0.3
    
    def _opencv_fallback(self, image):
        """OpenCV fallback emotion detection"""
        try:
            import cv2
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return "neutral", 0.1
            
            # Analyze largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            face_roi = gray[y:y+h, x:x+w]
            
            # Simple feature analysis
            brightness = np.mean(face_roi)
            contrast = np.std(face_roi)
            
            # Detect smile using mouth region
            mouth_roi = face_roi[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
            mouth_brightness = np.mean(mouth_roi)
            
            if mouth_brightness > brightness * 1.1 and contrast > 30:
                return 'happy', 0.65
            elif brightness < 100 and contrast < 25:
                return 'sad', 0.60
            elif contrast > 50:
                return 'surprised', 0.55
            else:
                return 'neutral', 0.50
                
        except Exception as e:
            logger.error(f"OpenCV fallback error: {e}")
            return "neutral", 0.2

class LazySpeechProcessor:
    """Lazy-loading speech processor using OpenAI Whisper base"""
    
    def __init__(self):
        self.whisper_model = None
        self.emotion_classifier = None
        self._initialized = False
        self._initialization_lock = threading.Lock()
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    def _ensure_initialized(self):
        """Ensure models are loaded (lazy loading)"""
        if not self._initialized:
            with self._initialization_lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
    
    def _initialize(self):
        """Initialize Whisper model and emotion classifier"""
        try:
            # Import OpenAI Whisper
            import whisper
            
            # Load Whisper base model
            self.whisper_model = whisper.load_model("base")
            logger.info("OpenAI Whisper base model loaded successfully")
            
            # Initialize emotion classifier
            self._initialize_emotion_classifier()
            
        except Exception as e:
            logger.error(f"Speech processor initialization error: {e}")
    
    def _initialize_emotion_classifier(self):
        """Initialize emotion classifier using wav2vec2"""
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            import torch
            import torch.nn as nn
            
            # Load wav2vec2 base model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            
            # Create simple emotion classifier head
            self.emotion_classifier = nn.Sequential(
                nn.Linear(768, 256),  # wav2vec2 base output size is 768
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.emotion_labels))
            )
            
            logger.info("Voice emotion classifier initialized")
            
        except Exception as e:
            logger.error(f"Voice emotion classifier initialization error: {e}")
    
    def process_audio_file(self, audio_path):
        """Process audio using Whisper for speech and emotion"""
        self._ensure_initialized()
        
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None, "neutral", 0.0
            
            # Transcribe speech using Whisper
            speech_text = self._whisper_transcribe(audio_path)
            
            # Analyze emotion from speech and audio features
            emotion, confidence = self._analyze_speech_emotion(speech_text, audio_path)
            
            return speech_text, emotion, confidence
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, "neutral", 0.0
    
    def _whisper_transcribe(self, audio_path):
        """Transcribe audio using OpenAI Whisper base"""
        try:
            if not self.whisper_model:
                return None
            
            # Ensure audio is in the right format for Whisper
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip() if result["text"] else None
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    def _analyze_speech_emotion(self, speech_text, audio_path):
        """Analyze emotion from speech text and audio features"""
        try:
            # Text-based emotion analysis
            if speech_text:
                text_emotion, text_confidence = self._text_emotion_analysis(speech_text)
            else:
                text_emotion, text_confidence = "neutral", 0.0
            
            # Audio feature analysis using wav2vec2
            audio_emotion, audio_confidence = self._wav2vec2_emotion_analysis(audio_path)
            
            # Combine results (prefer text if available)
            if text_confidence > audio_confidence and text_confidence > 0.3:
                return text_emotion, text_confidence
            else:
                return audio_emotion, audio_confidence
                
        except Exception as e:
            logger.error(f"Speech emotion analysis error: {e}")
            return "neutral", 0.0
    
    def _wav2vec2_emotion_analysis(self, audio_path):
        """Analyze emotion using wav2vec2 features"""
        try:
            if not hasattr(self, 'feature_extractor') or not hasattr(self, 'wav2vec2_model'):
                return "neutral", 0.2
            
            import librosa
            import torch
            
            # Load and resample audio to 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare input for wav2vec2
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Extract features
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)  # Average over time
            
            # Simple emotion classification based on feature statistics
            # This is a simplified approach - in production you'd use a trained classifier
            feature_mean = features.mean().item()
            feature_std = features.std().item()
            
            # Simple heuristic-based emotion detection
            if feature_std > 0.5:
                return 'excited', 0.6
            elif feature_mean > 0.1:
                return 'happy', 0.5
            elif feature_mean < -0.1:
                return 'sad', 0.5
            else:
                return 'neutral', 0.4
                
        except Exception as e:
            logger.error(f"Wav2vec2 emotion analysis error: {e}")
            return "neutral", 0.2
    
    def _text_emotion_analysis(self, text):
        """Simple text-based emotion analysis"""
        try:
            text_lower = text.lower()
            
            # Emotion keywords
            emotion_keywords = {
                'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'fantastic'],
                'sad': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'disappointed'],
                'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate', 'irritated'],
                'surprised': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable'],
                'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified']
            }
            
            scores = {}
            for emotion, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    scores[emotion] = score / len(keywords)
            
            if scores:
                best_emotion = max(scores, key=scores.get)
                confidence = min(scores[best_emotion] * 2, 0.9)  # Cap at 0.9
                return best_emotion, confidence
            else:
                return 'neutral', 0.4
                
        except Exception as e:
            logger.error(f"Text emotion analysis error: {e}")
            return 'neutral', 0.2

# Global getter functions with lazy loading
def get_emotion_detector():
    """Get global emotion detector instance with lazy loading"""
    global _emotion_detector
    if _emotion_detector is None:
        with _emotion_detector_lock:
            if _emotion_detector is None:
                _emotion_detector = LazyEmotionDetector()
    return _emotion_detector

def get_speech_processor():
    """Get global speech processor instance with lazy loading"""
    global _speech_processor
    if _speech_processor is None:
        with _speech_processor_lock:
            if _speech_processor is None:
                _speech_processor = LazySpeechProcessor()
    return _speech_processor 