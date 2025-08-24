import os
import time
import logging
from pathlib import Path
from threading import Lock
import json

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - emotion detection will use fallback methods")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available - using basic math operations")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - face detection will use fallback")

# Check TensorFlow availability without importing
try:
    import importlib.util
    tf_spec = importlib.util.find_spec("tensorflow")
    TENSORFLOW_AVAILABLE = tf_spec is not None
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - emotion model will use fallback")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - emotion model will use fallback")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Librosa not available - audio processing will use fallback")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Speech recognition not available")

# Check LangChain availability without importing
try:
    import importlib.util
    langchain_spec = importlib.util.find_spec("langchain")
    LANGCHAIN_AVAILABLE = langchain_spec is not None
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - chatbot will use fallback responses")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - chatbot will use fallback responses")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Requests not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe if available
if MEDIAPIPE_AVAILABLE:
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_face_detection = None
    mp_face_mesh = None
    mp_drawing = None

class EmotionDetector:
    def __init__(self):
        # Initialize MediaPipe components if available
        if MEDIAPIPE_AVAILABLE and mp_face_detection:
            self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        else:
            self.face_detection = None
            self.face_mesh = None
            logger.warning("MediaPipe not available - face detection disabled")
        
        # Initialize emotion model
        self.emotion_model = self._build_emotion_model()
        
        # Initialize speech recognizer if available
        if SPEECH_RECOGNITION_AVAILABLE:
            self.speech_recognizer = sr.Recognizer()
        else:
            self.speech_recognizer = None
            logger.warning("Speech recognition not available")
        
        self.emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.rate_limiter = RateLimiter(1.0)
        
    def _build_emotion_model(self):
        """Build a CNN model for emotion recognition"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - using fallback emotion detection")
            return None
        
        try:
            # Import TensorFlow only when actually needed
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
            
            # Try to load pre-trained face emotion models
            weights_paths = [
                os.path.join('models', 'face_emotion', 'fer.h5'),
                os.path.join('models', 'face_emotion', 'best_mobilenet_model.h5'),
                os.path.join('models', 'face_emotion', 'final_mobilenet_model.h5'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'face_emotion', 'fer.h5'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'face_emotion', 'best_mobilenet_model.h5'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'face_emotion', 'final_mobilenet_model.h5')
            ]
            
            model = None
            weights_loaded = False
            
            for weights_path in weights_paths:
                try:
                    if os.path.exists(weights_path):
                        # Try to load the full model first
                        try:
                            model = tf.keras.models.load_model(weights_path)
                            logger.info(f"Loaded complete emotion model from {weights_path}")
                            weights_loaded = True
                            break
                        except Exception as load_error:
                            logger.warning(f"Could not load complete model from {weights_path}: {str(load_error)}")
                            continue
                except Exception as e:
                    logger.warning(f"Could not access model file {weights_path}: {str(e)}")
                    continue
            
            if not weights_loaded:
                # Create default CNN model if no pre-trained model found
                logger.warning("No valid pre-trained models found, creating default CNN model")
                model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                    Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    Flatten(),
                    Dropout(0.5),
                    Dense(512, activation='relu'),
                    Dense(7, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            return model
        except Exception as e:
            logger.error(f"Error building emotion model: {str(e)}")
            return None

    def detect_face_emotion(self, image):
        """Detect emotion from facial expression"""
        if not self.rate_limiter:
            return "neutral"
            
        self.rate_limiter.wait()
        
        if image is None:
            return self._fallback_face_emotion()
        
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            return self._fallback_face_emotion()
        
        try:
            # Convert image to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            if not results.detections:
                return "no_face_detected"
            
            # Get the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            
            # Extract face region
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return "no_face_detected"
            
            # Use emotion model if available
            if self.emotion_model and TENSORFLOW_AVAILABLE:
                import tensorflow as tf
                
                # Preprocess face for emotion model
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                face_input = np.expand_dims(face_input, axis=-1)
                
                # Predict emotion
                predictions = self.emotion_model.predict(face_input, verbose=0)
                emotion_index = np.argmax(predictions[0])
                confidence = float(predictions[0][emotion_index])
                
                if confidence < 0.6:
                    return "uncertain"
                
                return self.emotion_labels[emotion_index]
            else:
                return self._fallback_face_emotion()
            
        except Exception as e:
            logger.error(f"Face emotion detection error: {str(e)}")
            return "neutral"
    
    def detect_voice_emotion(self, audio_path):
        """Detect emotion from voice/audio using wav2vec2 and advanced features"""
        if not self.rate_limiter:
            return "neutral"
            
        self.rate_limiter.wait()
        
        try:
            # Try wav2vec2 approach first
            return self._wav2vec2_emotion_detection(audio_path)
        except Exception as e:
            logger.warning(f"wav2vec2 emotion detection failed: {str(e)}")
            return self._librosa_emotion_detection(audio_path)
    
    def _wav2vec2_emotion_detection(self, audio_path):
        """Use wav2vec2 model for voice emotion detection"""
        try:
            # Import transformers only when needed
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            import torch
            import torchaudio
            
            # Load pre-trained wav2vec2 model
            model_name = "facebook/wav2vec2-base-960h"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2Model.from_pretrained(model_name)
            
            # Load and preprocess audio
            if LIBROSA_AVAILABLE:
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)
            else:
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio = resampler(waveform).squeeze().numpy()
                else:
                    audio = waveform.squeeze().numpy()
            
            # Process with wav2vec2
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                
            # Extract features from hidden states
            features = torch.mean(hidden_states, dim=1).squeeze().numpy()
            
            # Use the features for emotion classification
            return self._classify_emotion_from_wav2vec2_features(features, audio)
            
        except ImportError:
            logger.warning("Transformers/torch not available for wav2vec2")
            raise Exception("wav2vec2 dependencies not available")
        except Exception as e:
            logger.error(f"wav2vec2 processing error: {str(e)}")
            raise e
    
    def _classify_emotion_from_wav2vec2_features(self, wav2vec2_features, audio):
        """Classify emotion using wav2vec2 features combined with traditional audio features"""
        try:
            # Extract traditional audio features as well
            audio_features = self._extract_audio_features(audio)
            
            # Combine wav2vec2 features with traditional features for better accuracy
            energy = audio_features.get('energy', 0.001)
            spectral_centroid = audio_features.get('spectral_centroid', 1000)
            zero_crossing_rate = audio_features.get('zero_crossing_rate', 0.05)
            rms = audio_features.get('rms', 0.01)
            
            # Use wav2vec2 feature statistics
            wav2vec2_mean = np.mean(wav2vec2_features)
            wav2vec2_std = np.std(wav2vec2_features)
            wav2vec2_max = np.max(wav2vec2_features)
            
            # Enhanced classification using both feature sets
            if energy > 0.02 and spectral_centroid > 2500 and wav2vec2_std > 0.1:
                return "angry"
            elif energy > 0.015 and spectral_centroid > 2000 and wav2vec2_mean > 0:
                return "happy"
            elif energy < 0.005 and spectral_centroid < 1500 and wav2vec2_mean < -0.1:
                return "sad"
            elif energy > 0.01 and zero_crossing_rate > 0.12 and wav2vec2_max > 0.2:
                return "surprised"
            elif energy < 0.008 and rms < 0.015 and wav2vec2_std < 0.05:
                return "fearful"
            elif spectral_centroid < 1200 and wav2vec2_mean < -0.05:
                return "disgusted"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"wav2vec2 emotion classification error: {str(e)}")
            return "neutral"
    
    def _librosa_emotion_detection(self, audio_path):
        """Fallback emotion detection using librosa"""
        if not LIBROSA_AVAILABLE:
            return self._fallback_voice_emotion()
        
        try:
            import librosa
            # Load audio file
            audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
            
            # Extract comprehensive features
            features = self._extract_audio_features(audio)
            return self._classify_emotion_from_features(features)
                
        except Exception as e:
            logger.error(f"Librosa voice emotion detection error: {str(e)}")
            return self._fallback_voice_emotion()
    
    def _extract_audio_features(self, audio):
        """Extract comprehensive features from audio for emotion classification"""
        try:
            if LIBROSA_AVAILABLE:
                import librosa
                sr = 16000
                
                # Extract comprehensive audio features
                features = {}
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                features['mfcc_mean'] = np.mean(mfccs)
                features['mfcc_std'] = np.std(mfccs)
                
                # Spectral features
                features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
                features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
                
                # Energy and dynamics
                features['energy'] = np.sum(audio ** 2)
                features['rms'] = np.mean(librosa.feature.rms(y=audio))
                
                # Tempo and rhythm
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features['tempo'] = tempo
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                features['chroma_mean'] = np.mean(chroma)
                
                return features
            else:
                # Basic features without librosa
                return {
                    'energy': np.sum(audio ** 2),
                    'mean': np.mean(audio),
                    'std': np.std(audio),
                    'max': np.max(audio),
                    'min': np.min(audio),
                    'zero_crossing_rate': np.mean(np.diff(np.sign(audio)) != 0)
                }
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return {'energy': 0.001, 'mean': 0, 'std': 0.1}
    
    def _classify_emotion_from_features(self, features):
        """Advanced emotion classification based on extracted audio features"""
        try:
            # Enhanced rule-based classification using multiple features
            energy = features.get('energy', 0.001)
            spectral_centroid = features.get('spectral_centroid', 1000)
            zero_crossing_rate = features.get('zero_crossing_rate', 0.05)
            rms = features.get('rms', 0.01)
            tempo = features.get('tempo', 120)
            mfcc_mean = features.get('mfcc_mean', 0)
            
            # More sophisticated emotion classification with multiple criteria
            if energy > 0.02 and spectral_centroid > 2500 and zero_crossing_rate > 0.15 and tempo > 140:
                return "angry"
            elif energy > 0.015 and spectral_centroid > 2000 and rms > 0.02 and mfcc_mean > -5:
                return "happy"
            elif energy < 0.005 and spectral_centroid < 1500 and tempo < 100:
                return "sad"
            elif energy > 0.01 and zero_crossing_rate > 0.12 and spectral_centroid > 1800:
                return "surprised"
            elif energy < 0.008 and rms < 0.015 and mfcc_mean < -10:
                return "fearful"
            elif spectral_centroid < 1200 and zero_crossing_rate < 0.08:
                return "disgusted"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Emotion classification error: {str(e)}")
            return "neutral"
    
    def _fallback_face_emotion(self):
        """Fallback face emotion detection"""
        fallback_emotions = ["neutral", "happy", "calm"]
        return fallback_emotions[int(time.time()) % len(fallback_emotions)]
    
    def _fallback_voice_emotion(self):
        """Fallback voice emotion detection"""
        fallback_emotions = ["neutral", "calm", "content"]
        return fallback_emotions[int(time.time()) % len(fallback_emotions)]

class RateLimiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_time = 0
        self.lock = Lock()
    
    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()

class EmotionAwareChatbot:
    def __init__(self):
        """Initialize the emotion-aware chatbot with fallback support"""
        self.setup_chatbot()
        
    def setup_chatbot(self):
        """Setup chatbot with LangChain integration"""
        self.llm = None
        self.conversation = None
        
        if not LANGCHAIN_AVAILABLE:
            logger.info("Using fallback chatbot responses")
            return
            
        try:
            # Import LangChain only when actually needed
            from langchain_community.llms import Ollama
            from langchain.chains import ConversationChain
            from langchain.memory import ConversationBufferMemory
            from langchain.prompts import PromptTemplate
            
            # Try to initialize Ollama
            self.llm = Ollama(model="llama2", base_url="http://localhost:11434")
            
            # Create conversation memory and chain
            self.memory = ConversationBufferMemory(return_messages=True)
            
            # Create emotion-aware prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["history", "input", "face_emotion", "voice_emotion"],
                template="""You are an empathetic AI assistant designed to help people, especially those with autism.
                You should be:
                - Clear and direct in communication
                - Patient and understanding
                - Supportive and encouraging
                - Consistent in your responses
                
                Current detected emotions:
                Face emotion: {face_emotion}
                Voice emotion: {voice_emotion}
                
                Consider these emotions when crafting your response. Be supportive if negative emotions are detected,
                celebratory if positive emotions are detected, and always maintain a calm, helpful tone.
                
                Conversation history: {history}
                
                Human: {input}
                Assistant: """
            )
            
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt_template,
                verbose=False
            )
            
            logger.info("Successfully initialized LangChain chatbot with Ollama")
            
        except Exception as e:
            logger.warning(f"LangChain/Ollama setup failed: {str(e)}")
            self.llm = None
            self.conversation = None
    
    def get_response(self, message, face_emotion="neutral", voice_emotion="neutral"):
        """Get chatbot response with emotion awareness"""
        try:
            if self.conversation and self.llm:
                # Use LangChain for sophisticated responses
                response = self.conversation.predict(
                    input=message,
                    face_emotion=face_emotion,
                    voice_emotion=voice_emotion
                )
                return response.strip()
            else:
                # Use fallback responses
                return self._get_fallback_response(message, face_emotion, voice_emotion)
                
        except Exception as e:
            logger.error(f"Chatbot response error: {str(e)}")
            return self._get_fallback_response(message, face_emotion, voice_emotion)
    
    def _get_fallback_response(self, message, face_emotion="neutral", voice_emotion="neutral"):
        """Generate fallback responses based on emotions and message content"""
        message_lower = message.lower()
        
        # Emotion-aware responses
        if face_emotion == "sad" or voice_emotion == "sad":
            sad_responses = [
                "I understand you might be feeling down. Would you like to talk about what's bothering you?",
                "It's okay to feel sad sometimes. I'm here to listen and support you.",
                "I notice you seem sad. Remember that it's normal to have difficult emotions."
            ]
            return sad_responses[hash(message) % len(sad_responses)]
        
        elif face_emotion == "angry" or voice_emotion == "angry":
            angry_responses = [
                "I can sense some frustration. Let's take a deep breath and work through this together.",
                "It seems like something is bothering you. Would you like to share what's on your mind?",
                "I understand you might be feeling upset. I'm here to help in whatever way I can."
            ]
            return angry_responses[hash(message) % len(angry_responses)]
        
        elif face_emotion == "happy" or voice_emotion == "happy":
            happy_responses = [
                "I'm glad to see you're in a good mood! How can I help you today?",
                "You seem happy today! That's wonderful. What would you like to talk about?",
                "It's great to see you feeling positive! What's on your mind?"
            ]
            return happy_responses[hash(message) % len(happy_responses)]
        
        # Content-based responses
        if any(word in message_lower for word in ["help", "support", "need"]):
            return "I'm here to help you. Please let me know what specific support you need."
        elif any(word in message_lower for word in ["thank", "thanks"]):
            return "You're very welcome! I'm always happy to help."
        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm here to support you. How are you feeling today?"
        else:
            neutral_responses = [
                "I understand. Can you tell me more about that?",
                "That's interesting. How does that make you feel?",
                "I'm listening. Please continue sharing your thoughts.",
                "Thank you for sharing that with me. What would you like to explore further?"
            ]
            return neutral_responses[hash(message) % len(neutral_responses)]

# Global instances
_emotion_detector = None
_chatbot = None

def get_emotion_detector():
    """Get singleton emotion detector instance"""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    return _emotion_detector

def get_chatbot():
    """Get singleton chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = EmotionAwareChatbot()
    return _chatbot
