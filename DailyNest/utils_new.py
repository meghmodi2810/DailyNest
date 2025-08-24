import os
import logging
import time
import tempfile
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available")

try:
    from langchain_community.llms import FakeListLLM
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available")

class RateLimiter:
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_time = 0

    def wait(self):
        elapsed = time.time() - self.last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_time = time.time()

class EmotionDetector:
    def __init__(self):
        self.emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.emotion_model = None
        self.face_cascade = None
        self.mp_face_detection = None
        self.mp_drawing = None
        self.rate_limiter = RateLimiter(min_interval=0.5)
        
        # Initialize models
        self._initialize_face_detection()
        self._build_emotion_model()

    def _initialize_face_detection(self):
        """Initialize face detection"""
        try:
            if CV2_AVAILABLE:
                # Try to load OpenCV cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    logger.info("OpenCV face detection initialized")
            
            if MEDIAPIPE_AVAILABLE:
                # Initialize MediaPipe
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5)
                self.mp_drawing = mp.solutions.drawing_utils
                logger.info("MediaPipe face detection initialized")
                
        except Exception as e:
            logger.error(f"Face detection initialization error: {str(e)}")

    def _build_emotion_model(self):
        """Build or load emotion detection model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - using fallback emotion detection")
            return
        
        try:
            # Try to load pre-trained models from the models directory
            model_paths = [
                'models/face_emotion/best_mobilenet_model.h5',
                'models/face_emotion/fer.h5',
                'models/face_emotion/final_mobilenet_model.h5',
                'emotion_model_weights.h5'
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.emotion_model = tf.keras.models.load_model(model_path)
                        logger.info(f"Loaded emotion model from {model_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {str(e)}")
                        continue
            
            # If no pre-trained model found, create a simple CNN
            logger.info("Creating default CNN emotion model")
            self.emotion_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.emotion_labels), activation='softmax')
            ])
            
            self.emotion_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            logger.error(f"Model building error: {str(e)}")
            self.emotion_model = None

    def detect_face_emotion(self, image):
        """Detect emotion from face in image"""
        if image is None:
            return "neutral"
        
        self.rate_limiter.wait()
        
        try:
            # Convert image if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = []
            
            # Try MediaPipe first
            if self.mp_face_detection and MEDIAPIPE_AVAILABLE:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_detection.process(rgb_image)
                    
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            h, w = image.shape[:2]
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            faces.append((x, y, width, height))
                except Exception as e:
                    logger.warning(f"MediaPipe detection failed: {str(e)}")
            
            # Fallback to OpenCV
            if not faces and self.face_cascade and CV2_AVAILABLE:
                try:
                    detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    faces = [(x, y, w, h) for (x, y, w, h) in detected_faces]
                except Exception as e:
                    logger.warning(f"OpenCV detection failed: {str(e)}")
            
            if faces and self.emotion_model:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                face_input = np.expand_dims(face_input, axis=-1)
                
                # Predict emotion
                predictions = self.emotion_model.predict(face_input, verbose=0)
                emotion_index = np.argmax(predictions[0])
                confidence = float(predictions[0][emotion_index])
                
                predicted_emotion = self.emotion_labels[emotion_index]
                logger.info(f"Face emotion: {predicted_emotion} (confidence: {confidence:.3f})")
                
                # Only return uncertain if confidence is extremely low
                if confidence < 0.15:
                    return "uncertain"
                
                return predicted_emotion
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Face emotion detection error: {str(e)}")
            return "neutral"

    def detect_voice_emotion(self, audio_path):
        """Detect emotion from voice/audio"""
        if not audio_path or not os.path.exists(audio_path):
            return "neutral"
        
        self.rate_limiter.wait()
        
        try:
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:  # Less than 1KB
                logger.warning("Audio file too small")
                return "neutral"
            
            if LIBROSA_AVAILABLE:
                import librosa
                
                # Load audio
                audio, sr = librosa.load(audio_path, sr=16000, duration=5.0)
                
                if len(audio) == 0:
                    return "neutral"
                
                # Extract features
                rms = librosa.feature.rms(y=audio)[0]
                energy = float(rms.mean())
                
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                zcr_mean = float(zcr.mean())
                
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                spectral_centroid = float(spectral_centroids.mean())
                
                try:
                    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                    tempo = float(tempo)
                except:
                    tempo = 120.0
                
                logger.info(f"Audio features - Energy: {energy:.4f}, ZCR: {zcr_mean:.4f}, Centroid: {spectral_centroid:.1f}, Tempo: {tempo:.1f}")
                
                # Classification
                if energy > 0.03 and zcr_mean > 0.15:
                    return "angry"
                elif energy > 0.02 and tempo > 130:
                    return "happy"
                elif energy < 0.005 and tempo < 100:
                    return "sad"
                elif spectral_centroid > 2500:
                    return "surprised"
                elif energy < 0.01 and zcr_mean < 0.05:
                    return "calm"
                elif energy > 0.015 and spectral_centroid > 2000:
                    return "excited"
                else:
                    return "neutral"
            else:
                # Basic file analysis without librosa
                if file_size > 50000:
                    return "excited"
                elif file_size < 10000:
                    return "calm"
                else:
                    return "neutral"
                    
        except Exception as e:
            logger.error(f"Voice emotion detection error: {str(e)}")
            return "neutral"

class EmotionAwareChatbot:
    def __init__(self):
        self.llm = None
        self.conversation = None
        self.setup_chatbot()
        
    def setup_chatbot(self):
        """Setup chatbot with LangChain"""
        if not LANGCHAIN_AVAILABLE:
            logger.info("LangChain not available - using fallback responses")
            return
            
        try:
            # Use reliable FakeListLLM for consistent responses
            responses = [
                "I understand how you're feeling. Can you tell me more about that?",
                "That sounds challenging. I'm here to support you through this.",
                "Thank you for sharing that with me. How does that make you feel?",
                "I can sense that this is important to you. Let's explore this together.",
                "Your feelings are valid. What would help you feel better right now?",
                "I'm listening. Please continue sharing your thoughts with me.",
                "That's a great insight. How can we build on that?",
                "I appreciate you being open with me. What's on your mind next?",
                "It's okay to feel this way. Many people experience similar emotions.",
                "You're doing great by expressing yourself. Keep going.",
                "I notice you might be feeling different emotions. That's completely normal.",
                "Your emotional experience is unique and valid. Tell me more.",
                "Sometimes talking about our feelings helps us understand them better.",
                "I'm here to listen without judgment. What would you like to share?",
                "Every emotion serves a purpose. What do you think this feeling is telling you?"
            ]
            
            self.llm = FakeListLLM(responses=responses)
            
            # Create conversation chain
            self.memory = ConversationBufferMemory(return_messages=True)
            
            self.prompt_template = PromptTemplate(
                input_variables=["input", "face_emotion", "voice_emotion"],
                template="""You are an empathetic AI assistant for people with autism.
                Be clear, direct, patient, and supportive.
                
                Current emotions - Face: {face_emotion}, Voice: {voice_emotion}
                
                Human: {input}
                Assistant: """
            )
            
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt_template,
                verbose=False
            )
            
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Chatbot setup error: {str(e)}")
            self.llm = None
            self.conversation = None

    def get_response(self, message, face_emotion="neutral", voice_emotion="neutral"):
        """Get chatbot response with emotion awareness"""
        try:
            if self.conversation and self.llm:
                response = self.conversation.predict(
                    input=message,
                    face_emotion=face_emotion,
                    voice_emotion=voice_emotion
                )
                return response.strip()
            else:
                # Fallback responses
                fallback_responses = [
                    "I understand. Can you tell me more about how you're feeling?",
                    "That's interesting. What's on your mind right now?",
                    "I'm here to listen. Please continue.",
                    "Thank you for sharing that with me.",
                    "How does that make you feel?",
                    "I appreciate you being open about your thoughts.",
                    "What would help you feel better right now?",
                    "Your feelings are completely valid."
                ]
                import random
                return random.choice(fallback_responses)
                
        except Exception as e:
            logger.error(f"Chatbot response error: {str(e)}")
            return "I'm here to help. Could you please try rephrasing your message?"

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
