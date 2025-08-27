"""
Modern LangChain + ChatOllama Chatbot for DailyNest
Emotion-aware conversational AI with local LLM integration
"""

import os
import logging
from typing import Optional, Dict, Any, List
from django.conf import settings

logger = logging.getLogger(__name__)

# Import with fallbacks
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions
        from langchain.chat_models import ChatOllama
        from langchain.memory import ConversationBufferWindowMemory
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning("LangChain not available")

class OllamaChatbot:
    """Modern emotion-aware chatbot using LangChain + ChatOllama"""
    
    def __init__(self, model_name: str = "gemma:2b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.llm = None
        self.memory = None
        self.conversation_history = []
        self._initialize()
    
    def _initialize(self):
        """Initialize ChatOllama client"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available - install with: pip install langchain langchain-community")
            return
            
        try:
            # Initialize ChatOllama with timeout and error handling
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.7,
                timeout=30  # 30 second timeout
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=True
            )
            
            # Test the connection with timeout
            try:
                test_response = self.llm.invoke([HumanMessage(content="Hello")])
                logger.info(f"Ollama chatbot initialized successfully with model: {self.model_name}")
            except Exception as test_error:
                logger.warning(f"Ollama test failed, but continuing: {test_error}")
                # Don't fail initialization if test fails
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama chatbot: {e}")
            print(f"Ollama initialization error: {e}")
            self.llm = None
    
    def get_response(self, message: str, face_emotion: str = "neutral", 
                    voice_emotion: str = "neutral", face_confidence: float = 0.0,
                    voice_confidence: float = 0.0, speech_text: str = None) -> str:
        """Generate emotion-aware response"""
        
        if not self.llm:
            return self._fallback_response(message, face_emotion, voice_emotion)
        
        try:
            # Build emotion context
            emotion_context = self._build_emotion_context(
                face_emotion, voice_emotion, face_confidence, voice_confidence, speech_text
            )
            
            # Create enhanced prompt
            system_prompt = self._create_system_prompt(emotion_context)
            
            # Prepare messages for Ollama
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            # Add conversation history
            if self.conversation_history:
                # Add last few exchanges for context
                for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                    if msg['role'] == 'user':
                        messages.insert(-1, HumanMessage(content=msg['content']))
                    else:
                        messages.insert(-1, AIMessage(content=msg['content']))
            
            # Get response from Ollama
            response = self.llm.invoke(messages)
            bot_response = response.content
            
            # Update conversation history
            self.conversation_history.append({'role': 'user', 'content': message})
            self.conversation_history.append({'role': 'assistant', 'content': bot_response})
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return self._clean_response(bot_response)
            
        except Exception as e:
            logger.error(f"Ollama response error: {e}")
            return self._fallback_response(message, face_emotion, voice_emotion)
    
    def _build_emotion_context(self, face_emotion: str, voice_emotion: str, 
                              face_confidence: float, voice_confidence: float, 
                              speech_text: str) -> Dict[str, Any]:
        """Build emotion context for the prompt"""
        
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
        
        return {
            'primary_emotion': primary_emotion,
            'face_emotion': face_emotion,
            'voice_emotion': voice_emotion,
            'confidence': confidence,
            'speech_text': speech_text
        }
    
    def _create_system_prompt(self, emotion_context: Dict[str, Any]) -> str:
        """Create emotion-aware system prompt"""
        
        base_prompt = """You are DailyNest AI, an empathetic and supportive conversational assistant focused on emotional well-being and daily life support.

Your role:
- Provide thoughtful, caring responses that acknowledge the user's emotional state
- Offer practical advice and emotional support
- Be conversational and natural, not robotic
- Keep responses concise but meaningful (2-3 sentences max)
- Remember previous parts of the conversation

Guidelines:
- Always acknowledge emotions when detected
- Be supportive without being overly clinical
- Ask follow-up questions to encourage sharing
- Provide actionable suggestions when appropriate"""

        # Add emotion-specific context
        emotion = emotion_context.get('primary_emotion', 'neutral')
        confidence = emotion_context.get('confidence', 0.0)
        
        if confidence > 0.6:
            emotion_guidance = {
                'happy': "The user seems happy and positive. Match their energy while being supportive.",
                'sad': "The user appears sad or down. Be extra gentle and empathetic in your response.",
                'angry': "The user seems frustrated or angry. Acknowledge their feelings and help them process.",
                'surprised': "The user appears surprised. Be curious about what's happening in their life.",
                'fear': "The user seems anxious or worried. Provide reassurance and practical support.",
                'neutral': "The user's emotional state is neutral. Be warm and engaging."
            }
            
            base_prompt += f"\n\nCurrent context: {emotion_guidance.get(emotion, emotion_guidance['neutral'])}"
        
        return base_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any system artifacts
        response = response.strip()
        
        # Ensure reasonable length
        if len(response) > 400:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def _fallback_response(self, message: str, face_emotion: str, voice_emotion: str) -> str:
        """Fallback responses when Ollama is unavailable"""
        
        emotion_responses = {
            'happy': "I can sense your positive energy! That's wonderful to hear. What's been going well for you?",
            'sad': "I can sense you might be going through a difficult time. I'm here to listen and support you.",
            'angry': "I can sense some frustration. It's completely normal to feel this way sometimes. What's on your mind?",
            'surprised': "You seem surprised! Something unexpected must have happened. Tell me more about it.",
            'fear': "I sense you might be feeling anxious. That's okay - we all experience worry sometimes. How can I help?",
            'neutral': "I'm here to listen and chat with you. What's on your mind today?"
        }
        
        # Choose response based on dominant emotion
        primary_emotion = face_emotion if face_emotion != 'neutral' else voice_emotion
        base_response = emotion_responses.get(primary_emotion, emotion_responses['neutral'])
        
        # Add contextual response to the message
        if any(word in message.lower() for word in ['help', 'support', 'advice']):
            base_response += " How can I best support you right now?"
        elif any(word in message.lower() for word in ['work', 'job', 'stress']):
            base_response += " Work-related challenges can really impact our well-being."
        elif any(word in message.lower() for word in ['family', 'relationship']):
            base_response += " Relationships are such an important part of our emotional lives."
        
        return base_response
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_history = []
        if self.memory:
            self.memory.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
    
    def switch_model(self, model_name: str):
        """Switch to a different Ollama model"""
        self.model_name = model_name
        self._initialize()

# Global instance
_ollama_chatbot = None

def get_ollama_chatbot(model_name: str = "gemma:2b") -> OllamaChatbot:
    """Get global Ollama chatbot instance"""
    global _ollama_chatbot
    if _ollama_chatbot is None or _ollama_chatbot.model_name != model_name:
        _ollama_chatbot = OllamaChatbot(model_name)
    return _ollama_chatbot

def initialize_ollama_chatbot(model_name: str = "gemma:2b"):
    """Initialize Ollama chatbot with specific model"""
    global _ollama_chatbot
    _ollama_chatbot = OllamaChatbot(model_name)
    return _ollama_chatbot
