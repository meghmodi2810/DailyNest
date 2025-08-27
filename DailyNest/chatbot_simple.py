"""
Simple Chatbot for DailyNest - Direct Ollama integration without LangChain
"""

import os
import logging
import requests
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class SimpleOllamaChatbot:
    """Simple chatbot using direct Ollama API calls"""
    
    def __init__(self, model_name: str = "gemma:2b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                if self.model_name in models:
                    logger.info(f"Ollama connection successful with model: {self.model_name}")
                    return True
                else:
                    logger.warning(f"Model {self.model_name} not found. Available: {models}")
                    return False
            else:
                logger.error(f"Ollama connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            return False
    
    def get_response(self, message: str, face_emotion: str = "neutral", 
                    voice_emotion: str = "neutral", face_confidence: float = 0.0,
                    voice_confidence: float = 0.0, speech_text: str = None) -> str:
        """Generate emotion-aware response using Ollama API"""
        
        try:
            # Build emotion context
            emotion_context = self._build_emotion_context(
                face_emotion, voice_emotion, face_confidence, voice_confidence, speech_text
            )
            
            # Create enhanced prompt
            system_prompt = self._create_system_prompt(emotion_context)
            
            # Prepare conversation history
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add conversation history (last 4 exchanges)
            for msg in self.conversation_history[-8:]:
                messages.append(msg)
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get('message', {}).get('content', '')
                
                # Update conversation history
                self.conversation_history.append({'role': 'user', 'content': message})
                self.conversation_history.append({'role': 'assistant', 'content': bot_response})
                
                # Keep history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                return self._clean_response(bot_response)
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._fallback_response(message, face_emotion, voice_emotion)
                
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
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
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

# Global instance
_simple_chatbot = None

def get_simple_chatbot(model_name: str = "gemma:2b") -> SimpleOllamaChatbot:
    """Get global simple chatbot instance"""
    global _simple_chatbot
    if _simple_chatbot is None or _simple_chatbot.model_name != model_name:
        _simple_chatbot = SimpleOllamaChatbot(model_name)
    return _simple_chatbot 