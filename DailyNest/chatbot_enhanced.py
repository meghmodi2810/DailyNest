"""
Enhanced Chatbot for DailyNest - Improved memory and response handling
"""

import os
import logging
import requests
import json
import time
from typing import Optional, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

class EnhancedOllamaChatbot:
    """Enhanced chatbot using direct Ollama API calls with improved memory"""
    
    def __init__(self, model_name: str = "gemma:2b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = deque(maxlen=20)  # Keep last 20 exchanges
        self.system_prompt = self._create_base_system_prompt()
        self._connection_tested = False
        self._last_response_time = 0
        self._response_count = 0
    
    def _create_base_system_prompt(self):
        """Create the base system prompt"""
        return """You are DailyNest AI, an empathetic and supportive conversational assistant focused on emotional well-being and daily life support.

Your role:
- Provide thoughtful, caring responses that acknowledge the user's emotional state
- Offer practical advice and emotional support
- Be conversational and natural, not robotic
- Keep responses concise but meaningful (2-3 sentences max)
- Remember previous parts of the conversation
- Always be helpful and supportive

Guidelines:
- Always acknowledge emotions when detected
- Be supportive without being overly clinical
- Ask follow-up questions to encourage sharing
- Provide actionable suggestions when appropriate
- Use a warm, friendly tone
- Avoid being repetitive"""
    
    def _test_connection(self):
        """Test connection to Ollama"""
        if self._connection_tested:
            return True
            
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                if self.model_name in models:
                    logger.info(f"Ollama connection successful with model: {self.model_name}")
                    self._connection_tested = True
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
        """Generate emotion-aware response using Ollama API with enhanced memory"""
        
        start_time = time.time()
        
        try:
            # Test connection if not already tested
            if not self._test_connection():
                return self._fallback_response(message, face_emotion, voice_emotion)
            
            # Build emotion context
            emotion_context = self._build_emotion_context(
                face_emotion, voice_emotion, face_confidence, voice_confidence, speech_text
            )
            
            # Create enhanced prompt with emotion context
            system_prompt = self._create_emotion_aware_prompt(emotion_context)
            
            # Prepare conversation history
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add conversation history (last 6 exchanges for context)
            for msg in list(self.conversation_history)[-12:]:
                messages.append(msg)
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Call Ollama API with retry logic
            bot_response = self._call_ollama_with_retry(messages)
            
            if bot_response:
                # Update conversation history
                self.conversation_history.append({'role': 'user', 'content': message})
                self.conversation_history.append({'role': 'assistant', 'content': bot_response})
                
                # Update metrics
                self._last_response_time = time.time() - start_time
                self._response_count += 1
                
                logger.info(f"Chatbot response generated in {self._last_response_time:.2f}s")
                
                return self._clean_response(bot_response)
            else:
                return self._fallback_response(message, face_emotion, voice_emotion)
                
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            return self._fallback_response(message, face_emotion, voice_emotion)
    
    def _call_ollama_with_retry(self, messages, max_retries=2):
        """Call Ollama API with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40,
                            "num_ctx": 4096
                        }
                    },
                    timeout=45  # Increased timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get('message', {}).get('content', '')
                    if bot_response.strip():
                        return bot_response
                    else:
                        logger.warning("Empty response from Ollama")
                        return None
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    if attempt < max_retries:
                        time.sleep(1)  # Wait before retry
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    time.sleep(2)  # Wait longer before retry
                    continue
                return None
            except Exception as e:
                logger.error(f"Ollama API call error: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return None
        
        return None
    
    def _build_emotion_context(self, face_emotion: str, voice_emotion: str, 
                              face_confidence: float, voice_confidence: float, 
                              speech_text: str) -> Dict[str, Any]:
        """Build emotion context for the prompt"""
        
        # Determine dominant emotion
        if face_confidence > voice_confidence and face_confidence > 0.6:
            primary_emotion = face_emotion
            confidence = face_confidence
            source = "facial expression"
        elif voice_confidence > 0.6:
            primary_emotion = voice_emotion
            confidence = voice_confidence
            source = "voice tone"
        else:
            primary_emotion = "neutral"
            confidence = max(face_confidence, voice_confidence)
            source = "general"
        
        return {
            'primary_emotion': primary_emotion,
            'face_emotion': face_emotion,
            'voice_emotion': voice_emotion,
            'confidence': confidence,
            'source': source,
            'speech_text': speech_text
        }
    
    def _create_emotion_aware_prompt(self, emotion_context: Dict[str, Any]) -> str:
        """Create emotion-aware system prompt"""
        
        base_prompt = self.system_prompt
        
        # Add emotion-specific context
        emotion = emotion_context.get('primary_emotion', 'neutral')
        confidence = emotion_context.get('confidence', 0.0)
        source = emotion_context.get('source', 'general')
        
        if confidence > 0.6:
            emotion_guidance = {
                'happy': f"The user seems happy and positive (detected from {source}). Match their energy while being supportive and encouraging.",
                'sad': f"The user appears sad or down (detected from {source}). Be extra gentle, empathetic, and offer comfort in your response.",
                'angry': f"The user seems frustrated or angry (detected from {source}). Acknowledge their feelings, help them process, and offer calming support.",
                'surprised': f"The user appears surprised (detected from {source}). Be curious about what's happening and show genuine interest.",
                'fear': f"The user seems anxious or worried (detected from {source}). Provide reassurance, practical support, and help them feel safe.",
                'neutral': f"The user's emotional state is neutral (detected from {source}). Be warm, engaging, and supportive."
            }
            
            guidance = emotion_guidance.get(emotion, emotion_guidance['neutral'])
            base_prompt += f"\n\nCurrent emotional context: {guidance}"
        
        # Add speech context if available
        speech_text = emotion_context.get('speech_text')
        if speech_text:
            base_prompt += f"\n\nUser's speech: \"{speech_text}\""
        
        return base_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any system artifacts
        response = response.strip()
        
        # Remove common AI prefixes
        prefixes_to_remove = [
            "I'm DailyNest AI, and",
            "As DailyNest AI,",
            "DailyNest AI here,",
            "I am DailyNest AI, and"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure reasonable length
        if len(response) > 400:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        # Ensure it starts with a capital letter
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        return response
    
    def _fallback_response(self, message: str, face_emotion: str, voice_emotion: str) -> str:
        """Enhanced fallback responses when Ollama is unavailable"""
        
        emotion_responses = {
            'happy': "I can sense your positive energy! That's wonderful to hear. What's been going well for you lately?",
            'sad': "I can sense you might be going through a difficult time. I'm here to listen and support you. Would you like to talk about what's on your mind?",
            'angry': "I can sense some frustration. It's completely normal to feel this way sometimes. What's been bothering you? I'm here to help you work through it.",
            'surprised': "You seem surprised! Something unexpected must have happened. Tell me more about it - I'm curious to hear your story.",
            'fear': "I sense you might be feeling anxious. That's okay - we all experience worry sometimes. How can I help you feel more at ease?",
            'neutral': "I'm here to listen and chat with you. What's on your mind today? I'm ready to support you however I can."
        }
        
        # Choose response based on dominant emotion
        primary_emotion = face_emotion if face_emotion != 'neutral' else voice_emotion
        base_response = emotion_responses.get(primary_emotion, emotion_responses['neutral'])
        
        # Add contextual response to the message
        message_lower = message.lower()
        if any(word in message_lower for word in ['help', 'support', 'advice']):
            base_response += " How can I best support you right now?"
        elif any(word in message_lower for word in ['work', 'job', 'stress', 'busy']):
            base_response += " Work-related challenges can really impact our well-being. Remember to take care of yourself."
        elif any(word in message_lower for word in ['family', 'relationship', 'friend']):
            base_response += " Relationships are such an important part of our emotional lives. It's great that you're thinking about them."
        elif any(word in message_lower for word in ['tired', 'sleep', 'rest']):
            base_response += " Taking care of your physical needs is so important for emotional well-being."
        
        return base_response
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_history.clear()
        logger.info("Chatbot memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return list(self.conversation_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            'total_responses': self._response_count,
            'last_response_time': self._last_response_time,
            'memory_size': len(self.conversation_history),
            'model': self.model_name,
            'connection_tested': self._connection_tested
        }
    
    def reset_connection(self):
        """Reset connection test flag to force re-testing"""
        self._connection_tested = False
        logger.info("Connection test reset")

# Global instance
_enhanced_chatbot = None

def get_enhanced_chatbot(model_name: str = "gemma:2b") -> EnhancedOllamaChatbot:
    """Get global enhanced chatbot instance"""
    global _enhanced_chatbot
    if _enhanced_chatbot is None or _enhanced_chatbot.model_name != model_name:
        _enhanced_chatbot = EnhancedOllamaChatbot(model_name)
    return _enhanced_chatbot 