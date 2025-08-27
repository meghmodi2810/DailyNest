# 🚀 DailyNest Ollama Migration Guide

## Overview
Your DailyNest chatbot has been successfully migrated from Groq API to local Ollama integration using LangChain. This guide will help you set up and run Ollama locally.

## 🔧 Prerequisites

### 1. Install Dependencies
```bash
pip install langchain langchain-community
```

### 2. Install Ollama
**Windows:**
- Download from: https://ollama.ai/download/windows
- Run the installer
- Ollama will start automatically

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## 🤖 Available Models

### Popular Models for DailyNest:
- **llama2** (7B) - Good balance of speed and quality
- **mistral** (7B) - Fast and efficient
- **gemma** (7B) - Google's model, good for conversations
- **llama2:13b** - Higher quality, slower
- **codellama** - Good for technical discussions

## 🚀 Setup Steps

### 1. Start Ollama Service
```bash
# Start Ollama (runs on http://localhost:11434)
ollama serve
```

### 2. Download Your Preferred Model
```bash
# Download llama2 (default in DailyNest)
ollama pull llama2

# Or download other models
ollama pull mistral
ollama pull gemma
```

### 3. Test Ollama
```bash
# Test the model
ollama run llama2
# Type a message and press Enter to test
# Type /bye to exit
```

### 4. Update DailyNest Model (Optional)
Edit `views.py` line 186 to change model:
```python
chatbot = get_ollama_chatbot(model_name="mistral")  # Change to your preferred model
```

Or use the Django command:
```bash
python manage.py switch_model --model mistral
```

## 🔄 Running DailyNest with Ollama

### 1. Start Ollama
```bash
ollama serve
```

### 2. Start Django
```bash
python manage.py runserver
```

### 3. Test Chat
- Go to http://127.0.0.1:8000/chat/
- Send a message
- You should get intelligent AI responses!

## 🎯 Key Features

### ✅ What's New:
- **No API Key Required** - Runs completely locally
- **Conversation Memory** - Remembers previous messages
- **Emotion Awareness** - Responds based on detected emotions
- **Model Switching** - Easy to change models
- **Privacy** - All data stays on your machine

### 🔧 Configuration Options:

**Change Model in Code:**
```python
# In views.py line 186
chatbot = get_ollama_chatbot(model_name="your_model_name")
```

**Available Models:**
- `llama2` (default)
- `mistral`
- `gemma`
- `codellama`
- `llama2:13b`

## 🐛 Troubleshooting

### Error: "I'm having trouble connecting to the local AI"
**Solution:**
1. Make sure Ollama is running: `ollama serve`
2. Check if model is downloaded: `ollama list`
3. Test model directly: `ollama run llama2`

### Error: "Model not found"
**Solution:**
```bash
ollama pull your_model_name
```

### Slow Responses
**Solutions:**
1. Use smaller models (llama2 vs llama2:13b)
2. Ensure sufficient RAM (8GB+ recommended)
3. Close other applications

### Connection Refused
**Solution:**
1. Check Ollama is running on port 11434
2. Restart Ollama: `ollama serve`

## 🚀 Performance Tips

### 1. Model Selection
- **Fast**: mistral, gemma
- **Balanced**: llama2
- **High Quality**: llama2:13b

### 2. System Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM
- **Storage**: 4-8GB per model

### 3. Optimization
- Keep Ollama running in background
- Use SSD for better model loading
- Close unnecessary applications

## 🔄 Model Management

### List Downloaded Models
```bash
ollama list
```

### Remove Models
```bash
ollama rm model_name
```

### Update Models
```bash
ollama pull model_name
```

## 🎉 Success Indicators

When everything works correctly:
1. Ollama shows: "Ollama is running on http://localhost:11434"
2. Django console shows: "Ollama chatbot initialized successfully with model: llama2"
3. Chat responses are dynamic and contextual
4. No API key errors

## 🔧 Advanced Configuration

### Custom Ollama URL
If running Ollama on different port/host:
```python
# In chatbot_ollama.py
chatbot = OllamaChatbot(
    model_name="llama2",
    base_url="http://your-host:your-port"
)
```

### Memory Management
```python
# Clear conversation history
chatbot.clear_memory()

# Get conversation history
history = chatbot.get_conversation_history()
```

## 📝 Migration Summary

### ✅ Completed:
- ✅ Removed Groq API dependency
- ✅ Implemented ChatOllama integration
- ✅ Added conversation memory
- ✅ Updated Django views
- ✅ Created model switching utility
- ✅ Updated requirements.txt

### 🎯 Next Steps:
1. Install Ollama
2. Download preferred model
3. Start Ollama service
4. Test DailyNest chat

Your DailyNest is now ready for local AI conversations! 🚀
