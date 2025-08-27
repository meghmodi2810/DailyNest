# DailyNest - Emotion-Aware AI Companion

DailyNest is an emotion-aware conversational AI system that combines face emotion detection, voice emotion analysis, and intelligent chatbot responses using local AI models.

## Features

- **Face Emotion Detection**: Uses FER-2013 CNN model for accurate facial emotion recognition
- **Voice Emotion Analysis**: OpenAI Whisper base for speech transcription + wav2vec2 for emotion classification
- **Intelligent Chatbot**: LangChain + ChatOllama with gemma:2b for contextual conversations
- **Real-time Processing**: Lazy-loaded models for optimal performance
- **Web Interface**: Modern, responsive UI for emotion detection and chat

## System Requirements

- **OS**: Windows 10/11 (tested on Windows 10)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **Python**: 3.10+
- **Ollama**: Latest version

## Quick Start

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull the Required Model

```bash
ollama pull gemma:2b
```

### 3. Clone and Setup DailyNest

```bash
# Clone the repository
git clone <repository-url>
cd DailyNest

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Start DailyNest

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run Django server
python manage.py runserver
```

### 6. Access the Application

Open your browser and navigate to: `http://127.0.0.1:8000`

## Testing

### Run All Tests

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run comprehensive test
python test_fixes.py
```

### Run Individual Tests

```bash
# Test face emotion detection
python tests/test_face_inference.py

# Test voice emotion detection
python tests/test_voice_inference.py

# Test chat integration
python tests/test_chat_integration.py
```

### Test Endpoints

```bash
# Test chat endpoint
curl -X POST http://127.0.0.1:8000/chat-message/ \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello"}'

# Test emotion detection endpoint
curl -X POST http://127.0.0.1:8000/detect-emotion/ \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Architecture

### ML Models

- **Face Emotion**: FER-2013 CNN (48x48 grayscale) → MediaPipe landmarks → OpenCV fallback
- **Voice Emotion**: OpenAI Whisper base (transcription) + wav2vec2 (emotion features) + text analysis
- **Chatbot**: LangChain + ChatOllama with gemma:2b

### Lazy Loading

Models are loaded only when first accessed to prevent Django startup freezes:

```python
# Models are loaded on first use
detector = get_emotion_detector()  # Loads FER-2013 model
processor = get_speech_processor()  # Loads Whisper + wav2vec2
chatbot = get_ollama_chatbot()  # Connects to Ollama
```

## Troubleshooting

### Django Server Freezes

**Problem**: Server freezes after TensorFlow warning
**Solution**: 
1. Ensure you're using the fixed ML models (`ml_models_fixed.py`)
2. Set environment variable: `set TF_ENABLE_ONEDNN_OPTS=0`
3. Check that all dependencies are installed correctly

### Ollama Connection Issues

**Problem**: Chatbot returns canned responses
**Solution**:
1. Verify Ollama is running: `ollama list`
2. Check gemma:2b is available: `ollama list | grep gemma`
3. Test connection: `curl http://localhost:11434/api/tags`

### ML Model Loading Errors

**Problem**: Face/voice detection fails
**Solution**:
1. Check model files exist in `models/` directory
2. Verify sufficient disk space and memory
3. Ensure all Python dependencies are installed

### Memory Issues

**Problem**: System runs out of memory
**Solution**:
1. Close other applications
2. Use smaller models (consider quantized versions)
3. Increase virtual memory/page file

## Performance Optimization

### Memory Usage

- **FER-2013 model**: ~23MB
- **Whisper base model**: ~244MB  
- **wav2vec2 base model**: ~95MB
- **Total ML models**: ~362MB
- **gemma:2b**: ~1.7GB (managed by Ollama)

### Response Times

- **Face emotion detection**: ~100-500ms
- **Voice emotion detection**: ~1-3 seconds
- **Chatbot response**: ~2-10 seconds

### Optimization Tips

1. **Use SSD storage** for faster model loading
2. **Close unnecessary applications** to free RAM
3. **Consider quantized models** for better performance
4. **Use background processing** for heavy operations

## Development

### Project Structure

```
DailyNest/
├── config/                 # Django settings
├── DailyNest/             # Main app
│   ├── ml_models_fixed.py # Fixed ML models
│   ├── chatbot_ollama.py  # Ollama chatbot
│   ├── views.py           # API endpoints
│   └── templates/         # HTML templates
├── models/                # ML model files
├── tests/                 # Test files
├── diagnostics/           # Diagnostics and reports
└── requirements.txt       # Dependencies
```

### Adding New Features

1. **New ML Model**: Add to `ml_models_fixed.py` with lazy loading
2. **New Endpoint**: Add to `views.py` and `urls.py`
3. **New UI**: Add template and static files
4. **New Test**: Add to `tests/` directory

### Environment Variables

```bash
# Optional: Disable TensorFlow optimizations
set TF_ENABLE_ONEDNN_OPTS=0

# Optional: Set Django secret key
set DJANGO_SECRET_KEY=your-secret-key

# Optional: Set debug mode
set DJANGO_DEBUG=True
```

## Support

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Port Conflicts**: Change Django port with `python manage.py runserver 8001`
3. **Model Download Issues**: Check internet connection and disk space
4. **Permission Errors**: Run as administrator if needed

### Getting Help

1. Check the `diagnostics/report.md` for detailed issue analysis
2. Run `python test_fixes.py` to identify specific problems
3. Check the logs in `dailynest.log`
4. Verify all prerequisites are installed correctly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Acknowledgments

- **FER-2013**: Facial Emotion Recognition dataset and model
- **OpenAI Whisper**: Speech recognition model
- **wav2vec2**: Audio feature extraction
- **Ollama**: Local LLM inference
- **LangChain**: LLM framework
- **Django**: Web framework 