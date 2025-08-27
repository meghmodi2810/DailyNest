# DailyNest Production Deployment Guide

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python setup_production.py
   ```

2. **Run Application**
   ```bash
   python manage.py runserver
   ```

3. **Test Features**
   - Visit: http://127.0.0.1:8000
   - Test emotion detection: http://127.0.0.1:8000/emotion/
   - Test chat: http://127.0.0.1:8000/chat/

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 5GB free space for models and dependencies
- **GPU**: Optional (NVIDIA GPU with CUDA for faster TensorFlow inference)

### Software Requirements
- **Python**: 3.8 - 3.11
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## 🔧 Installation Steps

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup
Ensure ML models are in place:
- `models/face_emotion/fer.h5` (primary emotion model)
- `models/face_emotion/best_mobilenet_model.h5` (backup model)
- `emotion_model_weights.h5` (fallback model)

### 3. Database Configuration
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser  # Optional
```

### 4. Static Files
```bash
python manage.py collectstatic
```

## 🧪 Testing

### Run Production Tests
```bash
python test_production.py
```

### Manual Testing Checklist
- [ ] Emotion detection works with webcam
- [ ] Voice recording and processing
- [ ] Chat responses are contextual
- [ ] Database saves emotion records
- [ ] Static files load correctly
- [ ] Admin interface accessible

## 🔍 Troubleshooting

### Common Issues

#### 1. TensorFlow Import Errors
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.16.0,<2.18.0
```

#### 2. OpenCV Issues
```bash
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python>=4.8.0
```

#### 3. Audio Processing Errors
```bash
# Install audio dependencies
pip install pyaudio soundfile librosa
```

#### 4. Model Loading Failures
- Check model file sizes (should be > 1KB)
- Verify model paths in settings
- Check file permissions

### Performance Issues

#### Slow Emotion Detection
- Reduce image resolution in frontend
- Enable GPU acceleration for TensorFlow
- Use model caching

#### High Memory Usage
- Limit concurrent requests
- Implement request queuing
- Use lighter models

## 📊 Monitoring & Logging

### Log Files
- Application logs: `dailynest.log`
- Django logs: Console output
- Error tracking: Check Django admin

### Performance Metrics
- Emotion detection response time: < 2 seconds
- Chat response time: < 1 second
- Memory usage: < 2GB per process

## 🔒 Security Considerations

### Production Settings
```python
# In config/settings.py
DEBUG = False
ALLOWED_HOSTS = ['your-domain.com', 'localhost']
SECRET_KEY = 'your-secure-secret-key'

# Use environment variables
import os
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
```

### HTTPS Configuration
- Use SSL certificates
- Configure secure headers
- Enable CSRF protection

## 🚢 Deployment Options

### 1. Local Development
```bash
python manage.py runserver 0.0.0.0:8000
```

### 2. Production Server (Gunicorn)
```bash
pip install gunicorn
gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

### 3. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### 4. Cloud Deployment
- **Heroku**: Use Procfile and requirements.txt
- **AWS**: Deploy with Elastic Beanstalk or EC2
- **Google Cloud**: Use App Engine or Compute Engine

## 📈 Scaling Considerations

### Database Optimization
- Use PostgreSQL for production
- Implement database indexing
- Set up read replicas

### Caching
- Redis for session storage
- Memcached for query caching
- CDN for static files

### Load Balancing
- Nginx reverse proxy
- Multiple Django instances
- Separate ML processing workers

## 🔄 Maintenance

### Regular Tasks
- Monitor log files
- Update dependencies monthly
- Backup database weekly
- Review performance metrics

### Model Updates
- Retrain models with new data
- A/B test model performance
- Gradual model rollouts

## 📞 Support

### Debug Mode
Enable detailed logging:
```python
LOGGING['loggers']['DailyNest']['level'] = 'DEBUG'
```

### Health Check Endpoint
Create monitoring endpoint:
```python
# In views.py
def health_check(request):
    return JsonResponse({
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'ml_available': ML_AVAILABLE,
        'audio_available': AUDIO_ML_AVAILABLE
    })
```

## 🎯 Performance Benchmarks

### Expected Performance
- **Emotion Detection**: 0.5-2.0 seconds
- **Speech Processing**: 1.0-3.0 seconds  
- **Chat Response**: 0.1-0.5 seconds
- **Memory Usage**: 1-2GB
- **CPU Usage**: 20-60% during processing

### Optimization Tips
1. Use model quantization for faster inference
2. Implement request batching
3. Cache frequent responses
4. Optimize image preprocessing
5. Use async processing for heavy tasks

---

## 🏆 Production Checklist

Before going live:

- [ ] All tests pass
- [ ] Security settings configured
- [ ] SSL certificate installed
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Documentation updated
- [ ] Team trained on deployment

**Your DailyNest application is now production-ready! 🎉**
