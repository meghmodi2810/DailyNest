# DailyNest ML Models

This directory contains machine learning models for emotion detection.

## Files:
- `emotion_model_weights.h5` - Pre-trained weights for facial emotion recognition CNN
- `face_emotion/` - Face emotion detection models and utilities
- `voice_emotion/` - Voice emotion detection models and utilities

## Usage:
The emotion detection models are automatically loaded by the EmotionDetector class in utils.py.
If model files are missing, the system will use fallback detection methods.

## Training:
To train custom models, ensure you have the proper datasets and run the training scripts.