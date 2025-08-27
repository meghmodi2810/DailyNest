from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.core.validators import RegexValidator

class EmotionRecord(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('surprised', 'Surprised'),
        ('neutral', 'Neutral'),
        ('calm', 'Calm'),
        ('excited', 'Excited'),
        ('fear', 'Fear'),
        ('disgust', 'Disgust'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    face_emotion = models.CharField(max_length=50, choices=EMOTION_CHOICES, null=True, blank=True)
    voice_emotion = models.CharField(max_length=50, choices=EMOTION_CHOICES, null=True, blank=True)
    face_confidence = models.FloatField(default=0.0, help_text="Confidence score for face emotion (0-1)")
    voice_confidence = models.FloatField(default=0.0, help_text="Confidence score for voice emotion (0-1)")
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['face_emotion']),
            models.Index(fields=['voice_emotion']),
        ]

    def __str__(self):
        return f"Emotion Record - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def dominant_emotion(self):
        """Get the emotion with higher confidence"""
        if self.face_confidence > self.voice_confidence:
            return self.face_emotion or 'neutral'
        return self.voice_emotion or 'neutral'

class ChatMessage(models.Model):
    SENDER_CHOICES = [
        ('user', 'User'),
        ('bot', 'Bot'),
    ]
    
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    emotion_context = models.ForeignKey(EmotionRecord, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.sender}: {self.message[:50]}..."

class UserPreference(models.Model):
    THEME_CHOICES = [
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('high-contrast', 'High Contrast'),
    ]
    
    FONT_SIZE_CHOICES = [
        ('small', 'Small'),
        ('medium', 'Medium'),
        ('large', 'Large'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preferences')
    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default='light')
    font_size = models.CharField(max_length=10, choices=FONT_SIZE_CHOICES, default='medium')
    reduce_animations = models.BooleanField(default=True)
    high_contrast_mode = models.BooleanField(default=False)
    text_to_speech = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Preference'
        verbose_name_plural = 'User Preferences'
    
    def __str__(self):
        return f"{self.user.username} Preferences - Theme: {self.theme}, Font Size: {self.font_size}"
