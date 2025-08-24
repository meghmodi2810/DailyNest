from django.db import models
from django.utils import timezone

class EmotionRecord(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    face_emotion = models.CharField(max_length=50, null=True, blank=True)
    voice_emotion = models.CharField(max_length=50, null=True, blank=True)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"Emotion Record - {self.timestamp}"

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
    
    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default='light')
    font_size = models.CharField(max_length=10, choices=FONT_SIZE_CHOICES, default='medium')
    reduce_animations = models.BooleanField(default=True)
    high_contrast_mode = models.BooleanField(default=False)
    text_to_speech = models.BooleanField(default=False)
    
    def __str__(self):
        return f"User Preferences - Theme: {self.theme}, Font Size: {self.font_size}"
