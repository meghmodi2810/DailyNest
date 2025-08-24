from django.contrib import admin
from .models import EmotionRecord, ChatMessage, UserPreference

@admin.register(EmotionRecord)
class EmotionRecordAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'face_emotion', 'voice_emotion')
    list_filter = ('face_emotion', 'voice_emotion')
    search_fields = ('face_emotion', 'voice_emotion', 'notes')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('sender', 'message', 'timestamp')
    list_filter = ('sender', 'timestamp')
    search_fields = ('message',)

@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ('theme', 'font_size', 'reduce_animations', 'high_contrast_mode', 'text_to_speech')
