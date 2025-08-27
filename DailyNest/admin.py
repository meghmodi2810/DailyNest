from django.contrib import admin
from .models import EmotionRecord, ChatMessage, UserPreference

@admin.register(EmotionRecord)
class EmotionRecordAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'user', 'face_emotion', 'voice_emotion', 'face_confidence', 'voice_confidence')
    list_filter = ('face_emotion', 'voice_emotion', 'user')
    search_fields = ('face_emotion', 'voice_emotion', 'notes', 'user__username')
    readonly_fields = ('timestamp', 'dominant_emotion')
    
    def dominant_emotion(self, obj):
        return obj.dominant_emotion
    dominant_emotion.short_description = 'Dominant Emotion'

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('sender', 'message', 'timestamp')
    list_filter = ('sender', 'timestamp')
    search_fields = ('message',)

@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ('user', 'theme', 'font_size', 'reduce_animations', 'high_contrast_mode', 'text_to_speech', 'updated_at')
    list_filter = ('theme', 'font_size', 'reduce_animations', 'high_contrast_mode', 'text_to_speech')
    search_fields = ('user__username',)
    readonly_fields = ('created_at', 'updated_at')
