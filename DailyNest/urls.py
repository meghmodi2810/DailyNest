from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('home/', views.home, name='home'),
    path('emotion/', views.emotion, name='emotion'),
    path('chat/', views.chat, name='chat'),
    path('detect-emotion/', views.detect_emotion, name='detect_emotion'),
    path('chat-message/', views.chat_message, name='chat_message'),
    path('update-preferences/', views.update_preferences, name='update_preferences'),
    path('emotion-history/', views.emotion_history, name='emotion_history'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]
