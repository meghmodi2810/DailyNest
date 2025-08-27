from django.urls import path
from . import views

urlpatterns = [
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('verify-otp/<int:user_id>/', views.verify_otp_view, name='verify_otp'),
    path('reset-password/<int:user_id>/', views.reset_password_view, name='reset_password'),
    
    # Dashboard URLs
    path('', views.dashboard, name='dashboard'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('caregiver-dashboard/', views.caregiver_dashboard, name='caregiver_dashboard'),
    path('autistic-dashboard/', views.autistic_dashboard, name='autistic_dashboard'),
    path('profile/', views.profile_view, name='profile'),
    
    # Notes URLs
    path('schedule-note/', views.schedule_note_view, name='schedule_note'),
    path('scheduled-notes/', views.scheduled_notes_list_view, name='scheduled_notes_list'),
    
    # Application URLs
    path('home/', views.home, name='home'),
    path('emotion/', views.emotion, name='emotion'),
    path('chat/', views.chat, name='chat'),
    path('detect-emotion/', views.detect_emotion, name='detect_emotion'),
    path('chat-message/', views.chat_message, name='chat_message'),
    path('update-preferences/', views.update_preferences, name='update_preferences'),
    path('emotion-history/', views.emotion_history, name='emotion_history'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
]
