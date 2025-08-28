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
    
    # Caregiver Management URLs
    path('manage-relationships/', views.manage_care_relationships, name='manage_care_relationships'),
    path('remove-relationship/<int:relationship_id>/', views.remove_care_relationship, name='remove_care_relationship'),
    
    # Admin Management URLs
    path('admin/manage-users/', views.admin_manage_users, name='admin_manage_users'),
    path('admin/assign-caregiver/<int:autistic_id>/', views.admin_assign_caregiver, name='admin_assign_caregiver'),
    path('admin/edit-user/<int:user_id>/', views.admin_edit_user, name='admin_edit_user'),
    path('admin/delete-user/<int:user_id>/', views.admin_delete_user, name='admin_delete_user'),
    
    # Application URLs
    path('home/', views.home, name='home'),
    path('emotion/', views.emotion, name='emotion'),
    path('chat/', views.chat, name='chat'),
    path('detect-emotion/', views.detect_emotion, name='detect_emotion'),
    path('chat-message/', views.chat_message, name='chat_message'),
    path('update-preferences/', views.update_preferences, name='update_preferences'),
    path('emotion-history/', views.emotion_history, name='emotion_history'),
    path('clear-chat/', views.clear_chat, name='clear_chat'),
    
    # Game URLs
    path('games/', views.games_dashboard, name='games_dashboard'),
    path('games/<str:game_type>/', views.play_game, name='play_game'),
    path('games/save-result/', views.save_game_result, name='save_game_result'),
    path('games/progress/', views.game_progress, name='game_progress'),
]
