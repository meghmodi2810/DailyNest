from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import CustomUser, EmotionRecord, ChatMessage, UserPreference, PasswordResetOTP, ScheduledNote
from .forms import CustomUserRegistrationForm, CustomLoginForm, UserProfileForm, ForgotPasswordForm, OTPVerificationForm, ResetPasswordForm, ScheduledNoteForm
from .ml_models_fallback import get_emotion_detector, get_speech_processor

# Import enhanced chatbot
from .chatbot_enhanced import get_enhanced_chatbot
import logging
import json
import base64
import tempfile
import os
import random
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Authentication Views
def login_view(request):
    """Custom login view with role-based authentication"""
    if request.user.is_authenticated:
        return redirect_to_dashboard(request.user)
    
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.name}!')
            return redirect_to_dashboard(user)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomLoginForm()
    
    return render(request, 'auth/login.html', {'form': form})

def register_view(request):
    """User registration view with caregiver invitation system"""
    if request.user.is_authenticated:
        return redirect_to_dashboard(request.user)
    
    if request.method == 'POST':
        form = CustomUserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Handle caregiver invitation for autistic persons
            if (user.role == 'autistic_person' and 
                form.cleaned_data.get('caregiver_email')):
                
                from .models import CaregiverInvitation
                import uuid
                from datetime import datetime, timedelta
                
                # Create invitation
                invitation = CaregiverInvitation.objects.create(
                    autistic_person=user,
                    caregiver_email=form.cleaned_data['caregiver_email'],
                    relationship_type=form.cleaned_data.get('relationship_type', 'guardian'),
                    invitation_token=str(uuid.uuid4()),
                    message=form.cleaned_data.get('invitation_message', ''),
                    expires_at=datetime.now() + timedelta(days=7)
                )
                
                # TODO: Send email invitation (for now, just show success message)
                messages.success(request, f'Account created successfully! Invitation sent to {invitation.caregiver_email}')
            else:
                messages.success(request, f'Account created successfully for {user.name}!')
            
            login(request, user)
            return redirect_to_dashboard(user)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserRegistrationForm()
    
    return render(request, 'auth/register.html', {'form': form})

def logout_view(request):
    """User logout view"""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('login')

def redirect_to_dashboard(user):
    """Redirect user to appropriate dashboard based on role"""
    if user.role == 'admin':
        return redirect('admin_dashboard')
    elif user.role == 'caregiver':
        return redirect('caregiver_dashboard')
    else:  # autistic_person
        return redirect('autistic_dashboard')

# Dashboard Views
@login_required
def admin_dashboard(request):
    """Admin dashboard with system overview"""
    context = {
        'user_count': CustomUser.objects.count(),
        'emotion_records_count': EmotionRecord.objects.count(),
        'chat_messages_count': ChatMessage.objects.count(),
        'recent_users': CustomUser.objects.order_by('-created_at')[:5],
    }
    return render(request, 'dashboards/admin_dashboard.html', context)

@login_required
def caregiver_dashboard(request):
    """Caregiver dashboard with patient monitoring"""
    from .models import CareRelationship
    
    # Get only autistic persons under this caregiver's care
    care_relationships = CareRelationship.objects.filter(
        caregiver=request.user, 
        is_active=True
    ).select_related('autistic_person')
    
    autistic_users = [rel.autistic_person for rel in care_relationships]
    
    # Get recent emotions only for users under care
    user_ids = [user.id for user in autistic_users]
    recent_emotions = EmotionRecord.objects.filter(
        user_id__in=user_ids
    ).order_by('-timestamp')[:10]
    
    context = {
        'autistic_users': autistic_users,
        'recent_emotions': recent_emotions,
        'care_relationships': care_relationships,
    }
    return render(request, 'dashboards/caregiver_dashboard.html', context)

@login_required
def autistic_dashboard(request):
    """Autistic person dashboard with personal tools"""
    from .models import CareRelationship
    
    recent_emotions = EmotionRecord.objects.filter(user=request.user).order_by('-timestamp')[:5]
    recent_chats = ChatMessage.objects.filter(sender='user').order_by('-timestamp')[:5]
    
    # Get caregiver information
    caregiver_relationships = CareRelationship.objects.filter(
        autistic_person=request.user,
        is_active=True
    ).select_related('caregiver')
    
    context = {
        'recent_emotions': recent_emotions,
        'recent_chats': recent_chats,
        'caregiver_relationships': caregiver_relationships,
    }
    return render(request, 'dashboards/autistic_dashboard.html', context)

@login_required
def dashboard(request):
    """Dashboard page view - redirects to appropriate role dashboard"""
    return redirect_to_dashboard(request.user)

def home(request):
    """Home page view"""
    return render(request, 'home.html')

def emotion(request):
    """Emotion detection page view"""
    # Get or create preferences for the current user (or default user if anonymous)
    if request.user.is_authenticated:
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
    else:
        # For anonymous users, try to get the first superuser or create default preferences
        default_user = CustomUser.objects.filter(is_superuser=True).first()
        if default_user:
            preferences, created = UserPreference.objects.get_or_create(user=default_user)
        else:
            # Create a minimal context without preferences
            preferences = None
    
    context = {
        'preferences': preferences,
    }
    return render(request, 'emotion.html', context)

def chat(request):
    """Chat page view"""
    # Get or create preferences for the current user (or default user if anonymous)
    if request.user.is_authenticated:
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
    else:
        # For anonymous users, try to get the first superuser or create default preferences
        default_user = CustomUser.objects.filter(is_superuser=True).first()
        if default_user:
            preferences, created = UserPreference.objects.get_or_create(user=default_user)
        else:
            # Create a minimal context without preferences
            preferences = None
    
    # Get recent chat messages
    recent_messages = ChatMessage.objects.order_by('-timestamp')[:20]
    context = {
        'preferences': preferences,
        'recent_messages': reversed(recent_messages),
    }
    return render(request, 'chat.html', context)

@csrf_exempt
def detect_emotion(request):
    """Enhanced emotion detection endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            face_emotion = None
            voice_emotion = None
            face_confidence = 0.0
            voice_confidence = 0.0
            speech_text = None
            
            detector = get_emotion_detector()
            speech_processor = get_speech_processor()

            # Process face image
            if 'image' in data:
                try:
                    face_emotion, face_confidence = detector.detect_face_emotion(data['image'])
                except Exception as e:
                    logger.error(f"Face processing error: {str(e)}")
                    face_emotion, face_confidence = "neutral", 0.0

            # Process audio with speech recognition
            if 'audio' in data:
                try:
                    # Decode base64 audio
                    audio_data = base64.b64decode(data['audio'].split(',')[1])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                        temp_audio.write(audio_data)
                        temp_audio_path = temp_audio.name
                    
                    # Process audio for speech and emotion
                    speech_text, voice_emotion, voice_confidence = speech_processor.process_audio_file(temp_audio_path)
                    
                    # Clean up
                    os.unlink(temp_audio_path)
                    
                except Exception as e:
                    logger.error(f"Audio processing error: {str(e)}")
                    voice_emotion, voice_confidence = "neutral", 0.0
                    speech_text = None

            # Save emotion record with confidence scores
            record = EmotionRecord.objects.create(
                face_emotion=face_emotion if face_emotion and not face_emotion.startswith('Error') else None,
                voice_emotion=voice_emotion if voice_emotion and not voice_emotion.startswith('Error') else None,
                face_confidence=face_confidence,
                voice_confidence=voice_confidence,
                notes=f"Face: {face_emotion or 'None'} ({face_confidence:.2f}), Voice: {voice_emotion or 'None'} ({voice_confidence:.2f}), Speech: {speech_text or 'None'}"
            )

            return JsonResponse({
                'success': True,
                'face_emotion': face_emotion or "No face detected",
                'voice_emotion': voice_emotion or "No voice detected",
                'speech_text': speech_text or "No speech detected",
                'face_confidence': face_confidence,
                'voice_confidence': voice_confidence,
                'record_id': record.id,
                'confidence': 'high' if (face_confidence > 0.6 or voice_confidence > 0.6) else 'low'
            })

        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def chat_message(request):
    """Enhanced chat message handler with LangChain integration"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({'error': 'Empty message'}, status=400)
            
            # Get the latest emotion record for context
            latest_emotion = EmotionRecord.objects.order_by('-timestamp').first()
            
            # Extract emotions
            face_emotion = latest_emotion.face_emotion if latest_emotion else 'neutral'
            voice_emotion = latest_emotion.voice_emotion if latest_emotion else 'neutral'
            
            # Clean emotion strings (remove error messages for chatbot)
            if face_emotion and ('Error' in face_emotion or 'No face' in face_emotion):
                face_emotion = 'neutral'
            if voice_emotion and ('Error' in voice_emotion or 'No voice' in voice_emotion):
                voice_emotion = 'neutral'
            
            # Save user message
            user_chat_message = ChatMessage.objects.create(
                sender='user',
                message=user_message,
                emotion_context=latest_emotion
            )
            
            # Get chatbot response using enhanced Ollama integration
            try:
                chatbot = get_enhanced_chatbot(model_name="gemma:2b")  # Using your installed gemma:2b model
                bot_response = chatbot.get_response(
                    message=user_message,
                    face_emotion=face_emotion,
                    voice_emotion=voice_emotion,
                    face_confidence=getattr(latest_emotion, 'face_confidence', 0.0),
                    voice_confidence=getattr(latest_emotion, 'voice_confidence', 0.0),
                    speech_text=getattr(latest_emotion, 'notes', '').split('Speech: ')[-1] if latest_emotion and 'Speech: ' in getattr(latest_emotion, 'notes', '') else None
                )
            except Exception as e:
                print(f"Chatbot error: {e}")
                bot_response = "I'm having trouble connecting to the local AI. Please make sure Ollama is running."
            
            # Save bot response
            bot_chat_message = ChatMessage.objects.create(
                sender='bot',
                message=bot_response,
                emotion_context=latest_emotion
            )
            
            return JsonResponse({
                'success': True,
                'response': bot_response,
                'emotion_context': {
                    'face_emotion': face_emotion,
                    'voice_emotion': voice_emotion
                },
                'message_id': bot_chat_message.id
            })
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': 'Sorry, I had trouble processing your message. Please try again.',
                'response': 'I apologize, but I encountered an issue. Could you please try rephrasing your message?'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def update_preferences(request):
    """Update user accessibility preferences"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Get or create preferences for the current user (or default user if anonymous)
            if request.user.is_authenticated:
                preferences, created = UserPreference.objects.get_or_create(user=request.user)
            else:
                # For anonymous users, try to get the first superuser
                default_user = CustomUser.objects.filter(is_superuser=True).first()
                if default_user:
                    preferences, created = UserPreference.objects.get_or_create(user=default_user)
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'No user available for preferences'
                    }, status=400)
            
            # Update preferences
            if 'theme' in data:
                preferences.theme = data['theme']
            if 'font_size' in data:
                preferences.font_size = data['font_size']
            if 'reduce_animations' in data:
                preferences.reduce_animations = data['reduce_animations']
            if 'high_contrast_mode' in data:
                preferences.high_contrast_mode = data['high_contrast_mode']
            if 'text_to_speech' in data:
                preferences.text_to_speech = data['text_to_speech']
                
            preferences.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Preferences updated successfully'
            })
            
        except Exception as e:
            logger.error(f"Preferences update error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def emotion_history(request):
    """Get recent emotion detection history"""
    return get_emotion_history(request)

@csrf_exempt
def get_emotion_history(request):
    """Get recent emotion detection history"""
    if request.method == 'GET':
        try:
            # Get last 10 emotion records
            records = EmotionRecord.objects.order_by('-timestamp')[:10]
            
            history = []
            for record in records:
                history.append({
                    'id': record.id,
                    'timestamp': record.timestamp.isoformat(),
                    'face_emotion': record.face_emotion,
                    'voice_emotion': record.voice_emotion,
                    'notes': record.notes
                })
            
            return JsonResponse({
                'success': True,
                'emotions': history
            })
            
        except Exception as e:
            logger.error(f"History retrieval error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def clear_chat(request):
    """Clear chat message history"""
    return clear_chat_history(request)

@login_required
def profile_view(request):
    """User profile view and edit"""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=request.user)
    
    return render(request, 'profile/profile.html', {
        'form': form,
        'user': request.user
    })

@csrf_exempt
def forgot_password_view(request):
    """Handle forgot password request"""
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            try:
                user = CustomUser.objects.get(email=email)
                
                # Generate 6-digit OTP
                otp_code = str(random.randint(100000, 999999))
                
                # Create OTP record
                otp_record = PasswordResetOTP.objects.create(
                    user=user,
                    otp_code=otp_code,
                    expires_at=timezone.now() + timedelta(minutes=10)
                )
                
                # Send OTP email (you'll need to configure email settings)
                try:
                    send_mail(
                        'Password Reset OTP - DailyNest',
                        f'Your password reset OTP is: {otp_code}\n\nThis OTP will expire in 10 minutes.',
                        settings.DEFAULT_FROM_EMAIL,
                        [email],
                        fail_silently=False,
                    )
                    messages.success(request, f'OTP sent to {email}. Please check your email.')
                    return redirect('verify_otp', user_id=user.id)
                except Exception as e:
                    messages.error(request, f'Error sending email: {str(e)}')
                    # For development, show OTP in message
                    messages.info(request, f'Development mode - OTP: {otp_code}')
                    return redirect('verify_otp', user_id=user.id)
                    
            except CustomUser.DoesNotExist:
                messages.error(request, 'No account found with this email address.')
    else:
        form = ForgotPasswordForm()
    
    return render(request, 'auth/forgot_password.html', {'form': form})

@csrf_exempt
def verify_otp_view(request, user_id):
    """Handle OTP verification"""
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        messages.error(request, 'Invalid request.')
        return redirect('login')
    
    if request.method == 'POST':
        form = OTPVerificationForm(request.POST)
        if form.is_valid():
            otp_code = form.cleaned_data['otp_code']
            
            # Find valid OTP
            otp_record = PasswordResetOTP.objects.filter(
                user=user,
                otp_code=otp_code,
                is_used=False
            ).first()
            
            if otp_record and not otp_record.is_expired():
                # Mark OTP as used
                otp_record.is_used = True
                otp_record.save()
                
                messages.success(request, 'OTP verified successfully. Please set your new password.')
                return redirect('reset_password', user_id=user.id)
            else:
                messages.error(request, 'Invalid or expired OTP. Please try again.')
    else:
        form = OTPVerificationForm()
    
    return render(request, 'auth/verify_otp.html', {
        'form': form,
        'user': user
    })

@csrf_exempt
def reset_password_view(request, user_id):
    """Handle password reset"""
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        messages.error(request, 'Invalid request.')
        return redirect('login')
    
    # Check if user has a valid used OTP (security check)
    recent_otp = PasswordResetOTP.objects.filter(
        user=user,
        is_used=True,
        created_at__gte=timezone.now() - timedelta(minutes=15)
    ).first()
    
    if not recent_otp:
        messages.error(request, 'Invalid request. Please start the password reset process again.')
        return redirect('forgot_password')
    
    if request.method == 'POST':
        form = ResetPasswordForm(request.POST)
        if form.is_valid():
            new_password = form.cleaned_data['new_password']
            
            # Update user password
            user.set_password(new_password)
            user.save()
            
            # Delete all OTP records for this user
            PasswordResetOTP.objects.filter(user=user).delete()
            
            messages.success(request, 'Password reset successfully. Please login with your new password.')
            return redirect('login')
    else:
        form = ResetPasswordForm()
    
    return render(request, 'auth/reset_password.html', {
        'form': form,
        'user': user
    })

@login_required
def schedule_note_view(request):
    """Handle scheduling notes for autistic persons"""
    if request.user.role != 'caregiver':
        messages.error(request, 'Only caregivers can schedule notes.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = ScheduledNoteForm(request.POST)
        if form.is_valid():
            scheduled_note = form.save(commit=False)
            scheduled_note.caregiver = request.user
            
            # Get autistic person from form or session
            autistic_person_id = request.POST.get('autistic_person_id')
            if autistic_person_id:
                try:
                    autistic_person = CustomUser.objects.get(
                        id=autistic_person_id,
                        role='autistic_person'
                    )
                    scheduled_note.autistic_person = autistic_person
                    scheduled_note.next_run_time = scheduled_note.scheduled_time
                    scheduled_note.save()
                    
                    messages.success(request, f'Note scheduled successfully for {autistic_person.name}.')
                    return redirect('caregiver_dashboard')
                except CustomUser.DoesNotExist:
                    messages.error(request, 'Invalid autistic person selected.')
            else:
                messages.error(request, 'Please select an autistic person.')
    else:
        form = ScheduledNoteForm()
    
    # Get autistic persons under this caregiver's care
    from .models import CareRelationship
    care_relationships = CareRelationship.objects.filter(caregiver=request.user)
    autistic_persons = [rel.autistic_person for rel in care_relationships]
    
    return render(request, 'notes/schedule_note.html', {
        'form': form,
        'autistic_persons': autistic_persons
    })

@login_required
def scheduled_notes_list_view(request):
    """List all scheduled notes for caregiver"""
    if request.user.role != 'caregiver':
        messages.error(request, 'Only caregivers can view scheduled notes.')
        return redirect('dashboard')
    
    scheduled_notes = ScheduledNote.objects.filter(
        caregiver=request.user,
        is_active=True
    ).order_by('next_run_time')
    
    return render(request, 'notes/scheduled_notes_list.html', {
        'scheduled_notes': scheduled_notes
    })

@csrf_exempt
def clear_chat_history(request):
    """Clear chat message history"""
    if request.method == 'POST':
        try:
            ChatMessage.objects.all().delete()
            return JsonResponse({
                'success': True,
                'message': 'Chat history cleared'
            })
        except Exception as e:
            logger.error(f"Clear history error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)