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
from datetime import timedelta
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
    """Admin dashboard with comprehensive system overview"""
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin role required.')
        return redirect('dashboard')
    
    from .models import CareRelationship
    
    all_users = CustomUser.objects.all().order_by('-created_at')
    admins = all_users.filter(role='admin')
    caregivers = all_users.filter(role='caregiver')
    autistic_persons = all_users.filter(role='autistic_person')
    
    care_relationships = CareRelationship.objects.all().order_by('-created_at')
    
    total_emotions = EmotionRecord.objects.count()
    total_chats = ChatMessage.objects.count()
    
    context = {
        'all_users': all_users,
        'admins': admins,
        'caregivers': caregivers,
        'autistic_persons': autistic_persons,
        'care_relationships': care_relationships,
        'total_emotions': total_emotions,
        'total_chats': total_chats,
        'user_count': all_users.count(),
        'emotion_records_count': total_emotions,
        'chat_messages_count': total_chats,
        'recent_users': all_users[:5],
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
    """Chat page view - user isolated"""
    # Get or create preferences for the current user (or default user if anonymous)
    if request.user.is_authenticated:
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
        # Get recent chat messages for current user only
        recent_messages = ChatMessage.objects.filter(user=request.user).order_by('-timestamp')[:20]
    else:
        # For anonymous users, try to get the first superuser or create default preferences
        default_user = CustomUser.objects.filter(is_superuser=True).first()
        if default_user:
            preferences, created = UserPreference.objects.get_or_create(user=default_user)
        else:
            # Create a minimal context without preferences
            preferences = None
        # Get recent chat messages for anonymous users
        recent_messages = ChatMessage.objects.filter(user__isnull=True).order_by('-timestamp')[:20]
    
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

            # Save emotion record with confidence scores and user association
            record = EmotionRecord.objects.create(
                user=request.user if request.user.is_authenticated else None,
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
            
            # Get the latest emotion record for context (user-specific)
            if request.user.is_authenticated:
                latest_emotion = EmotionRecord.objects.filter(user=request.user).order_by('-timestamp').first()
            else:
                latest_emotion = EmotionRecord.objects.filter(user__isnull=True).order_by('-timestamp').first()
            
            # Extract emotions
            face_emotion = latest_emotion.face_emotion if latest_emotion else 'neutral'
            voice_emotion = latest_emotion.voice_emotion if latest_emotion else 'neutral'
            
            # Clean emotion strings (remove error messages for chatbot)
            if face_emotion and ('Error' in face_emotion or 'No face' in face_emotion):
                face_emotion = 'neutral'
            if voice_emotion and ('Error' in voice_emotion or 'No voice' in voice_emotion):
                voice_emotion = 'neutral'
            
            # Save user message with user association
            user_chat_message = ChatMessage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                sender='user',
                message=user_message,
                emotion_context=latest_emotion,
                is_bot=False
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
            
            # Save bot response with user association
            bot_chat_message = ChatMessage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                sender='bot',
                message=bot_response,
                emotion_context=latest_emotion,
                is_bot=True
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

def get_emotion_history(request):
    """Get recent emotion detection history"""
    if request.method == 'GET':
        try:
            # Get last 10 emotion records for current user
            if request.user.is_authenticated:
                records = EmotionRecord.objects.filter(user=request.user).order_by('-timestamp')[:10]
            else:
                records = EmotionRecord.objects.filter(user__isnull=True).order_by('-timestamp')[:10]
            
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
    """Clear chat message history - user isolated"""
    if request.method == 'POST':
        try:
            if request.user.is_authenticated:
                ChatMessage.objects.filter(user=request.user).delete()
            else:
                ChatMessage.objects.filter(user__isnull=True).delete()
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

# Caregiver-Autistic Relationship Management Views
@login_required
def manage_care_relationships(request):
    """Manage caregiver-autistic relationships"""
    if request.user.role != 'caregiver':
        messages.error(request, 'Only caregivers can manage care relationships.')
        return redirect('dashboard')
    
    from .models import CareRelationship
    from .forms import CareRelationshipForm
    
    if request.method == 'POST':
        form = CareRelationshipForm(request.POST)
        if form.is_valid():
            relationship = form.save(commit=False)
            relationship.caregiver = request.user
            
            # Check if relationship already exists
            existing = CareRelationship.objects.filter(
                caregiver=request.user,
                autistic_person=relationship.autistic_person
            ).first()
            
            if existing:
                messages.error(request, 'You already have a relationship with this person.')
            else:
                relationship.save()
                messages.success(request, f'Care relationship with {relationship.autistic_person.name} created successfully.')
                return redirect('manage_care_relationships')
    else:
        form = CareRelationshipForm()
    
    # Get current relationships
    relationships = CareRelationship.objects.filter(
        caregiver=request.user,
        is_active=True
    ).select_related('autistic_person')
    
    return render(request, 'care/manage_relationships.html', {
        'form': form,
        'relationships': relationships
    })

@login_required
def remove_care_relationship(request, relationship_id):
    """Remove a care relationship"""
    if request.user.role != 'caregiver':
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    
    from .models import CareRelationship
    
    try:
        relationship = CareRelationship.objects.get(
            id=relationship_id,
            caregiver=request.user
        )
        relationship.is_active = False
        relationship.save()
        messages.success(request, f'Relationship with {relationship.autistic_person.name} removed.')
    except CareRelationship.DoesNotExist:
        messages.error(request, 'Relationship not found.')
    
    return redirect('manage_care_relationships')

# Admin User Management Views
@login_required
def admin_manage_users(request):
    """Admin user management"""
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin role required.')
        return redirect('dashboard')
    
    from .forms import AdminUserForm
    
    if request.method == 'POST':
        form = AdminUserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password('DailyNest2024!')  # Default password
            user.save()
            messages.success(request, f'User {user.name} created successfully. Default password: DailyNest2024!')
            return redirect('admin_manage_users')
    else:
        form = AdminUserForm()
    
    users = CustomUser.objects.all().order_by('-created_at')
    
    return render(request, 'admin/manage_users.html', {
        'form': form,
        'users': users
    })

@login_required
def admin_assign_caregiver(request, autistic_id):
    """Admin assign caregiver to autistic person"""
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin role required.')
        return redirect('dashboard')
    
    from .models import CareRelationship
    
    try:
        autistic_person = CustomUser.objects.get(id=autistic_id, role='autistic_person')
    except CustomUser.DoesNotExist:
        messages.error(request, 'Autistic person not found.')
        return redirect('admin_dashboard')
    
    if request.method == 'POST':
        caregiver_id = request.POST.get('caregiver_id')
        relationship_type = request.POST.get('relationship_type', 'other')
        
        try:
            caregiver = CustomUser.objects.get(id=caregiver_id, role='caregiver')
            
            # Check if relationship exists
            existing = CareRelationship.objects.filter(
                caregiver=caregiver,
                autistic_person=autistic_person
            ).first()
            
            if existing:
                existing.is_active = True
                existing.relationship_type = relationship_type
                existing.save()
                messages.success(request, f'Relationship between {caregiver.name} and {autistic_person.name} updated.')
            else:
                CareRelationship.objects.create(
                    caregiver=caregiver,
                    autistic_person=autistic_person,
                    relationship_type=relationship_type
                )
                messages.success(request, f'Caregiver {caregiver.name} assigned to {autistic_person.name}.')
        except CustomUser.DoesNotExist:
            messages.error(request, 'Caregiver not found.')
        
        return redirect('admin_dashboard')
    
    caregivers = CustomUser.objects.filter(role='caregiver')
    current_relationships = CareRelationship.objects.filter(
        autistic_person=autistic_person,
        is_active=True
    ).select_related('caregiver')
    
    return render(request, 'admin/assign_caregiver.html', {
        'autistic_person': autistic_person,
        'caregivers': caregivers,
        'current_relationships': current_relationships
    })

@login_required
def admin_edit_user(request, user_id):
    """Admin edit user"""
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin role required.')
        return redirect('dashboard')
    
    try:
        user = CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('admin_manage_users')
    
    from .forms import AdminUserForm
    
    if request.method == 'POST':
        form = AdminUserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, f'User {user.name} updated successfully.')
            return redirect('admin_manage_users')
    else:
        form = AdminUserForm(instance=user)
    
    return render(request, 'admin/edit_user.html', {
        'form': form,
        'user': user
    })

@login_required
def admin_delete_user(request, user_id):
    """Admin delete user"""
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin role required.')
        return redirect('dashboard')
    
    try:
        user = CustomUser.objects.get(id=user_id)
        if user == request.user:
            messages.error(request, 'You cannot delete your own account.')
        else:
            user_name = user.name
            user.delete()
            messages.success(request, f'User {user_name} deleted successfully.')
    except CustomUser.DoesNotExist:
        messages.error(request, 'User not found.')
    
    return redirect('admin_manage_users')

# Game Views
@login_required
def games_dashboard(request):
    """Games dashboard view for autistic users"""
    from .models import GameProgress, GameSession
    
    # Get user's game progress
    user_progress = GameProgress.objects.filter(user=request.user)
    
    # Get recent game sessions
    recent_sessions = GameSession.objects.filter(user=request.user).order_by('-started_at')[:5]
    
    # Calculate total stats
    total_games_played = user_progress.count()
    total_score = sum(progress.score for progress in user_progress)
    total_time = sum(progress.time_spent for progress in user_progress)
    
    context = {
        'user_progress': user_progress,
        'recent_sessions': recent_sessions,
        'total_games_played': total_games_played,
        'total_score': total_score,
        'total_time': total_time,
        'available_games': [
            {
                'id': 'guess_the_bowl',
                'name': 'Guess The Bowl',
                'description': 'Guess which bowl hides the ball after shuffling',
                'icon': 'fas fa-bowl-food',
                'color': 'var(--calm-blue)',
                'difficulty': 'Medium'
            },
            {
                'id': 'bubble_pop',
                'name': 'Bubble Pop',
                'description': 'A calming bubble popping game to reduce stress',
                'icon': 'fas fa-bubbles',
                'color': 'var(--soft-green)',
                'difficulty': 'Easy'
            },
            {
                'id': 'breathing_exercise',
                'name': 'Breathing Exercise',
                'description': 'Guided breathing for relaxation and focus',
                'icon': 'fas fa-wind',
                'color': 'var(--soothing-teal)',
                'difficulty': 'Easy'
            },
            {
                'id': 'colorfill',
                'name': 'Creative Color Fill',
                'description': 'Create beautiful artwork by filling shapes with colors',
                'icon': 'fas fa-palette',
                'color': 'var(--warm-orange)',
                'difficulty': 'Easy'
            },
            {
                'id': 'memory_match',
                'name': 'Happy Memory Match',
                'description': 'Match pairs of happy emojis to improve memory',
                'icon': 'fas fa-brain',
                'color': 'var(--gentle-purple)',
                'difficulty': 'Medium'
            }
        ]
    }
    
    return render(request, 'games/games_dashboard.html', context)

@login_required
def play_game(request, game_type):
    """Play a specific game"""
    from .models import GameSession, GameProgress
    
    # Validate game type
    valid_games = [choice[0] for choice in GameProgress.GAME_CHOICES]
    if game_type not in valid_games:
        messages.error(request, 'Invalid game type.')
        return redirect('games_dashboard')
    
    # Create or get active session
    active_session, created = GameSession.objects.get_or_create(
        user=request.user,
        game_type=game_type,
        ended_at__isnull=True,
        defaults={'started_at': timezone.now()}
    )
    
    # Get game template based on type
    game_templates = {
        'guess_the_bowl': 'games/guess_the_bowl.html',
        'bubble_pop': 'games/bubble_pop.html',
        'breathing_exercise': 'games/breathing_exercise.html',
        'colorfill': 'games/colorfill.html',
        'memory_match': 'games/memory_match.html',
    }
    
    template_name = game_templates.get(game_type, 'games/default_game.html')
    
    context = {
        'game_type': game_type,
        'game_name': dict(GameProgress.GAME_CHOICES)[game_type],
        'session_id': active_session.id,
    }
    
    return render(request, template_name, context)

@csrf_exempt
@require_http_methods(['POST'])
def save_game_result(request):
    """Save game results and update progress"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        score = data.get('score', 0)
        duration = data.get('duration', 0)
        game_type = data.get('game_type')
        session_data = data.get('session_data', {})
        
        if not all([session_id, game_type]):
            return JsonResponse({'error': 'Missing required data'}, status=400)
        
        # Get the session and end it
        try:
            session = GameSession.objects.get(
                id=session_id,
                user=request.user,
                ended_at__isnull=True
            )
            session.end_session(score, duration, session_data)
            
            return JsonResponse({
                'success': True,
                'message': 'Game result saved successfully',
                'new_high_score': session.session_score
            })
            
        except GameSession.DoesNotExist:
            return JsonResponse({'error': 'Session not found'}, status=404)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        logger.error(f"Error saving game result: {str(e)}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

@login_required
def game_progress(request):
    """View user's game progress and statistics"""
    from .models import GameProgress, GameSession
    
    # Get detailed progress for each game
    progress_data = []
    for choice in GameProgress.GAME_CHOICES:
        game_id, game_name = choice
        try:
            progress = GameProgress.objects.get(user=request.user, game_type=game_id)
            progress_data.append({
                'game_id': game_id,
                'game_name': game_name,
                'progress': progress,
                'sessions_count': GameSession.objects.filter(
                    user=request.user, 
                    game_type=game_id
                ).count()
            })
        except GameProgress.DoesNotExist:
            progress_data.append({
                'game_id': game_id,
                'game_name': game_name,
                'progress': None,
                'sessions_count': 0
            })
    
    # Get recent achievements and milestones
    recent_achievements = []
    for progress in GameProgress.objects.filter(user=request.user, score__gt=0):
        if progress.score >= 100:
            recent_achievements.append(f"Scored 100+ in {progress.get_game_type_display()}")
        if progress.time_spent >= 300:  # 5 minutes
            recent_achievements.append(f"Played {progress.get_game_type_display()} for 5+ minutes")
    
    context = {
        'progress_data': progress_data,
        'recent_achievements': recent_achievements[:5],
        'total_games_played': GameProgress.objects.filter(user=request.user, score__gt=0).count(),
        'total_score': sum(p.score for p in GameProgress.objects.filter(user=request.user)),
        'total_time': sum(p.time_spent for p in GameProgress.objects.filter(user=request.user)),
    }
    
    return render(request, 'games/game_progress.html', context)

# New Games Views
@login_required
def games_hub(request):
    """Main games hub page with all therapeutic games"""
    return render(request, 'games/games_hub.html')

@login_required
def calm_maze(request):
    """Calm maze game view"""
    return render(request, 'games/calm_maze.html')

@login_required
def bubble_pop(request):
    """Bubble pop game view"""
    from .models import GameSession
    
    # Create or get active session
    active_session, created = GameSession.objects.get_or_create(
        user=request.user,
        game_type='bubble_pop',
        ended_at__isnull=True,
        defaults={'started_at': timezone.now()}
    )
    
    context = {
        'game_type': 'bubble_pop',
        'game_name': 'Calming Bubble Pop',
        'session_id': active_session.id,
    }
    
    return render(request, 'games/bubble_pop.html', context)

@login_required
def memory_match(request):
    """Memory match game view"""
    return render(request, 'games/memory_match.html')

@login_required
def breathing_garden(request):
    """Breathing garden game view"""
    return render(request, 'games/breathing_garden.html')

@login_required
def guess_the_bowl(request):
    """Guess the bowl game view"""
    return render(request, 'games/guess_the_bowl.html')