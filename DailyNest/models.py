from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.contrib.auth.hashers import make_password

class CustomUser(AbstractUser):
    """Custom User model with role-based authentication"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('caregiver', 'Caregiver'),
        ('autistic_person', 'Autistic Person'),
    ]
    
    name = models.CharField(max_length=150, help_text="Full name of the user")
    email = models.EmailField(unique=True, help_text="Email address for login")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='autistic_person')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Additional profile information
    phone = models.CharField(max_length=20, blank=True, null=True, help_text="Phone number")
    date_of_birth = models.DateField(blank=True, null=True, help_text="Date of birth")
    address = models.TextField(blank=True, null=True, help_text="Home address")
    emergency_contact = models.CharField(max_length=150, blank=True, null=True, help_text="Emergency contact name")
    emergency_phone = models.CharField(max_length=20, blank=True, null=True, help_text="Emergency contact phone")
    bio = models.TextField(blank=True, null=True, help_text="Personal bio or notes")
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True, help_text="Profile picture")
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'name']
    
    def save(self, *args, **kwargs):
        # Ensure email is used as username if not provided
        if not self.username:
            self.username = self.email
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.name} ({self.get_role_display()})"

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
    
    user = models.ForeignKey('CustomUser', on_delete=models.CASCADE, null=True, blank=True)
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
    
    user = models.ForeignKey('CustomUser', on_delete=models.CASCADE, null=True, blank=True)
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    emotion_context = models.ForeignKey(EmotionRecord, on_delete=models.SET_NULL, null=True, blank=True)
    is_bot = models.BooleanField(default=False)

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
    
    user = models.OneToOneField('CustomUser', on_delete=models.CASCADE, related_name='preferences')
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

class CareRelationship(models.Model):
    """Model to define caregiver-autistic person relationships"""
    caregiver = models.ForeignKey(
        'CustomUser', 
        on_delete=models.CASCADE, 
        related_name='care_relationships_as_caregiver',
        limit_choices_to={'role': 'caregiver'}
    )
    autistic_person = models.ForeignKey(
        'CustomUser', 
        on_delete=models.CASCADE, 
        related_name='care_relationships_as_autistic',
        limit_choices_to={'role': 'autistic_person'}
    )
    relationship_type = models.CharField(
        max_length=50, 
        choices=[
            ('parent', 'Parent'),
            ('guardian', 'Guardian'),
            ('therapist', 'Therapist'),
            ('teacher', 'Teacher'),
            ('support_worker', 'Support Worker'),
            ('other', 'Other'),
        ],
        default='guardian'
    )
    start_date = models.DateField(default=timezone.now)
    notes = models.TextField(blank=True, null=True, help_text="Additional notes about the care relationship")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['caregiver', 'autistic_person']
        verbose_name = 'Care Relationship'
        verbose_name_plural = 'Care Relationships'
    
    def __str__(self):
        return f"{self.caregiver.name} cares for {self.autistic_person.name} ({self.get_relationship_type_display()})"

class CaregiverInvitation(models.Model):
    """Model for caregiver invitations during autistic person registration"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('declined', 'Declined'),
        ('expired', 'Expired'),
    ]
    
    autistic_person = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='caregiver_invitations',
        limit_choices_to={'role': 'autistic_person'}
    )
    caregiver_email = models.EmailField(help_text="Email of the caregiver to invite")
    caregiver = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='received_invitations',
        null=True,
        blank=True,
        limit_choices_to={'role': 'caregiver'}
    )
    relationship_type = models.CharField(
        max_length=50,
        choices=[
            ('parent', 'Parent'),
            ('guardian', 'Guardian'),
            ('therapist', 'Therapist'),
            ('teacher', 'Teacher'),
            ('support_worker', 'Support Worker'),
            ('other', 'Other'),
        ],
        default='guardian'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    invitation_token = models.CharField(max_length=100, unique=True)
    message = models.TextField(blank=True, null=True, help_text="Optional message from autistic person")
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    responded_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = 'Caregiver Invitation'
        verbose_name_plural = 'Caregiver Invitations'
    
    def __str__(self):
        return f"Invitation for {self.autistic_person.name} to {self.caregiver_email} ({self.status})"

class CareNote(models.Model):
    """Model for caregiver notes about autistic persons"""
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    TYPE_CHOICES = [
        ('general', 'General Note'),
        ('behavior', 'Behavior Observation'),
        ('progress', 'Progress Update'),
        ('concern', 'Concern'),
        ('achievement', 'Achievement'),
        ('medication', 'Medication'),
        ('appointment', 'Appointment'),
        ('emergency', 'Emergency Contact'),
    ]
    
    caregiver = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='care_notes_created',
        limit_choices_to={'role': 'caregiver'}
    )
    autistic_person = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='care_notes_received',
        limit_choices_to={'role': 'autistic_person'}
    )
    title = models.CharField(max_length=200, help_text="Brief title for the note")
    content = models.TextField(help_text="Detailed note content")
    note_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='general')
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    is_private = models.BooleanField(default=False, help_text="Private notes only visible to caregivers")
    tags = models.CharField(max_length=200, blank=True, help_text="Comma-separated tags")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Care Note'
        verbose_name_plural = 'Care Notes'
    
    def __str__(self):
        return f"{self.title} - {self.autistic_person.name} by {self.caregiver.name}"
    
    def get_tags_list(self):
        """Return tags as a list"""
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]

class PasswordResetOTP(models.Model):
    """Model for password reset OTP verification"""
    user = models.ForeignKey('CustomUser', on_delete=models.CASCADE, related_name='password_reset_otps')
    otp_code = models.CharField(max_length=6, help_text="6-digit OTP code")
    is_used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Password Reset OTP'
        verbose_name_plural = 'Password Reset OTPs'
    
    def __str__(self):
        return f"OTP for {self.user.email} - {self.otp_code}"
    
    def is_expired(self):
        """Check if OTP is expired"""
        from django.utils import timezone
        return timezone.now() > self.expires_at

class ScheduledNote(models.Model):
    """Model for scheduled notes for autistic persons"""
    FREQUENCY_CHOICES = [
        ('once', 'One Time'),
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('sent', 'Sent'),
        ('cancelled', 'Cancelled'),
    ]
    
    caregiver = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='scheduled_notes_created',
        limit_choices_to={'role': 'caregiver'}
    )
    autistic_person = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='scheduled_notes_received',
        limit_choices_to={'role': 'autistic_person'}
    )
    title = models.CharField(max_length=200, help_text="Title for the scheduled note")
    content = models.TextField(help_text="Content of the scheduled note")
    frequency = models.CharField(max_length=20, choices=FREQUENCY_CHOICES, default='once')
    scheduled_time = models.DateTimeField(help_text="When to send/create the note")
    next_run_time = models.DateTimeField(help_text="Next time this note should be processed")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['next_run_time']
        verbose_name = 'Scheduled Note'
        verbose_name_plural = 'Scheduled Notes'
    
    def __str__(self):
        return f"Scheduled: {self.title} for {self.autistic_person.name}"

class GameProgress(models.Model):
    """Model to track user progress in games"""
    GAME_CHOICES = [
        ('guess_the_bowl', 'Guess The Bowl'),
        ('bubble_pop', 'Bubble Pop'),
        ('breathing_exercise', 'Breathing Exercise'),
        ('colorfill', 'Creative Color Fill'),
        ('memory_match', 'Happy Memory Match'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='game_progress')
    game_type = models.CharField(max_length=50, choices=GAME_CHOICES)
    score = models.IntegerField(default=0)
    time_spent = models.IntegerField(default=0, help_text="Time spent in seconds")
    completed = models.BooleanField(default=False)
    last_played = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user', 'game_type']
        ordering = ['-last_played']
        indexes = [
            models.Index(fields=['user', 'game_type']),
            models.Index(fields=['last_played']),
        ]
    
    def __str__(self):
        return f"{self.user.name} - {self.get_game_type_display()} (Score: {self.score})"
    
    @property
    def formatted_time(self):
        """Format time spent in MM:SS format"""
        minutes = self.time_spent // 60
        seconds = self.time_spent % 60
        return f"{minutes:02d}:{seconds:02d}"

class GameSession(models.Model):
    """Model to track individual game sessions"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='game_sessions')
    game_type = models.CharField(max_length=50, choices=GameProgress.GAME_CHOICES)
    session_score = models.IntegerField(default=0)
    session_duration = models.IntegerField(default=0, help_text="Session duration in seconds")
    session_data = models.JSONField(default=dict, blank=True, help_text="Additional session data")
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['user', 'game_type']),
            models.Index(fields=['started_at']),
        ]
    
    def __str__(self):
        return f"{self.user.name} - {self.get_game_type_display()} Session ({self.started_at.strftime('%Y-%m-%d %H:%M')})"
    
    def end_session(self, final_score, final_duration, session_data=None):
        """End the game session and update progress"""
        self.session_score = final_score
        self.session_duration = final_duration
        if session_data:
            self.session_data = session_data
        self.ended_at = timezone.now()
        self.save()
        
        # Update or create game progress
        progress, created = GameProgress.objects.get_or_create(
            user=self.user,
            game_type=self.game_type,
            defaults={'score': final_score, 'time_spent': final_duration}
        )
        
        if not created:
            # Update existing progress
            progress.score = max(progress.score, final_score)
            progress.time_spent += final_duration
            progress.last_played = timezone.now()
            progress.save()
