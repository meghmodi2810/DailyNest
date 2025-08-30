from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.contrib.auth.hashers import make_password

class CustomUser(AbstractUser):
    """Custom User model with caregiver mode and email verification support"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('autistic_person', 'Autistic Person'),
    ]
    
    name = models.CharField(max_length=150, help_text="Full name of the user")
    email = models.EmailField(unique=True, help_text="Email address for login")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='autistic_person')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Email verification fields
    is_email_verified = models.BooleanField(
        default=False,
        help_text="Whether the email has been verified"
    )
    email_verification_token = models.UUIDField(
        default=None,
        null=True,
        blank=True,
        editable=False,
        help_text="Token for email verification"
    )
    email_verification_sent_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the verification email was sent"
    )
    
    # Caregiver mode fields
    caregiver_pin = models.CharField(max_length=128, blank=True, null=True, help_text="6-digit PIN for caregiver mode access")
    caregiver_mode_enabled = models.BooleanField(default=False, help_text="Whether caregiver mode is set up")
    
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
    
    def set_caregiver_pin(self, pin):
        """Set the caregiver mode PIN"""
        self.caregiver_pin = make_password(pin)
        self.caregiver_mode_enabled = True
        self.save()
    
    def check_caregiver_pin(self, pin):
        """Check if the provided PIN matches the caregiver PIN"""
        from django.contrib.auth.hashers import check_password
        return check_password(pin, self.caregiver_pin)
    
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
    
    EMOTION_CHECK_INTERVALS = [
        (0, 'Disabled'),
        (3, 'Every 3 hours'),
        (6, 'Every 6 hours'),
        (12, 'Every 12 hours'),
        (24, 'Daily'),
        (-1, 'Morning only (8-10 AM)'),
        (-2, 'Every login'),
    ]
    
    user = models.OneToOneField('CustomUser', on_delete=models.CASCADE, related_name='preferences')
    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default='light')
    font_size = models.CharField(max_length=10, choices=FONT_SIZE_CHOICES, default='medium')
    reduce_animations = models.BooleanField(default=True)
    high_contrast_mode = models.BooleanField(default=False)
    text_to_speech = models.BooleanField(default=False)
    emotion_check_interval = models.IntegerField(
        choices=EMOTION_CHECK_INTERVALS,
        default=24,
        help_text="How often to prompt for emotion check"
    )
    skip_emotion_checks = models.BooleanField(
        default=False,
        help_text="Skip automatic emotion checks"
    )
    last_emotion_check = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last time emotion was checked"
    )
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

class ScheduledNote(models.Model):
    """Model for scheduled notes assigned to autistic individuals"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('missed', 'Missed'),
        ('cancelled', 'Cancelled'),
    ]

    REMINDER_CHOICES = [
        (15, '15 minutes before'),
        (30, '30 minutes before'),
        (60, '1 hour before'),
        (120, '2 hours before'),
        (1440, '1 day before'),
    ]

    caregiver = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='notes_scheduled',
        limit_choices_to={'role': 'caregiver'}
    )
    autistic_person = models.ForeignKey(
        'CustomUser',
        on_delete=models.CASCADE,
        related_name='scheduled_notes_received',
        limit_choices_to={'role': 'autistic_person'}
    )
    title = models.CharField(max_length=200)
    content = models.TextField()
    scheduled_time = models.DateTimeField()
    reminder_time = models.IntegerField(choices=REMINDER_CHOICES, default=30)
    priority = models.CharField(
        max_length=10,
        choices=[
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High'),
        ],
        default='medium'
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    is_recurring = models.BooleanField(default=False)
    recurrence_pattern = models.CharField(
        max_length=20,
        choices=[
            ('daily', 'Daily'),
            ('weekly', 'Weekly'),
            ('monthly', 'Monthly'),
        ],
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['scheduled_time', '-priority']
        verbose_name = 'Scheduled Note'
        verbose_name_plural = 'Scheduled Notes'

    def __str__(self):
        return f"{self.title} - {self.autistic_person.name} ({self.get_status_display()})"

    def mark_completed(self, completion_notes=None):
        """Mark the note as completed with optional notes"""
        self.status = 'completed'
        self.completed_at = timezone.now()
        if completion_notes:
            self.notes = completion_notes
        self.save()

        # If recurring, create next occurrence
        if self.is_recurring:
            self.create_next_occurrence()

    def mark_missed(self):
        """Mark the note as missed if not completed by scheduled time"""
        self.status = 'missed'
        self.save()

        # If recurring, create next occurrence
        if self.is_recurring:
            self.create_next_occurrence()

    def create_next_occurrence(self):
        """Create next occurrence for recurring notes"""
        if not self.is_recurring:
            return

        next_time = self.scheduled_time
        if self.recurrence_pattern == 'daily':
            next_time += timedelta(days=1)
        elif self.recurrence_pattern == 'weekly':
            next_time += timedelta(weeks=1)
        elif self.recurrence_pattern == 'monthly':
            # Add one month (approximately)
            next_time += timedelta(days=30)

        ScheduledNote.objects.create(
            caregiver=self.caregiver,
            autistic_person=self.autistic_person,
            title=self.title,
            content=self.content,
            scheduled_time=next_time,
            reminder_time=self.reminder_time,
            priority=self.priority,
            is_recurring=True,
            recurrence_pattern=self.recurrence_pattern
        )

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

class JournalEntry(models.Model):
    """Model for daily voice journaling by autistic users"""
    MOOD_CHOICES = [
        (1, 'Very Sad'),
        (2, 'Sad'),
        (3, 'Neutral'),
        (4, 'Happy'),
        (5, 'Very Happy'),
    ]
    
    user = models.ForeignKey(
        CustomUser, 
        on_delete=models.CASCADE, 
        related_name='journal_entries',
        limit_choices_to={'role': 'autistic_person'}
    )
    title = models.CharField(max_length=200, null=True, blank=True, help_text="Optional title for the journal entry")
    content = models.TextField(null=True, blank=True, help_text="Transcribed content from voice recording")
    audio_file = models.FileField(upload_to='journal_audio/', null=True, blank=True, help_text="Original voice recording")
    transcription_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score from Whisper transcription")
    mood_rating = models.IntegerField(choices=MOOD_CHOICES, null=True, blank=True, help_text="Self-reported mood rating")
    word_count = models.IntegerField(default=0, help_text="Number of words in the entry")
    is_private = models.BooleanField(default=False, help_text="Hide from caregivers if true")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Journal Entry'
        verbose_name_plural = 'Journal Entries'
        
    def __str__(self):
        return f"{self.user.name}'s Journal - {self.created_at.strftime('%Y-%m-%d')}"
    
    def save(self, *args, **kwargs):
        # Auto-calculate word count
        if self.content:
            self.word_count = len(self.content.split())
        super().save(*args, **kwargs)
    
    @property
    def date_created(self):
        return self.created_at.date()
    
    @property
    def duration_minutes(self):
        """Estimate reading/speaking duration in minutes"""
        return max(1, self.word_count // 150)  # Average speaking rate

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

class DailyPlannerActivity(models.Model):
    """
    Daily Planning System for structuring the user's day.
    Enables caregivers to create and manage detailed schedules for autistic individuals.
    """
    ACTIVITY_TYPE_CHOICES = [
        ('routine', 'Routine'),
        ('therapy', 'Therapy'),
        ('learning', 'Learning'),
        ('play', 'Play'),
        ('meal', 'Meal'),
        ('rest', 'Rest'),
        ('social', 'Social'),
        ('other', 'Other'),
    ]
    
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('missed', 'Missed'),
    ]
    
    # Core fields
    title = models.CharField(max_length=200, help_text="Activity title")
    description = models.TextField(blank=True, help_text="Detailed activity description")
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPE_CHOICES, default='routine')
    
    # Scheduling fields
    scheduled_date = models.DateField(help_text="Date when activity is scheduled")
    scheduled_time = models.TimeField(help_text="Time when activity starts")
    duration_minutes = models.PositiveIntegerField(default=30, help_text="Duration in minutes")
    
    # Priority and status
    priority = models.CharField(max_length=10, choices=PRIORITY_CHOICES, default='medium')
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='scheduled')
    
    # Relationships
    assigned_to = models.ForeignKey('CustomUser', on_delete=models.CASCADE, related_name='daily_activities', help_text="User this activity is assigned to")
    created_by = models.ForeignKey('CustomUser', on_delete=models.CASCADE, related_name='created_activities', help_text="Caregiver who created this activity")
    
    # Reminder settings
    reminder_enabled = models.BooleanField(default=True, help_text="Whether to show reminders")
    reminder_minutes_before = models.PositiveIntegerField(default=15, help_text="Minutes before activity to show reminder")
    
    # Tracking fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True, help_text="When activity was marked as completed")
    
    # Notes and feedback
    completion_notes = models.TextField(blank=True, help_text="Notes added when activity is completed")
    caregiver_notes = models.TextField(blank=True, help_text="Private notes for caregivers")
    
    class Meta:
        ordering = ['scheduled_date', 'scheduled_time']
        indexes = [
            models.Index(fields=['scheduled_date', 'scheduled_time']),
            models.Index(fields=['assigned_to', 'status']),
            models.Index(fields=['activity_type']),
            models.Index(fields=['priority']),
        ]
        verbose_name = "Daily Planner Activity"
        verbose_name_plural = "Daily Planner Activities"
    
    def __str__(self):
        return f"{self.title} - {self.assigned_to.name} ({self.scheduled_date} {self.scheduled_time})"
    
    @property
    def is_upcoming(self):
        """Check if activity is upcoming (scheduled for future)"""
        from django.utils import timezone
        now = timezone.now()
        scheduled_datetime = timezone.datetime.combine(self.scheduled_date, self.scheduled_time)
        if timezone.is_naive(scheduled_datetime):
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
        return scheduled_datetime > now and self.status == 'scheduled'
    
    @property
    def is_due_soon(self):
        """Check if activity is due within reminder time"""
        from django.utils import timezone
        now = timezone.now()
        scheduled_datetime = timezone.datetime.combine(self.scheduled_date, self.scheduled_time)
        if timezone.is_naive(scheduled_datetime):
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
        reminder_time = scheduled_datetime - timezone.timedelta(minutes=self.reminder_minutes_before)
        return now >= reminder_time and scheduled_datetime > now and self.status == 'scheduled'
    
    @property
    def end_time(self):
        """Calculate end time based on start time and duration"""
        from datetime import datetime, timedelta
        start_datetime = datetime.combine(self.scheduled_date, self.scheduled_time)
        end_datetime = start_datetime + timedelta(minutes=self.duration_minutes)
        return end_datetime.time()
    
    def mark_completed(self, notes=""):
        """Mark activity as completed"""
        from django.utils import timezone
        self.status = 'completed'
        self.completed_at = timezone.now()
        if notes:
            self.completion_notes = notes
        self.save()
    
    def get_activity_type_icon(self):
        """Get FontAwesome icon for activity type"""
        icons = {
            'routine': 'fas fa-clock',
            'therapy': 'fas fa-heart',
            'learning': 'fas fa-book',
            'play': 'fas fa-gamepad',
            'meal': 'fas fa-utensils',
            'rest': 'fas fa-bed',
            'social': 'fas fa-users',
            'other': 'fas fa-star',
        }
        return icons.get(self.activity_type, 'fas fa-star')
    
    def get_priority_color(self):
        """Get color class for priority"""
        colors = {
            'low': 'success',
            'medium': 'warning', 
            'high': 'danger',
        }
        return colors.get(self.priority, 'secondary')
