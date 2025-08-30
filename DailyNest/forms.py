from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password, password_validators_help_text_html
from django.core.exceptions import ValidationError
from .models import (CustomUser, CareNote, PasswordResetOTP, ScheduledNote, 
                     CareRelationship, UserPreference)
import uuid
from datetime import datetime, timedelta

class CustomUserRegistrationForm(UserCreationForm):
    """Custom registration form with role selection (no admin option)"""
    name = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name',
            'required': True
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True
        })
    )
    caregiver_email = forms.EmailField(
        required=False,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter caregiver email (optional)',
        }),
        help_text="If you're an autistic person, enter your caregiver's email to send them an invitation"
    )
    relationship_type = forms.ChoiceField(
        choices=[
            ('parent', 'Parent'),
            ('guardian', 'Guardian'),
            ('therapist', 'Therapist'),
            ('teacher', 'Teacher'),
            ('support_worker', 'Support Worker'),
            ('other', 'Other'),
        ],
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control',
        }),
        help_text="Relationship with your caregiver"
    )
    invitation_message = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Optional message to your caregiver...'
        }),
        help_text="Optional message to include with the invitation"
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter password',
            'required': True
        }),
        help_text="Password must be at least 8 characters long and not too common. " + password_validators_help_text_html()
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm password',
            'required': True
        })
    )

    class Meta:
        model = CustomUser
        fields = ('name', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError("A user with this email already exists.")
        return email

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        if password1:
            try:
                # Create a temporary user instance for validation
                user = CustomUser(
                    email=self.cleaned_data.get('email', ''),
                    name=self.cleaned_data.get('name', ''),
                    username=self.cleaned_data.get('email', '')
                )
                validate_password(password1, user)
            except ValidationError as e:
                raise ValidationError(e.messages)
        return password1

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise ValidationError("The two password fields didn't match.")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.name = self.cleaned_data['name']
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['email']  # Use email as username
        user.role = 'autistic_person'  # Always set role to autistic_person
        # Set the password properly using set_password method
        user.set_password(self.cleaned_data['password1'])
        if commit:
            user.save()
        return user

class CustomLoginForm(AuthenticationForm):
    """Custom login form without role selection - auto-detects user role"""
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'required': True
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the default username field
        if 'username' in self.fields:
            del self.fields['username']

    def clean(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')

        if email and password:
            # Authenticate using email as username
            self.user_cache = authenticate(
                self.request,
                username=email,
                password=password
            )
            
            if self.user_cache is None:
                raise forms.ValidationError("Invalid email or password.")
            
            if not self.user_cache.is_active:
                raise forms.ValidationError("This account is inactive.")

        return self.cleaned_data

class UserProfileForm(forms.ModelForm):
    """Form for editing user profile information"""
    class Meta:
        model = CustomUser
        fields = ['name', 'phone', 'date_of_birth', 'address', 'emergency_contact', 
                 'emergency_phone', 'bio', 'profile_picture']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+1234567890'}),
            'date_of_birth': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'emergency_contact': forms.TextInput(attrs={'class': 'form-control'}),
            'emergency_phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '+1234567890'}),
            'bio': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Tell us about yourself...'}),
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'})
        }

class CareNoteForm(forms.ModelForm):
    """Form for creating and editing care notes"""
    class Meta:
        model = CareNote
        fields = ['title', 'content', 'note_type', 'priority', 'is_private', 'tags']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter note title...'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Enter note content...'
            }),
            'note_type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'priority': forms.Select(attrs={
                'class': 'form-control'
            }),
            'is_private': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter tags separated by commas...'
            }),
        }

class ForgotPasswordForm(forms.Form):
    """Form for requesting password reset"""
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True
        }),
        help_text="Enter the email address associated with your account"
    )

class OTPVerificationForm(forms.Form):
    """Form for OTP verification"""
    otp_code = forms.CharField(
        max_length=6,
        min_length=6,
        widget=forms.TextInput(attrs={
            'class': 'form-control text-center',
            'placeholder': '000000',
            'pattern': '[0-9]{6}',
            'maxlength': '6',
            'required': True,
            'style': 'font-size: 1.5rem; letter-spacing: 0.5rem;'
        }),
        help_text="Enter the 6-digit OTP sent to your email"
    )

class ResetPasswordForm(forms.Form):
    """Form for setting new password"""
    new_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter new password',
            'required': True
        }),
        help_text="Enter your new password"
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm new password',
            'required': True
        }),
        help_text="Confirm your new password"
    )
    
    def clean(self):
        cleaned_data = super().clean()
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')
        
        if new_password and confirm_password:
            if new_password != confirm_password:
                raise forms.ValidationError("Passwords do not match")
        
        return cleaned_data

class ScheduledNoteForm(forms.ModelForm):
    schedule_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        help_text='Select the date for this note'
    )
    schedule_time = forms.TimeField(
        widget=forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'}),
        help_text='Select the time for this note'
    )
    recurrence_pattern = forms.ChoiceField(
        choices=[
            ('', 'No recurrence'),
            ('daily', 'Daily'),
            ('weekly', 'Weekly'),
            ('monthly', 'Monthly'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'}),
    )
    is_recurring = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
    )
    reminder_time = forms.IntegerField(
        required=False,
        min_value=5,
        max_value=120,
        initial=30,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Minutes before the scheduled time to send a reminder (5-120 minutes)'
    )

    class Meta:
        model = ScheduledNote
        fields = ['title', 'content', 'priority', 'schedule_date', 'schedule_time', 
                 'is_recurring', 'recurrence_pattern', 'reminder_time']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 4}),
            'priority': forms.Select(attrs={'class': 'form-select'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        schedule_date = cleaned_data.get('schedule_date')
        schedule_time = cleaned_data.get('schedule_time')
        is_recurring = cleaned_data.get('is_recurring')
        recurrence_pattern = cleaned_data.get('recurrence_pattern')

        if schedule_date and schedule_time:
            # Combine date and time for validation
            from django.utils import timezone
            import datetime
            scheduled_datetime = datetime.datetime.combine(
                schedule_date, 
                schedule_time, 
                tzinfo=timezone.get_current_timezone()
            )
            
            # Check if datetime is in the past
            if scheduled_datetime < timezone.now():
                raise forms.ValidationError('Scheduled time must be in the future.')

            # Store the combined datetime
            cleaned_data['scheduled_time'] = scheduled_datetime

        # Validate recurrence pattern if is_recurring is checked
        if is_recurring and not recurrence_pattern:
            raise forms.ValidationError('Please select a recurrence pattern for recurring notes.')

        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.scheduled_time = self.cleaned_data['scheduled_time']
        if commit:
            instance.save()
        return instance

class CareRelationshipForm(forms.ModelForm):
    """Form for creating care relationships"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show autistic persons in the dropdown
        self.fields['autistic_person'].queryset = CustomUser.objects.filter(role='autistic_person')
    
    class Meta:
        model = CareRelationship
        fields = ['autistic_person', 'relationship_type', 'notes']
        widgets = {
            'autistic_person': forms.Select(attrs={
                'class': 'form-control',
                'required': True
            }),
            'relationship_type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'notes': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional notes about the relationship...'
            }),
        }

class AdminUserForm(forms.ModelForm):
    """Form for admin to create/edit users"""
    
    class Meta:
        model = CustomUser
        fields = ['name', 'email', 'role', 'phone', 'address', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter full name',
                'required': True
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter email address',
                'required': True
            }),
            'role': forms.Select(attrs={
                'class': 'form-control',
                'required': True
            }),
            'phone': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '+1234567890'
            }),
            'address': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Enter address...'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.username = user.email  # Use email as username
        if commit:
            user.save()
        return user

class EmailVerificationForm(forms.Form):
    """Form for email verification during registration"""
    verification_code = forms.CharField(
        max_length=6,
        min_length=6,
        widget=forms.TextInput(attrs={
            'class': 'form-control text-center',
            'placeholder': '000000',
            'pattern': '[0-9]{6}',
            'maxlength': '6',
            'required': True,
            'style': 'font-size: 1.5rem; letter-spacing: 0.5rem;'
        }),
        help_text="Enter the 6-digit verification code sent to your email"
    )

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def clean_verification_code(self):
        code = self.cleaned_data.get('verification_code')
        if not self.user or not self.user.email_verification_token:
            raise forms.ValidationError("Invalid verification request.")
        
        # Check if code matches and is not expired (24 hours)
        from django.utils import timezone
        from datetime import timedelta
        
        if str(self.user.email_verification_token)[:6] != code:
            raise forms.ValidationError("Invalid verification code.")
            
        if self.user.email_verification_sent_at < (timezone.now() - timedelta(hours=24)):
            raise forms.ValidationError("Verification code has expired. Please request a new one.")
            
        return code

class UserPreferenceForm(forms.ModelForm):
    """Form for updating user preferences including emotion check settings"""
    class Meta:
        model = UserPreference
        fields = ['theme', 'font_size', 'reduce_animations', 'high_contrast_mode', 
                 'text_to_speech', 'emotion_check_interval', 'skip_emotion_checks']
        widgets = {
            'theme': forms.Select(attrs={'class': 'form-select'}),
            'font_size': forms.Select(attrs={'class': 'form-select'}),
            'reduce_animations': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'high_contrast_mode': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'text_to_speech': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'emotion_check_interval': forms.Select(attrs={'class': 'form-select'}),
            'skip_emotion_checks': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['emotion_check_interval'].help_text = 'How often to prompt for emotion checks'
