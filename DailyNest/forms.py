from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate
from .models import CustomUser, CareNote, PasswordResetOTP, ScheduledNote
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
    role = forms.ChoiceField(
        choices=[
            ('caregiver', 'Caregiver'),
            ('autistic_person', 'Autistic Person'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-control',
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
        })
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
        fields = ('name', 'email', 'role', 'password1', 'password2')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.name = self.cleaned_data['name']
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['email']  # Use email as username
        user.role = self.cleaned_data['role']
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
    """Form for creating scheduled notes"""
    
    class Meta:
        model = ScheduledNote
        fields = ['title', 'content', 'frequency', 'scheduled_time']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter note title...'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter note content...'
            }),
            'frequency': forms.Select(attrs={
                'class': 'form-control'
            }),
            'scheduled_time': forms.DateTimeInput(attrs={
                'class': 'form-control',
                'type': 'datetime-local'
            }),
        }
