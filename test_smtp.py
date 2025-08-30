import os
import sys
import django
from django.core.mail import send_mail
from django.conf import settings

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

def test_smtp_connection():
    """Test SMTP connection and email sending"""
    try:
        # Test email sending
        send_mail(
            'SMTP Test - DailyNest',
            'This is a test email from DailyNest SMTP configuration.',
            settings.DEFAULT_FROM_EMAIL,
            [settings.DEFAULT_FROM_EMAIL],  # Send to yourself for testing
            fail_silently=False,
        )
        print("✅ SMTP test email sent successfully!")
        print(f"From: {settings.DEFAULT_FROM_EMAIL}")
        print(f"Host: {settings.EMAIL_HOST}:{settings.EMAIL_PORT}")
        print("TLS:", "Enabled" if settings.EMAIL_USE_TLS else "Disabled")
        return True
    except Exception as e:
        print("❌ Failed to send test email:")
        print(f"Error: {str(e)}")
        print(f"From: {settings.DEFAULT_FROM_EMAIL}")
        print(f"Host: {settings.EMAIL_HOST}:{settings.EMAIL_PORT}")
        print("TLS:", "Enabled" if settings.EMAIL_USE_TLS else "Disabled")
        return False

if __name__ == "__main__":
    print("Testing SMTP configuration...")
    test_smtp_connection()
