from django.db import migrations, models
import uuid
from django.utils import timezone


def create_verification_codes(apps, schema_editor):
    # For existing users, set them as verified
    CustomUser = apps.get_model('DailyNest', 'CustomUser')
    CustomUser.objects.all().update(is_email_verified=True)


class Migration(migrations.Migration):

    dependencies = [
        ('DailyNest', '0017_remove_carenote_is_visible_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='is_email_verified',
            field=models.BooleanField(default=False, help_text='Whether the email has been verified'),
        ),
        migrations.AddField(
            model_name='customuser',
            name='email_verification_token',
            field=models.UUIDField(default=uuid.uuid4, editable=False, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='customuser',
            name='email_verification_sent_at',
            field=models.DateTimeField(null=True, blank=True),
        ),
        migrations.RunPython(create_verification_codes, reverse_code=migrations.RunPython.noop),
    ]
