# Generated manually for improved emotion detection models

from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('DailyNest', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='emotionrecord',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='emotionrecord',
            name='face_confidence',
            field=models.FloatField(default=0.0, help_text='Confidence score for face emotion (0-1)'),
        ),
        migrations.AddField(
            model_name='emotionrecord',
            name='voice_confidence',
            field=models.FloatField(default=0.0, help_text='Confidence score for voice emotion (0-1)'),
        ),
        migrations.AddField(
            model_name='userpreference',
            name='user',
            field=models.OneToOneField(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='preferences', to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='userpreference',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='userpreference',
            name='updated_at',
            field=models.DateTimeField(auto_now=True, null=True),
        ),
        migrations.AlterField(
            model_name='emotionrecord',
            name='face_emotion',
            field=models.CharField(blank=True, choices=[('happy', 'Happy'), ('sad', 'Sad'), ('angry', 'Angry'), ('surprised', 'Surprised'), ('neutral', 'Neutral'), ('calm', 'Calm'), ('excited', 'Excited'), ('fear', 'Fear'), ('disgust', 'Disgust')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='emotionrecord',
            name='voice_emotion',
            field=models.CharField(blank=True, choices=[('happy', 'Happy'), ('sad', 'Sad'), ('angry', 'Angry'), ('surprised', 'Surprised'), ('neutral', 'Neutral'), ('calm', 'Calm'), ('excited', 'Excited'), ('fear', 'Fear'), ('disgust', 'Disgust')], max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='emotionrecord',
            name='timestamp',
            field=models.DateTimeField(db_index=True, default=django.utils.timezone.now),
        ),
        migrations.AddIndex(
            model_name='emotionrecord',
            index=models.Index(fields=['timestamp'], name='DailyNest_e_timesta_8b5a7e_idx'),
        ),
        migrations.AddIndex(
            model_name='emotionrecord',
            index=models.Index(fields=['face_emotion'], name='DailyNest_e_face_em_a8c9d1_idx'),
        ),
        migrations.AddIndex(
            model_name='emotionrecord',
            index=models.Index(fields=['voice_emotion'], name='DailyNest_e_voice_e_b7f2e3_idx'),
        ),
        migrations.AlterModelOptions(
            name='emotionrecord',
            options={'ordering': ['-timestamp']},
        ),
        migrations.AlterModelOptions(
            name='userpreference',
            options={'verbose_name': 'User Preference', 'verbose_name_plural': 'User Preferences'},
        ),
    ]
