# Generated manually for JournalEntry model

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('DailyNest', '0011_alter_userpreference_emotion_check_interval_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='JournalEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(blank=True, max_length=200, null=True)),
                ('content', models.TextField(blank=True, null=True)),
                ('audio_file', models.FileField(blank=True, null=True, upload_to='journal_audio/')),
                ('transcription_confidence', models.FloatField(blank=True, null=True)),
                ('mood_rating', models.IntegerField(blank=True, choices=[(1, 'Very Sad'), (2, 'Sad'), (3, 'Neutral'), (4, 'Happy'), (5, 'Very Happy')], null=True)),
                ('is_private', models.BooleanField(default=False)),
                ('word_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='journal_entries', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'Journal Entry',
                'verbose_name_plural': 'Journal Entries',
                'ordering': ['-created_at'],
            },
        ),
    ]
