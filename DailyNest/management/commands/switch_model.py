"""
Django management command to switch Ollama models
Usage: python manage.py switch_model --model llama2
"""

from django.core.management.base import BaseCommand
from DailyNest.chatbot_ollama import initialize_ollama_chatbot

class Command(BaseCommand):
    help = 'Switch Ollama model for the chatbot'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            default='gemma:2b',
            help='Ollama model name (gemma:2b, llama2, mistral, etc.)'
        )

    def handle(self, *args, **options):
        model_name = options['model']
        
        try:
            chatbot = initialize_ollama_chatbot(model_name)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully switched to model: {model_name}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Failed to switch model: {e}')
            )
