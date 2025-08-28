# DailyNest – Setup & Collaboration Guide

DailyNest is a Django-based, emotion-aware companion with Face/Voice emotion detection, AI chat, and a Games dashboard tailored for autistic users.

## Prerequisites
- Windows 10/11 or macOS/Linux
- Python 3.9–3.11 (project tested on Windows 10 + Python 3.9)
- Git
- Optional (for chatbot models): Ollama installed and running

## 1) Clone and Environment
```bash
# Clone
git clone <your-repo-url>
cd DailyNest

# Create venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# python3 -m venv .venv
# source .venv/bin/activate
```

## 2) Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you change dependencies later, re-freeze:
```bash
pip freeze > requirements.txt
```

## 3) Project Setup
```bash
# Apply database migrations
python manage.py migrate

# (Optional) Create superuser to access Django admin
python manage.py createsuperuser
```

## 4) Run the App
```bash
python manage.py runserver
```
Open `http://127.0.0.1:8000` in your browser.

## 5) Role-based Dashboards
- Autistic user dashboard: `/autistic-dashboard/`
- Caregiver dashboard: `/caregiver-dashboard/`
- Admin dashboard: `/admin-dashboard/`

## 6) Key Features
- Emotion Detection: `/emotion/`
- AI Chat: `/chat/`
- Games Dashboard: `/games/` (Games button on autistic dashboard links here)

Games progress is stored per user using `GameProgress` and `GameSession`.

## 7) Adding New Games
1. Create a template in `DailyNest/templates/games/` (e.g., `my_game.html`).
2. Add a `game_type` to `GameProgress.GAME_CHOICES` in `DailyNest/models.py`.
3. Map the template in `play_game` (in `DailyNest/views.py`) or rely on `games/default_game.html`.
4. If models changed: `python manage.py makemigrations && python manage.py migrate`.
5. Visit `/games/<game_type>/` and ensure results POST to `/games/save-result/`.

## 8) Collaboration Workflow
- Create a feature branch: `git checkout -b feature/<name>`
- Keep UI consistent with variables in `templates/base.html`
- Include migrations with model changes
- Update `README.md` if adding features/deps
- Submit a pull request

## 9) Troubleshooting
- Games button style: it uses the same style as AI Chat (`tool-card secondary`). If it appears plain, hard-refresh the page (Ctrl+F5).
- If chatbot returns canned responses, ensure Ollama is running and models are available.
- If ML models fail to load, verify files under `models/` and reinstall requirements.

## 10) Environment Variables (optional)
```powershell
# Windows PowerShell examples
# Disable TensorFlow OneDNN optimizations (sometimes helps stability)
$env:TF_ENABLE_ONEDNN_OPTS="0"
# Django debug
$env:DJANGO_DEBUG="True"
```

## 11) Directory Overview
```
DailyNest/
├─ config/                     # Django project settings
├─ DailyNest/                  # App (views, models, templates)
│  ├─ templates/
│  │  ├─ dashboards/          # Role dashboards
│  │  └─ games/               # Game templates (bubble_pop, default, etc.)
│  ├─ models.py               # Includes GameProgress, GameSession
│  ├─ views.py                # Includes games views and save endpoint
│  └─ urls.py                 # Routes (emotion, chat, games)
├─ models/                     # ML model files
├─ requirements.txt            # Dependencies (from pip freeze)
└─ manage.py
```

You’re ready to collaborate and extend DailyNest across devices. If you run into issues, check `dailynest.log` and the tests in `tests/`. 