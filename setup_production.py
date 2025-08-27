#!/usr/bin/env python
"""
Production setup script for DailyNest.
Handles model validation, database setup, and environment configuration.
"""

import os
import sys
import subprocess
import django
from pathlib import Path

def setup_django():
    """Setup Django environment"""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'django', 'tensorflow', 'opencv-python', 'numpy', 
        'librosa', 'soundfile', 'speech-recognition'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def validate_models():
    """Validate ML model files"""
    print("\n🧠 Validating ML models...")
    
    base_dir = Path(__file__).parent
    model_paths = [
        base_dir / 'models' / 'face_emotion' / 'fer.h5',
        base_dir / 'models' / 'face_emotion' / 'best_mobilenet_model.h5',
        base_dir / 'emotion_model_weights.h5'
    ]
    
    valid_models = []
    
    for model_path in model_paths:
        if model_path.exists():
            file_size = model_path.stat().st_size
            if file_size > 1000:  # More than 1KB
                valid_models.append(model_path)
                print(f"  ✓ {model_path.name} ({file_size:,} bytes)")
            else:
                print(f"  ⚠️  {model_path.name} appears corrupted ({file_size} bytes)")
        else:
            print(f"  ✗ {model_path.name} - NOT FOUND")
    
    if valid_models:
        print(f"✅ Found {len(valid_models)} valid model(s)")
        return True
    else:
        print("❌ No valid models found")
        print("Models will use fallback detection methods")
        return False

def setup_database():
    """Setup database and run migrations"""
    print("\n💾 Setting up database...")
    
    try:
        # Run migrations
        subprocess.run([sys.executable, 'manage.py', 'makemigrations'], check=True)
        subprocess.run([sys.executable, 'manage.py', 'migrate'], check=True)
        print("✅ Database migrations completed")
        
        # Create superuser if needed
        setup_django()
        from django.contrib.auth.models import User
        
        if not User.objects.filter(is_superuser=True).exists():
            print("Creating default superuser...")
            User.objects.create_superuser('admin', 'admin@dailynest.com', 'admin123')
            print("✅ Superuser created (admin/admin123)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Database setup failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def collect_static():
    """Collect static files"""
    print("\n📁 Collecting static files...")
    
    try:
        subprocess.run([sys.executable, 'manage.py', 'collectstatic', '--noinput'], check=True)
        print("✅ Static files collected")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Static files collection failed: {e}")
        return False

def run_tests():
    """Run production tests"""
    print("\n🧪 Running production tests...")
    
    try:
        result = subprocess.run([sys.executable, 'test_production.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All production tests passed")
            return True
        else:
            print("❌ Some production tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 DailyNest Production Setup")
    print("=" * 50)
    
    setup_steps = [
        ("Dependencies", check_dependencies),
        ("ML Models", validate_models),
        ("Database", setup_database),
        ("Static Files", collect_static),
        ("Production Tests", run_tests),
    ]
    
    success_count = 0
    
    for step_name, step_func in setup_steps:
        print(f"\n📋 {step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"⚠️  {step_name} had issues but setup continues...")
    
    print("\n" + "=" * 50)
    print(f"🏁 Setup Results: {success_count}/{len(setup_steps)} steps completed successfully")
    
    if success_count == len(setup_steps):
        print("🎉 DailyNest is ready for production!")
        print("\nNext steps:")
        print("1. Run: python manage.py runserver")
        print("2. Visit: http://127.0.0.1:8000")
        print("3. Test emotion detection and chat features")
    else:
        print("⚠️  Some setup steps had issues. Review the output above.")
        print("The application may still work with reduced functionality.")

if __name__ == "__main__":
    main()
