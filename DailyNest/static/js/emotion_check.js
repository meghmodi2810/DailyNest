class EmotionCheck {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.detectButton = document.getElementById('detect-emotion');
        this.resultDiv = document.getElementById('emotion-result');
        this.emotionSpan = document.getElementById('detected-emotion');
        this.startActivityBtn = document.getElementById('start-activity');
        this.popup = document.getElementById('draggablePopup');
        this.dragHandle = document.getElementById('dragHandle');
        this.loadingState = document.getElementById('loading-state');
        this.detectionIndicator = document.getElementById('detectionIndicator');
        this.csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        this.detectionInterval = null;
        this.recommendedActivity = null;
        this.isDetecting = false;
        
        // Emotion to emoji mapping
        this.emotionEmojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprised': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐',
            'joy': '😄',
            'calm': '😌',
            'excited': '🤩'
        };
        
        // Activity recommendations
        this.activityRecommendations = {
            'happy': {
                title: 'Creative Expression',
                description: 'Channel your positive energy into creative activities!',
                icon: '🎨',
                url: '/games/calm-maze/'
            },
            'sad': {
                title: 'Calming Activities',
                description: 'Gentle activities to help lift your spirits.',
                icon: '🌸',
                url: '/games/breathing-garden/'
            },
            'angry': {
                title: 'Stress Relief',
                description: 'Activities to help you release tension and relax.',
                icon: '🎯',
                url: '/games/bubble-pop/'
            },
            'neutral': {
                title: 'Engaging Games',
                description: 'Fun activities to brighten your day!',
                icon: '🎮',
                url: '/games/'
            }
        };
        
        this.setupCamera();
        this.setupEventListeners();
        this.setupDraggable();
    }

    async setupCamera() {
        try {
            this.updateIndicator('Requesting camera access...', 'orange');
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.video.srcObject = stream;
            
            this.video.onloadedmetadata = () => {
                this.updateIndicator('Camera ready!', 'green');
                setTimeout(() => {
                    this.updateIndicator('Ready to detect emotions', 'blue');
                }, 1000);
            };
        } catch (err) {
            console.error('Camera error:', err);
            this.updateIndicator('Camera access denied', 'red');
            this.showCameraError();
        }
    }
    
    updateIndicator(message, color = 'green') {
        if (this.detectionIndicator) {
            const span = this.detectionIndicator.querySelector('span');
            const pulse = this.detectionIndicator.querySelector('.pulse');
            if (span) span.textContent = message;
            if (pulse) {
                pulse.style.backgroundColor = color === 'green' ? '#10b981' : 
                                              color === 'red' ? '#ef4444' : 
                                              color === 'orange' ? '#f59e0b' : '#6366f1';
            }
        }
    }
    
    showCameraError() {
        const cameraContainer = document.getElementById('camera-container');
        if (cameraContainer) {
            cameraContainer.innerHTML = `
                <div class="camera-error">
                    <i class="fas fa-camera-slash" style="font-size: 3rem; color: #ef4444; margin-bottom: 1rem;"></i>
                    <h4>Camera Access Required</h4>
                    <p>Please allow camera access to detect your emotions.</p>
                    <button onclick="location.reload()" class="btn-primary">
                        <i class="fas fa-refresh"></i> Try Again
                    </button>
                </div>
            `;
        }
    }

    setupEventListeners() {
        this.detectButton.addEventListener('click', () => this.detectEmotion());
        this.startActivityBtn.addEventListener('click', () => this.startRecommendedActivity());
    }

    setupDraggable() {
        let isDragging = false;
        let currentX;
        let currentY;
        let initialX;
        let initialY;
        let xOffset = 0;
        let yOffset = 0;

        const dragStart = (e) => {
            if (e.type === "touchstart") {
                initialX = e.touches[0].clientX - xOffset;
                initialY = e.touches[0].clientY - yOffset;
            } else {
                initialX = e.clientX - xOffset;
                initialY = e.clientY - yOffset;
            }
            
            if (e.target === this.dragHandle || e.target.parentNode === this.dragHandle) {
                isDragging = true;
                this.popup.classList.add('dragging');
            }
        };

        const dragEnd = () => {
            isDragging = false;
            this.popup.classList.remove('dragging');
        };

        const drag = (e) => {
            if (isDragging) {
                e.preventDefault();
                
                if (e.type === "touchmove") {
                    currentX = e.touches[0].clientX - initialX;
                    currentY = e.touches[0].clientY - initialY;
                } else {
                    currentX = e.clientX - initialX;
                    currentY = e.clientY - initialY;
                }

                xOffset = currentX;
                yOffset = currentY;
                
                this.popup.style.transform = 
                    `translate(${currentX}px, ${currentY}px)`;
            }
        };

        this.dragHandle.addEventListener('mousedown', dragStart);
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', dragEnd);
        this.dragHandle.addEventListener('touchstart', dragStart);
        document.addEventListener('touchmove', drag);
        document.addEventListener('touchend', dragEnd);
    }

    startAutomaticDetection() {
        // Detect emotion every 3 seconds
        this.detectionInterval = setInterval(() => {
            this.detectEmotion();
        }, 3000);
    }

    stopAutomaticDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }

    async detectEmotion() {
        if (this.isDetecting) return;
        
        this.isDetecting = true;
        this.showLoading();
        this.updateIndicator('Analyzing your emotion...', 'blue');
        
        // Capture video frame
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.canvas.getContext('2d').drawImage(this.video, 0, 0);
        
        const imageData = this.canvas.toDataURL('image/jpeg');
        
        try {
            const response = await fetch('/detect-emotion/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.csrfToken
                },
                body: JSON.stringify({
                    image: imageData
                })
            });
            
            const data = await response.json();
            if (data.success) {
                this.showResult(data.face_emotion, data.face_confidence || 0.8);
                this.updateIndicator('Emotion detected!', 'green');
            } else {
                this.showError('Could not detect emotion. Please try again.');
                this.updateIndicator('Detection failed', 'red');
            }
        } catch (err) {
            console.error('Detection error:', err);
            this.showError('Network error. Please check your connection.');
            this.updateIndicator('Connection error', 'red');
        } finally {
            this.isDetecting = false;
            this.hideLoading();
        }
    }
    
    showLoading() {
        if (this.loadingState) {
            this.loadingState.style.display = 'block';
        }
        this.detectButton.disabled = true;
        this.detectButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Analyzing...</span>';
    }
    
    hideLoading() {
        if (this.loadingState) {
            this.loadingState.style.display = 'none';
        }
        this.detectButton.disabled = false;
        this.detectButton.innerHTML = '<i class="fas fa-camera"></i> <span>Check My Emotion</span>';
    }
    
    showError(message) {
        // Show error in a user-friendly way
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <p>${message}</p>
        `;
        errorDiv.style.cssText = `
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #dc2626;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            text-align: center;
        `;
        
        // Remove existing error messages
        const existingError = document.querySelector('.error-message');
        if (existingError) existingError.remove();
        
        // Insert after camera container
        const cameraContainer = document.getElementById('camera-container');
        if (cameraContainer && cameraContainer.parentNode) {
            cameraContainer.parentNode.insertBefore(errorDiv, cameraContainer.nextSibling);
        }
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    showResult(emotion, confidence = 0.8) {
        // Update emotion display
        this.emotionSpan.textContent = emotion;
        
        // Update emotion icon
        const emotionIcon = document.getElementById('emotionIcon');
        if (emotionIcon) {
            emotionIcon.textContent = this.emotionEmojis[emotion.toLowerCase()] || '😊';
        }
        
        // Update confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceText = document.getElementById('confidenceText');
        if (confidenceBar && confidenceText) {
            const confidencePercent = Math.round(confidence * 100);
            confidenceBar.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `${confidencePercent}%`;
        }
        
        // Show result section
        this.resultDiv.style.display = 'block';
        this.detectButton.innerHTML = '<i class="fas fa-camera"></i> <span>Detect Again</span>';
        
        // Get activity recommendation
        this.getRecommendation(emotion);
        
        // Stop automatic detection after first successful detection
        this.stopAutomaticDetection();
    }

    async getRecommendation(emotion) {
        // Use local recommendations for better reliability
        const recommendation = this.activityRecommendations[emotion.toLowerCase()] || 
                              this.activityRecommendations['neutral'];
        
        // Update UI with recommendation
        const activityTitle = document.getElementById('activityTitle');
        const activityRecommendation = document.getElementById('activity-recommendation');
        const activityIcon = document.getElementById('activityIcon');
        
        if (activityTitle) activityTitle.textContent = recommendation.title;
        if (activityRecommendation) activityRecommendation.textContent = recommendation.description;
        if (activityIcon) activityIcon.textContent = recommendation.icon;
        
        this.recommendedActivity = recommendation.url;
        
        // Also try to fetch server recommendation as backup
        try {
            const response = await fetch('/get-activity-recommendation/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.csrfToken
                },
                body: JSON.stringify({ emotion })
            });
            
            const data = await response.json();
            if (data.success && data.recommendation) {
                // Update with server recommendation if available
                if (activityRecommendation) {
                    activityRecommendation.textContent = data.recommendation;
                }
                if (data.activity_url) {
                    this.recommendedActivity = data.activity_url;
                }
            }
        } catch (err) {
            console.log('Server recommendation unavailable, using local recommendation');
        }
    }

    startRecommendedActivity() {
        if (this.recommendedActivity) {
            // Close the popup first
            document.getElementById('emotionCheckPopup').style.display = 'none';
            
            // Update the database to mark emotion check as completed
            fetch('/skip-emotion-check/', { 
                method: 'POST',
                headers: {
                    'X-CSRFToken': this.csrfToken
                }
            }).then(() => {
                // Navigate to the recommended activity
                window.location.href = this.recommendedActivity;
            }).catch(() => {
                // Navigate even if the database update fails
                window.location.href = this.recommendedActivity;
            });
        } else {
            // Fallback to games hub if no specific activity
            window.location.href = '/games/';
        }
    }

    minimize() {
        this.popup.classList.toggle('minimized');
        if (this.popup.classList.contains('minimized')) {
            this.stopAutomaticDetection();
        } else {
            this.startAutomaticDetection();
        }
    }
}

function skipEmotionCheck() {
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    document.getElementById('emotionCheckPopup').style.display = 'none';
    fetch('/skip-emotion-check/', { 
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken
        }
    });
}

function minimizePopup() {
    if (window.emotionChecker) {
        window.emotionChecker.minimize();
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('emotionCheckPopup')) {
        window.emotionChecker = new EmotionCheck();
    }
});