document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const startCamera = document.getElementById('startCamera');
    const capturePhoto = document.getElementById('capturePhoto');
    const stopCamera = document.getElementById('stopCamera');
    const startRecording = document.getElementById('startRecording');
    const stopRecording = document.getElementById('stopRecording');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const videoOverlay = document.getElementById('videoOverlay');
    const emotionCircle = document.getElementById('emotionCircle');
    const emotionIcon = document.getElementById('emotionIcon');
    const emotionText = document.getElementById('emotionText');
    const confidenceBar = document.getElementById('confidenceBar');
    const micIcon = document.getElementById('micIcon');
    const audioStatus = document.getElementById('audioStatus');
    const audioBars = document.querySelectorAll('.bar');

    let videoStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isProcessing = false;
    let currentMode = 'face';

    // Mode switching
    window.switchMode = function(mode) {
        currentMode = mode;
        const faceMode = document.getElementById('faceMode');
        const voiceMode = document.getElementById('voiceMode');
        const cameraSection = document.getElementById('cameraSection');
        const microphoneSection = document.getElementById('microphoneSection');

        if (mode === 'face') {
            faceMode.classList.add('active');
            voiceMode.classList.remove('active');
            cameraSection.style.display = 'block';
            microphoneSection.style.display = 'none';
        } else {
            voiceMode.classList.add('active');
            faceMode.classList.remove('active');
            cameraSection.style.display = 'none';
            microphoneSection.style.display = 'block';
        }
    };

    // Camera controls
    if (startCamera) {
        startCamera.addEventListener('click', async () => {
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = videoStream;
                
                video.onloadedmetadata = () => {
                    videoOverlay.style.display = 'none';
                    startCamera.style.display = 'none';
                    capturePhoto.style.display = 'inline-flex';
                    stopCamera.style.display = 'inline-flex';
                };
                
            } catch (error) {
                console.error('Camera access error:', error);
                showError('Camera access denied. Please check permissions.');
            }
        });
    }

    if (capturePhoto) {
        capturePhoto.addEventListener('click', async () => {
            if (isProcessing || !videoStream) return;
            
            isProcessing = true;
            capturePhoto.disabled = true;
            capturePhoto.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
            
            try {
                // Capture frame
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                // Send to backend
                const response = await fetch('/detect-emotion/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({
                        image: canvas.toDataURL('image/jpeg', 0.8)
                    })
                });

                const data = await response.json();
                
                if (data.success && data.face_emotion) {
                    updateEmotionDisplay(data.face_emotion, data.confidence || 'medium');
                } else {
                    showError(data.error || 'Failed to detect emotion');
                }
                
            } catch (error) {
                console.error('Face detection error:', error);
                showError('Failed to process image');
            } finally {
                isProcessing = false;
                capturePhoto.disabled = false;
                capturePhoto.innerHTML = '<i class="fas fa-camera-retro"></i><span>Detect Emotion</span>';
            }
        });
    }

    if (stopCamera) {
        stopCamera.addEventListener('click', () => {
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
                video.srcObject = null;
                videoOverlay.style.display = 'flex';
                startCamera.style.display = 'inline-flex';
                capturePhoto.style.display = 'none';
                stopCamera.style.display = 'none';
            }
        });
    }

    // Audio controls
    if (startRecording) {
        startRecording.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    processAudioEmotion();
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                
                // Update UI
                startRecording.style.display = 'none';
                stopRecording.style.display = 'inline-flex';
                micIcon.className = 'fas fa-microphone fa-3x recording';
                audioStatus.textContent = 'Recording... Speak now!';
                startAudioVisualization();
                
                // Set maximum recording time (5 minutes)
                recordingTimer = setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                    }
                }, MAX_RECORDING_TIME);
                
            } catch (error) {
                console.error('Microphone access error:', error);
                showError('Microphone access denied. Please check permissions.');
            }
        });
    }

    if (stopRecording) {
        stopRecording.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        });
    }

    async function processAudioEmotion() {
        try {
            stopRecording.style.display = 'none';
            startRecording.style.display = 'inline-flex';
            startRecording.disabled = true;
            startRecording.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
            
            micIcon.className = 'fas fa-microphone fa-3x';
            audioStatus.textContent = 'Processing audio...';
            stopAudioVisualization();

            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            
            reader.onload = async () => {
                try {
                    const response = await fetch('/detect-emotion/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': getCookie('csrftoken')
                        },
                        body: JSON.stringify({
                            audio: reader.result
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success && data.voice_emotion) {
                        updateEmotionDisplay(data.voice_emotion, data.confidence || 'medium');
                        audioStatus.textContent = `Detected: ${data.voice_emotion}`;
                    } else {
                        showError(data.error || 'Failed to detect voice emotion');
                        audioStatus.textContent = 'Detection failed';
                    }
                    
                } catch (error) {
                    console.error('Voice processing error:', error);
                    showError('Failed to process audio');
                    audioStatus.textContent = 'Processing failed';
                } finally {
                    startRecording.disabled = false;
                    startRecording.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
                }
            };

            reader.readAsDataURL(audioBlob);
            
        } catch (error) {
            console.error('Audio processing error:', error);
            showError('Failed to process audio');
        }
    }

    function updateEmotionDisplay(emotion, confidence) {
        if (!emotionText || !emotionIcon || !emotionCircle) return;
        
        emotionText.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        // Update emotion icon
        const emotionIcons = {
            'happy': 'fas fa-smile',
            'sad': 'fas fa-frown',
            'angry': 'fas fa-angry',
            'surprised': 'fas fa-surprise',
            'fearful': 'fas fa-meh-blank',
            'disgusted': 'fas fa-grimace',
            'neutral': 'fas fa-meh',
            'excited': 'fas fa-grin-stars',
            'calm': 'fas fa-smile-beam'
        };
        
        emotionIcon.className = emotionIcons[emotion] || 'fas fa-meh';
        
        // Update circle color
        const emotionColors = {
            'happy': '#10B981',
            'sad': '#3B82F6',
            'angry': '#EF4444',
            'surprised': '#F59E0B',
            'fearful': '#8B5CF6',
            'disgusted': '#6B7280',
            'neutral': '#6B7280',
            'excited': '#F59E0B',
            'calm': '#10B981'
        };
        
        emotionCircle.style.background = `linear-gradient(135deg, ${emotionColors[emotion] || '#6B7280'}, ${emotionColors[emotion] || '#6B7280'}aa)`;
        
        // Update confidence bar
        if (confidenceBar) {
            const confidenceValue = confidence === 'high' ? 90 : confidence === 'medium' ? 70 : 50;
            confidenceBar.style.width = `${confidenceValue}%`;
        }
    }

    function startAudioVisualization() {
        let animationId;
        
        function animate() {
            audioBars.forEach((bar, index) => {
                const height = Math.random() * 100 + 20;
                bar.style.height = `${height}%`;
                bar.style.animationDelay = `${index * 0.1}s`;
            });
            animationId = requestAnimationFrame(animate);
        }
        
        animate();
        
        // Store animation ID for cleanup
        window.audioAnimationId = animationId;
    }

    function stopAudioVisualization() {
        if (window.audioAnimationId) {
            cancelAnimationFrame(window.audioAnimationId);
        }
        
        audioBars.forEach(bar => {
            bar.style.height = '20%';
        });
    }

    function showError(message) {
        if (emotionText) {
            emotionText.textContent = 'Error';
        }
        if (emotionIcon) {
            emotionIcon.className = 'fas fa-exclamation-triangle';
        }
        if (emotionCircle) {
            emotionCircle.style.background = 'linear-gradient(135deg, #EF4444, #EF4444aa)';
        }
        
        console.error('Emotion detection error:', message);
        
        // Show user-friendly error
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #EF4444;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            z-index: 1000;
            max-width: 300px;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
