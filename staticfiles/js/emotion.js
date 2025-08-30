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
    const emotionText = document.getElementById('emotionText');
    const emotionIcon = document.getElementById('emotionIcon');
    const emotionCircle = document.getElementById('emotionCircle');
    const confidenceBar = document.getElementById('confidenceBar');
    const micIcon = document.getElementById('micIcon');
    const audioStatus = document.getElementById('audioStatus');

    let videoStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isProcessing = false;

    // Mode switching
    window.switchMode = function(mode) {
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
                console.log('Starting camera...');
                videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = videoStream;
                
                video.onloadedmetadata = () => {
                    console.log('Video loaded');
                    if (videoOverlay) videoOverlay.style.display = 'none';
                    startCamera.style.display = 'none';
                    if (capturePhoto) capturePhoto.style.display = 'inline-flex';
                    if (stopCamera) stopCamera.style.display = 'inline-flex';
                };
                
            } catch (error) {
                console.error('Camera error:', error);
                showResult('Camera access denied', 'error');
            }
        });
    }

    if (capturePhoto) {
        capturePhoto.addEventListener('click', async () => {
            if (isProcessing || !videoStream) return;
            
            console.log('Capturing photo...');
            isProcessing = true;
            capturePhoto.disabled = true;
            capturePhoto.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
            
            try {
                // Capture frame
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
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
                console.log('Face detection response:', data);
                
                if (data.success && data.face_emotion) {
                    showResult(data.face_emotion, 'face');
                } else {
                    showResult(data.error || 'Detection failed', 'error');
                }
                
            } catch (error) {
                console.error('Face detection error:', error);
                showResult('Processing failed', 'error');
            } finally {
                isProcessing = false;
                capturePhoto.disabled = false;
                capturePhoto.innerHTML = '<i class="fas fa-camera-retro"></i><span>Detect Emotion</span>';
            }
        });
    }

    if (stopCamera) {
        stopCamera.addEventListener('click', () => {
            console.log('Stopping camera...');
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
                video.srcObject = null;
                if (videoOverlay) videoOverlay.style.display = 'flex';
                startCamera.style.display = 'inline-flex';
                if (capturePhoto) capturePhoto.style.display = 'none';
                stopCamera.style.display = 'none';
            }
        });
    }

    // Audio controls
    if (startRecording) {
        startRecording.addEventListener('click', async () => {
            try {
                console.log('Starting recording...');
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: true
                });
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    console.log('Recording stopped, processing...');
                    processAudio();
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                
                // Update UI
                startRecording.style.display = 'none';
                if (stopRecording) stopRecording.style.display = 'inline-flex';
                if (micIcon) micIcon.className = 'fas fa-microphone fa-3x recording';
                if (audioStatus) audioStatus.textContent = 'Recording... Speak now!';
                
                // Set maximum recording time (5 minutes)
                recordingTimeout = setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        if (audioStatus) audioStatus.textContent = 'Maximum recording time (5 minutes) reached';
                    }
                }, 300000); // 5 minutes
                
            } catch (error) {
                console.error('Microphone error:', error);
                showResult('Microphone access denied', 'error');
            }
        });
    }

    if (stopRecording) {
        stopRecording.addEventListener('click', () => {
            console.log('Manual stop recording...');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        });
    }

    async function processAudio() {
        try {
            if (stopRecording) stopRecording.style.display = 'none';
            startRecording.style.display = 'inline-flex';
            startRecording.disabled = true;
            startRecording.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
            
            if (micIcon) micIcon.className = 'fas fa-microphone fa-3x';
            if (audioStatus) audioStatus.textContent = 'Processing audio...';

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
                    console.log('Voice detection response:', data);
                    
                    if (data.success && data.voice_emotion) {
                        showResult(data.voice_emotion, 'voice');
                        if (audioStatus) audioStatus.textContent = `Detected: ${data.voice_emotion}`;
                    } else {
                        showResult(data.error || 'Detection failed', 'error');
                        if (audioStatus) audioStatus.textContent = 'Detection failed';
                    }
                    
                } catch (error) {
                    console.error('Voice processing error:', error);
                    showResult('Processing failed', 'error');
                    if (audioStatus) audioStatus.textContent = 'Processing failed';
                } finally {
                    startRecording.disabled = false;
                    startRecording.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
                }
            };

            reader.readAsDataURL(audioBlob);
            
        } catch (error) {
            console.error('Audio processing error:', error);
            showResult('Processing failed', 'error');
        }
    }

    function showResult(emotion, type) {
        console.log('Showing result:', emotion, type);
        
        if (emotionText) {
            emotionText.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        }
        
        if (emotionIcon) {
            const emotionIcons = {
                'happy': 'fas fa-smile',
                'sad': 'fas fa-frown',
                'angry': 'fas fa-angry',
                'surprised': 'fas fa-surprise',
                'neutral': 'fas fa-meh',
                'calm': 'fas fa-smile-beam',
                'excited': 'fas fa-grin-stars',
                'error': 'fas fa-exclamation-triangle'
            };
            emotionIcon.innerHTML = `<i class="${emotionIcons[emotion] || 'fas fa-meh'}"></i>`;
        }
        
        if (emotionCircle) {
            const emotionColors = {
                'happy': '#10B981',
                'sad': '#3B82F6',
                'angry': '#EF4444',
                'surprised': '#F59E0B',
                'neutral': '#6B7280',
                'calm': '#10B981',
                'excited': '#F59E0B',
                'error': '#EF4444'
            };
            emotionCircle.style.background = `linear-gradient(135deg, ${emotionColors[emotion] || '#6B7280'}, ${emotionColors[emotion] || '#6B7280'}aa)`;
        }
        
        if (confidenceBar) {
            const confidence = type === 'error' ? 0 : 85;
            confidenceBar.style.width = `${confidence}%`;
        }
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
