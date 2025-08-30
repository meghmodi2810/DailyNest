document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const startVideo = document.getElementById('startVideo');
    const startAudio = document.getElementById('startAudio');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const faceEmotion = document.getElementById('faceEmotion');
    const voiceEmotion = document.getElementById('voiceEmotion');
    
    let videoStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isProcessing = false;

    // Video handling
    startVideo.addEventListener('click', async () => {
        try {
            if (videoStream) {
                stopVideo();
                return;
            }

            videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = videoStream;
            startVideo.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
            
            // Start face detection
            detectFaceEmotion();
        } catch (error) {
            console.error('Camera error:', error);
            showError(faceEmotion, 'Camera access denied');
        }
    });

    // Audio handling
    startAudio.addEventListener('click', async () => {
        try {
            if (mediaRecorder?.state === 'recording') {
                stopAudioRecording();
                return;
            }

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener('stop', () => {
                processAudioEmotion();
            });

            mediaRecorder.start();
            startAudio.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            recordingTimeout = setTimeout(() => stopAudioRecording(), 300000); // Auto-stop after 5 minutes
        } catch (error) {
            console.error('Microphone error:', error);
            showError(voiceEmotion, 'Microphone access denied');
        }
    });

    // Face emotion detection
    async function detectFaceEmotion() {
        const interval = setInterval(async () => {
            if (!videoStream || isProcessing) {
                if (!videoStream) clearInterval(interval);
                return;
            }

            isProcessing = true;
            startVideo.disabled = true;

            try {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                
                const response = await fetch('/detect-emotion/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: canvas.toDataURL('image/jpeg')
                    })
                });

                const data = await response.json();
                updateEmotionDisplay(faceEmotion, data.face_emotion);
            } catch (error) {
                console.error('Detection error:', error);
                showError(faceEmotion, 'Detection failed');
            } finally {
                isProcessing = false;
                startVideo.disabled = false;
            }
        }, 3000);
    }

    // Audio emotion processing
    async function processAudioEmotion() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        
        startAudio.disabled = true;
        
        reader.onload = async () => {
            try {
                const response = await fetch('/detect-emotion/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio: reader.result })
                });

                const data = await response.json();
                updateEmotionDisplay(voiceEmotion, data.voice_emotion);
            } catch (error) {
                console.error('Audio processing error:', error);
                showError(voiceEmotion, 'Processing failed');
            } finally {
                startAudio.disabled = false;
                startAudio.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            }
        };

        reader.readAsDataURL(audioBlob);
    }

    function stopVideo() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            video.srcObject = null;
            startVideo.innerHTML = '<i class="fas fa-video"></i> Start Camera';
        }
    }

    function stopAudioRecording() {
        if (mediaRecorder?.state === 'recording') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    function updateEmotionDisplay(element, emotion) {
        element.textContent = emotion || '-';
        element.className = emotion?.toLowerCase() || '';
    }

    function showError(element, message) {
        element.textContent = message;
        element.className = 'error';
    }
});