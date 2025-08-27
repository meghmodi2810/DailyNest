document.addEventListener('DOMContentLoaded', () => {
    // Initialize settings
    const settingsButton = document.getElementById('settingsButton');
    const settingsPanel = document.getElementById('settingsPanel');
    const messageInput = document.getElementById('messageInput');
    const sendMessage = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    
    // Accessibility settings
    const themeSelect = document.getElementById('theme');
    const fontSizeSelect = document.getElementById('fontSize');
    const reduceAnimations = document.getElementById('reduceAnimations');
    const highContrast = document.getElementById('highContrast');
    const textToSpeech = document.getElementById('textToSpeech');
    
    // Video and audio elements
    const startVideo = document.getElementById('startVideo');
    const startAudio = document.getElementById('startAudio');
    const videoContainer = document.getElementById('videoContainer');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const emotionResults = document.getElementById('emotionResults');
    const faceEmotion = document.getElementById('faceEmotion');
    const voiceEmotion = document.getElementById('voiceEmotion');

    let mediaRecorder = null;
    let audioChunks = [];
    let videoStream = null;
    let isProcessing = false;

    function setButtonState(button, isLoading) {
        if (isLoading) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            button.disabled = false;
            button.innerHTML = button === startVideo ? 
                '<i class="fas fa-video"></i>' : 
                '<i class="fas fa-microphone"></i>';
        }
    }

    function showEmotionResult(element, emotion) {
        if (emotion === "uncertain") {
            element.innerHTML = '<span class="uncertain">Uncertain (Low confidence)</span>';
        } else if (emotion && emotion.startsWith("Error")) {
            element.innerHTML = `<span class="error">${emotion}</span>`;
        } else {
            element.textContent = emotion || '-';
        }
    }

    // Settings panel toggle
    settingsButton.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
    });

    // Theme handling
    themeSelect.addEventListener('change', async (e) => {
        document.body.dataset.theme = e.target.value;
        await updatePreferences({ theme: e.target.value });
    });

    // Font size handling
    fontSizeSelect.addEventListener('change', async (e) => {
        document.body.dataset.fontSize = e.target.value;
        await updatePreferences({ fontSize: e.target.value });
    });

    // Other accessibility settings
    reduceAnimations.addEventListener('change', async (e) => {
        document.body.dataset.reduceAnimations = e.target.checked;
        await updatePreferences({ reduce_animations: e.target.checked });
    });

    highContrast.addEventListener('change', async (e) => {
        document.body.dataset.highContrast = e.target.checked;
        await updatePreferences({ high_contrast_mode: e.target.checked });
    });

    textToSpeech.addEventListener('change', async (e) => {
        await updatePreferences({ text_to_speech: e.target.checked });
        if (e.target.checked) {
            initTextToSpeech();
        }
    });

    // Video handling
    startVideo.addEventListener('click', async () => {
        try {
            if (videoStream) {
                stopVideo();
                return;
            }

            videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = videoStream;
            videoContainer.classList.remove('hidden');
            startVideo.innerHTML = '<i class="fas fa-stop"></i>';
            
            // Start emotion detection
            startFaceEmotionDetection();
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Could not access camera. Please check permissions.');
        }
    });

    // Audio handling
    startAudio.addEventListener('click', async () => {
        try {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
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
            startAudio.innerHTML = '<i class="fas fa-stop"></i>';
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    });

    // Chat handling
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessageToBot();
        }
    });

    sendMessage.addEventListener('click', sendMessageToBot);

    // Helper functions
    async function updatePreferences(preferences) {
        try {
            await fetch('/update-preferences/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(preferences)
            });
        } catch (error) {
            console.error('Error updating preferences:', error);
        }
    }

    function stopVideo() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            video.srcObject = null;
            videoContainer.classList.add('hidden');
            startVideo.innerHTML = '<i class="fas fa-video"></i>';
        }
    }

    function stopAudioRecording() {
        if (mediaRecorder) {
            mediaRecorder.stop();
            startAudio.innerHTML = '<i class="fas fa-microphone"></i>';
        }
    }

    async function startFaceEmotionDetection() {
        const captureInterval = setInterval(async () => {
            if (!videoStream || isProcessing) {
                if (!videoStream) clearInterval(captureInterval);
                return;
            }

            isProcessing = true;
            setButtonState(startVideo, true);

            // Capture video frame
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            try {
                const response = await fetch('/detect-emotion/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: canvas.toDataURL('image/jpeg')
                    })
                });

                const data = await response.json();
                if (data.face_emotion) {
                    showEmotionResult(faceEmotion, data.face_emotion);
                    emotionResults.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error detecting face emotion:', error);
                showEmotionResult(faceEmotion, "Error: Failed to detect emotion");
            } finally {
                isProcessing = false;
                setButtonState(startVideo, false);
            }
        }, 3000); // Detect every 3 seconds
    }

    async function processAudioEmotion() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        
        setButtonState(startAudio, true);
        
        reader.onload = async () => {
            try {
                const response = await fetch('/detect-emotion/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio: reader.result })
                });
    
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
    
                const data = await response.json();
                if (data.error) {
                    showEmotionResult(voiceEmotion, `Error: ${data.error}`);
                } else {
                    showEmotionResult(voiceEmotion, data.voice_emotion);
                    emotionResults.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error detecting voice emotion:', error);
                showEmotionResult(voiceEmotion, "Error: Failed to detect emotion");
            } finally {
                setButtonState(startAudio, false);
            }
        };
    
        reader.readAsDataURL(audioBlob);
    }
    
    async function sendMessageToBot() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessageToChat('user', message);
        messageInput.value = '';

        try {
            const response = await fetch('/chat-message/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            if (data.response) {
                addMessageToChat('bot', data.response);
                if (textToSpeech.checked) {
                    speakText(data.response);
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToChat('bot', 'Sorry, I had trouble processing your message.');
        }
    }

    function addMessageToChat(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Text-to-speech functionality
    let speechSynthesis = window.speechSynthesis;
    let speaking = false;

    function initTextToSpeech() {
        if (!('speechSynthesis' in window)) {
            console.error('Text-to-speech not supported');
            return;
        }
    }

    function speakText(text) {
        if (!speechSynthesis || speaking) return;

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9; // Slightly slower rate for better comprehension
        utterance.pitch = 1;
        
        utterance.onstart = () => {
            speaking = true;
        };
        
        utterance.onend = () => {
            speaking = false;
        };

        speechSynthesis.speak(utterance);
    }
});