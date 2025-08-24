// Chat functionality for DailyNest
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendMessage');
    const currentFaceEmotion = document.getElementById('currentFaceEmotion');
    const currentVoiceEmotion = document.getElementById('currentVoiceEmotion');

    // Add initial welcome message
    addMessage('bot', 'Hello! I\'m your emotion-aware assistant. How are you feeling today?');

    // Send message function
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        console.log('Sending message:', message);

        // Disable send button to prevent multiple submissions
        sendButton.disabled = true;
        sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

        // Add user message
        addMessage('user', message);
        messageInput.value = '';

        // Send to backend with simple data
        fetch('/chat-message/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                message: message,
                face_emotion: 'neutral',
                voice_emotion: 'neutral'
            })
        })
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Response data:', data);
            if (data.success && data.response) {
                addMessage('bot', data.response);
            } else if (data.error) {
                addMessage('bot', `Error: ${data.error}`);
            } else {
                addMessage('bot', 'I received your message. How can I help you today?');
            }
        })
        .catch(error => {
            console.error('Chat error:', error);
            addMessage('bot', 'I\'m here to help. What would you like to talk about?');
        })
        .finally(() => {
            // Re-enable send button
            sendButton.disabled = false;
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        });
    }

    // Add message to chat
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        messageDiv.innerHTML = `
            <div class="message-content">${text}</div>
            <div class="message-time">${timestamp}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Update emotion indicators periodically
    function updateEmotionIndicators() {
        fetch('/emotion-history/')
        .then(response => response.json())
        .then(data => {
            if (data.emotions && data.emotions.length > 0) {
                const latest = data.emotions[0];
                if (latest.face_emotion) {
                    currentFaceEmotion.innerHTML = `<i class="fas fa-camera"></i> ${latest.face_emotion}`;
                }
                if (latest.voice_emotion) {
                    currentVoiceEmotion.innerHTML = `<i class="fas fa-microphone"></i> ${latest.voice_emotion}`;
                }
            }
        })
        .catch(error => console.error('Error updating emotions:', error));
    }

    // Update emotions every 30 seconds
    setInterval(updateEmotionIndicators, 30000);

    // Get CSRF token
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
