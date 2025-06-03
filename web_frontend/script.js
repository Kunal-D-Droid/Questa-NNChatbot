const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    msgDiv.textContent = text;
    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    appendMessage(message, 'user');
    userInput.value = '';
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage(data.response, 'bot');
    })
    .catch(() => {
        appendMessage('Sorry, there was an error connecting to Questa.', 'bot');
    });
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') sendMessage();
}); 