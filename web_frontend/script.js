const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-button');
const endChatIcon = document.getElementById('end-chat-icon');

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

function endChat() {
    userInput.disabled = true;
    sendBtn.disabled = true;
    endChatIcon.style.pointerEvents = 'none';

    appendMessage('Ending chat and saving log...', 'bot');

    fetch('/save_chat_log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        appendMessage(data.message, 'bot');
    })
    .catch(error => {
        console.error('Error saving chat log:', error);
        appendMessage('Sorry, there was an error saving the chat log.', 'bot');
    })
    .finally(() => {
        // Optionally, provide a way to start a new chat (e.g., reload the page)
        // appendMessage('Please refresh the page to start a new chat.', 'bot');
    });
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') sendMessage();
});

endChatIcon.addEventListener('click', endChat); 