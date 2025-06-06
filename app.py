from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from src.core.chat_interface import load_model_and_vocabulary, process_user_input, ChatState, log_conversation_emotions

app = Flask(__name__, static_folder='web_frontend')
CORS(app)

# Load model and vocabulary once at startup
chatbot = load_model_and_vocabulary()
chat_state = ChatState()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    response = process_user_input(user_message, chatbot, chat_state)
    return jsonify({'response': response})

# Add a new endpoint to save the chat log
@app.route('/save_chat_log', methods=['POST'])
def save_chat_log():
    global chat_state  # Declare chat_state as global at the beginning
    try:
        # Log the current conversation state
        log_file_path = log_conversation_emotions(chat_state)

        # Reset chat state for a new conversation after logging
        chat_state = ChatState()

        return jsonify({'message': f'Conversation logged to: {log_file_path}'}) # Return the file path or confirmation
    except Exception as e:
        return jsonify({'error': f'Failed to save chat log: {str(e)}'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True) 