from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from src.core.chat_interface import load_model_and_vocabulary, process_user_input, ChatState

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

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True) 