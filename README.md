# Emotion-Aware Hotel Assistant Chatbot

This project implements a chatbot focused on hotel booking that also uses an Emotion Text Analyzer Model to detect emotions in user messages and influences responses based on them, as well as logs the emotional summary of conversations.

## Project Structure

```
.
├── data/                       # Contains data related to models and training (raw, processed, models)
├── emotion_api/                # FastAPI application for the Emotion Text Analyzer Model
│   └── main.py
├── logs/                       # Stores conversation logs and potentially training logs
│   ├── emotion_log_YYYYMMDD_HHMMSS.json # Example conversation log file
│   └── ...
├── src/                        # Contains the core Python source code
│   ├── core/                   # Core chatbot logic and interfaces
│   │   ├── chat_interface.py   # Handles chat flow, intent detection, state management
│   │   └── hotel_chatbot.py    # (Likely contains the neural model logic, used by chat_interface)
│   ├── emotion_analyzer/       # Code for the Emotion Text Analyzer Model
│   │   └── model.py            # EmotionAnalyzer class implementation
│   ├── test_model.py           # Script to test the core chatbot model
│   └── ... (other utility/training folders)
├── web_frontend/               # Static files for the web interface (HTML, CSS, JS, images)
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── switch.png              # End chat icon
├── visualizations/             # Contains visualizations related to model training/analysis
├── .git/                       # Git version control
├── chatbot_env/                # Python virtual environment
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv chatbot_env
    # On Windows:
    chatbot_env\Scripts\activate
    # On macOS/Linux:
    source chatbot_env/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the necessary models (Chatbot and Emotion Analyzer):**
    *   *(Note: Training scripts and process are assumed to be handled separately within the `src/` and `data/` structure. Ensure models and tokenizer files are present in `data/models/` as expected by the code.)*
    ```bash
    # Example: (Adjust based on actual training scripts)
    # python src/training/train_chatbot.py
    # python src/emotion_analyzer/train_emotion_model.py
    ```
5.  **Start the Emotion Analyzer API:**
    ```bash
    cd emotion_api
    uvicorn main:app --reload
    # Keep this terminal window open or run in the background
    ```
6.  **Run the main Chatbot application (Flask server):**
    ```bash
    cd .. # Go back to the project root if you are in emotion_api
    python app.py
    # Keep this terminal window open or run in the background
    ```
7.  **Access the web interface:** Open your browser and go to `http://127.0.0.1:5000/` (or the address where Flask is running).

## API Endpoints

*   **`/chat` (POST)**:
    *   **Description:** Sends a user message to the chatbot and receives a response.
    *   **Input:** `application/json` with a `message` key (e.g., `{"message": "Hello"}`).
    *   **Output:** `application/json` with a `response` key (e.g., `{"response": "Hi there!"}`).
*   **`/save_chat_log` (POST)**:
    *   **Description:** Triggers saving the current conversation log with emotion analysis and resets the chat state.
    *   **Input:** None (empty body or `{}`).
    *   **Output:** `application/json` with a `message` key indicating the log file path or an `error` key on failure.
*   **`/predict` (POST) - Emotion API**:
    *   **Description:** Predicts emotion from input text using the Emotion Analyzer Model.
    *   **Input:** `application/json` with a `text` key (e.g., `{"text": "I am very happy"}`).
    *   **Output:** `application/json` with `emotion`, `confidence`, and `all_emotions` keys.

## Features

*   **Hotel Booking Functionality:** Supports booking, cancelling, and modifying hotel reservations through a conversational interface with state management.
*   **Place Suggestions:** Provides suggestions for places to visit based on user queries and specified city.
*   **Emotion Detection Integration:** Utilizes an external Emotion Text Analyzer API to detect emotions in user messages.
*   **Emotion-Aware Responses:** Chatbot responses can be influenced by detected emotions, including specific handling for complaints/negative feedback.
*   **Conversation Logging:** Saves a log of the conversation history, including emotion analysis, triggered by an "End Chat" action in the UI.
*   **Feedback/Suggestion Handling:** Includes intent detection and response logic for user feedback and suggestions.
*   **Web User Interface:** Provides a simple web interface for interacting with the chatbot.

## Logging

The chatbot logs conversations and their emotion summaries in the `logs/` directory. Each conversation is saved to a file named `emotion_log_YYYYMMDD_HHMMSS.json` when the "End Chat" action is performed in the web interface.

## Deployment

*(This section primarily describes deploying the FastAPI Emotion API. Deployment of the Flask chatbot application would require additional steps.)*

To deploy the API to AWS EC2:

1. Launch an EC2 instance (t2.micro or larger recommended).
2. Install Python and required dependencies.
3. Copy the project files to the instance.
4. Ensure trained models and tokenizer are present in the expected `data/models/` paths relative to the application.
5. Run the FastAPI server using:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

For production deployment, consider using:
- Gunicorn as the WSGI server
- Nginx as a reverse proxy
- Supervisor for process management

## Contributing

Feel free to submit issues and enhancement requests!
