import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import pickle
from .hotel_chatbot import HotelBookingChatbot
import tensorflow as tf
import re
from datetime import datetime
import os
import uuid
import requests
import random

def load_model_and_vocabulary():
    """Load the trained model and vocabulary"""
    try:
        # Load vocabulary
        with open('data/models/vocabulary.pkl', 'rb') as f:
            word2idx, idx2word = pickle.load(f)
        
        # Initialize chatbot
        chatbot = HotelBookingChatbot()
        chatbot.word2idx = word2idx
        chatbot.idx2word = idx2word
        
        # Load model weights
        try:
            # First try loading the full model
            try:
                chatbot.model = tf.keras.models.load_model('data/models/best_model.keras')
                print("Successfully loaded full model from Keras format")
            except:
                # If that fails, build the model and load weights
                print("Building model and loading weights...")
                chatbot.build_model()
                chatbot.model.load_weights('data/models/best_model.weights.h5')
                print("Successfully loaded model weights")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        
        return chatbot
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return None

def extract_booking_details(message):
    """Extract booking details from user message"""
    # Extract dates
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, message)
    
    # Extract number of people
    people_pattern = r'(\d+)\s*(?:people|persons|guests)'
    people_match = re.search(people_pattern, message.lower())
    
    # Extract city
    cities = ['london', 'paris', 'new york', 'tokyo', 'sydney', 'chennai', 'mumbai', 'delhi', 'bangalore']
    city = None
    for c in cities:
        if c in message.lower():
            city = c
            break
    
    if len(dates) >= 2 and people_match and city:
        return {
            'check_in': dates[0],
            'check_out': dates[1],
            'city': city,
            'num_people': int(people_match.group(1))
        }
    return None

class ChatState:
    def __init__(self):
        self.pending_booking = None
        self.pending_cancellation = None
        self.pending_modification = None
        self.modification_type = None
        self.active_bookings = {}
        self.last_response = None
        self.waiting_for_booking_id = False
        self.waiting_for_place_suggestion_city = False
        self.conversation_emotions = []  # Track emotions throughout conversation
        self.feedback_given = False  # Track if feedback was requested

def generate_booking_id():
    """Generate a unique booking ID"""
    return str(uuid.uuid4())[:8].upper()  # Using first 8 characters for simplicity

def extract_booking_id(message):
    """Extract booking ID from user message"""
    # Match 8-character alphanumeric booking ID
    pattern = r'[A-Z0-9]{8}'
    match = re.search(pattern, message.upper())
    return match.group(0) if match else None

def analyze_emotion(text):
    """Analyze emotion using the deployed API"""
    try:
        response = requests.post(
            'http://localhost:8000/predict',  # Changed to use local API
            json={'text': text}
        )
        if response.status_code == 200:
            return response.json()
        print(f"Error from emotion API: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error calling emotion API: {str(e)}")
        return None

def get_emotion_based_response(emotion_result):
    """Generate emotion-aware response"""
    if not emotion_result:
        return "How can I assist you today?"
        
    emotion = emotion_result.get('emotion', 'neutral')
    confidence = emotion_result.get('confidence', 0)
    
    # Base responses for different emotions
    emotion_responses = {
        'joy': [
            "I'm glad to hear you're feeling happy! How can I make your day even better?",
            "Your happiness is contagious! What can I help you with today?",
            "It's wonderful to hear you're feeling joyful! How may I assist you?"
        ],
        'sadness': [
            "I'm sorry to hear you're feeling down. Is there something specific I can help you with?",
            "I'm here to help make things better. What would you like to know about our services?",
            "Let me try to brighten your day. What can I do for you?"
        ],
        'anger': [
            "I understand you're feeling frustrated. Let me help resolve any issues you're facing.",
            "I'm here to help address your concerns. What's troubling you?",
            "Let's work together to find a solution. What can I help you with?"
        ],
        'fear': [
            "There's no need to worry. I'm here to help guide you through this.",
            "Let me help put your mind at ease. What would you like to know?",
            "I'm here to support you. What can I clarify for you?"
        ],
        'surprise': [
            "I see you're surprised! Is there something specific you'd like to know more about?",
            "What caught you by surprise? I'm here to help explain things.",
            "Let me help you understand better. What would you like to know?"
        ],
        'neutral': [
            "How can I assist you today?",
            "What would you like to know about our services?",
            "I'm here to help. What can I do for you?"
        ]
    }
    
    # Get a random response for the detected emotion
    response = random.choice(emotion_responses.get(emotion, emotion_responses['neutral']))
    
    # Add a follow-up question based on the emotion
    if emotion == 'joy':
        response += "\nWould you like to know about our special offers or amenities?"
    elif emotion == 'sadness':
        response += "\nPerhaps I can tell you about our spa services or room upgrades?"
    elif emotion == 'anger':
        response += "\nWould you like to speak with our customer service team?"
    elif emotion == 'fear':
        response += "\nWould you like me to explain our safety measures and policies?"
    elif emotion == 'surprise':
        response += "\nWould you like to know more about our unique features?"
    
    return response

def get_conversation_summary(emotions):
    """Calculate emotion summary for the entire conversation"""
    if not emotions:
        return "No emotions recorded in conversation."
    
    emotion_counts = {}
    for emotion_data in emotions:
        emotion = emotion_data.get('emotion')
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    summary = "Conversation Emotion Summary:\n"
    for emotion, count in emotion_counts.items():
        percentage = (count / len(emotions)) * 100
        summary += f"- {emotion}: {percentage:.1f}%\n"
    
    return summary

def log_conversation_emotions(chat_state):
    """Log conversation emotions to a JSON file"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'emotion_log_{timestamp}.json')
    
    log_data = {
        'timestamp': timestamp,
        'emotions': chat_state.conversation_emotions,
        'summary': get_conversation_summary(chat_state.conversation_emotions)
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file

def process_user_input(user_input, chatbot, chat_state):
    """Process user input and generate appropriate response"""
    user_input = user_input.lower().strip()

    # 0. Handle pending state first (cancellation, modification, place suggestion city)
    if chat_state.pending_cancellation:
        booking_id = user_input.strip().upper()
        response = chatbot.cancel_booking(booking_id)
        chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
        chat_state.pending_cancellation = False
        return response
    if chat_state.pending_modification and chat_state.modification_type:
        booking_id = chat_state.pending_modification
        user_message = user_input.strip()
        # Handle if user just says 'guests' or 'dates'
        if user_message in ["guests", "number of guests"]:
            return "Please enter the new number of guests."
        if user_message in ["dates", "date", "check-in", "check-out"]:
            return "Please enter the new check-in and check-out dates in YYYY-MM-DD to YYYY-MM-DD format."
        changes = {}
        # If user enters just a number, treat as guests
        if user_message.isdigit():
            changes['num_people'] = int(user_message)
        # If user enters two dates (e.g., 2025-12-18 to 2025-12-22)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, user_message)
        if len(dates) == 2:
            changes['check_in'] = dates[0]
            changes['check_out'] = dates[1]
        # Fallback to previous extraction logic
        details = extract_booking_details(user_message)
        if details:
            if details.get('check_in'):
                changes['check_in'] = details['check_in']
            if details.get('check_out'):
                changes['check_out'] = details['check_out']
            if details.get('num_people'):
                changes['num_people'] = details['num_people']
        people_pattern = r'(\d+)\s*(?:people|persons|guests)'
        people_match = re.search(people_pattern, user_message.lower())
        if people_match:
            changes['num_people'] = int(people_match.group(1))
        if not changes:
            return "Please specify the new number of guests (e.g., '4') or the new dates (e.g., '2025-12-18 to 2025-12-22')."
        response = chatbot.modify_booking(booking_id, **changes)
        chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
        chat_state.pending_modification = None
        chat_state.modification_type = None
        return response
    if chat_state.pending_modification:
        booking_id = user_input.strip().upper()
        if booking_id in chatbot.bookings and chatbot.bookings[booking_id]['status'] == 'confirmed':
            chat_state.pending_modification = booking_id
            chat_state.modification_type = True
            return "What would you like to modify for this booking? (You can provide new dates or number of guests)"
        else:
            chat_state.pending_modification = None
            return "Booking ID not found or not active. Please provide a valid Booking ID."

    # Handle pending place suggestion city
    if chat_state.waiting_for_place_suggestion_city:
        city = user_input.strip().lower()
        chat_state.waiting_for_place_suggestion_city = False # Reset the flag
        suggestions = chatbot.get_place_suggestions(city)
        if suggestions and suggestions[0] != "No suggestions available for this city":
            return f"Here are some places to visit in {city.title()}:\n- " + "\n- ".join(suggestions)
        else:
            return f"Sorry, I don't have specific place suggestions for {city.title()} yet."

    print(f"[DEBUG] User input for intent detection: {user_input}")

    # 1. Rule-based intent detection for FAQs and specific tasks
    intent = None
    # Check for specific intents in order of priority
    if any(word in user_input for word in ["show bookings", "my bookings", "view bookings", "list bookings", "show booking", "my booking", "view booking", "list booking"]):
        intent = "show_bookings"
    elif any(word in user_input for word in ["cancel", "cancellation"]):
        intent = "cancel"
    elif any(word in user_input for word in ["modify", "change", "update"]):
        intent = "modify"
    elif any(word in user_input for word in ["place suggestion", "attraction", "sightseeing", "what to see", "best places", "places nearby"]):
        intent = "place_suggestion"
    elif any(word in user_input.split() for word in ["complaint", "angry", "bad", "disgusting", "dirty", "disappointed", "dissatisfied", "unhappy", "problem", "disliked"]):
        intent = "complaint"
    elif any(word in user_input for word in ["feedback", "suggestion", "suggest"]):
        intent = "feedback_suggestion"
    elif any(word in user_input for word in ["book", "reservation", "reserve", "room"]):
        intent = "booking"
    elif any(word in user_input for word in ["wifi", "internet"]):
        intent = "wifi"
    elif any(word in user_input for word in ["facility", "facilities", "amenities", "services"]):
        intent = "facilities"
    elif any(word in user_input for word in ["pet", "pets", "dog", "cat", "animal"]):
        intent = "pet_policy"
    elif any(word in user_input for word in ["child", "children", "kid", "kids", "infant", "baby"]):
        intent = "children_policy"
    elif any(word in user_input for word in ["gym", "fitness", "workout"]):
        intent = "gym"
    elif any(word in user_input for word in ["pool", "swimming pool"]):
        intent = "pool"
    elif any(word in user_input for word in ["parking", "car park", "garage"]):
        intent = "parking"
    elif any(word in user_input for word in ["checkin", "check-in", "checkout", "check-out", "check in", "check out", "timing", "time"]):
        intent = "checkin_checkout"
    # Add more intents as needed

    print(f"[DEBUG] Detected intent: {intent}")

    # 2. Handle detected intents
    if intent == "booking":
        booking_details = extract_booking_details(user_input)
        if booking_details:
            response = chatbot.handle_booking(
                booking_details['check_in'],
                booking_details['check_out'],
                booking_details['city'],
                booking_details['num_people']
            )
            # Sync bookings between chatbot and chat_state
            chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
            return response
        else:
            return "To book a hotel, please provide:\n- City (e.g., London, Paris, New York)\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- Number of guests\n\nExample: I want to book a hotel in London for 2 people from 2024-03-01 to 2024-03-05"
    elif intent == "wifi":
        return "Yes, we provide complimentary high-speed WiFi in all rooms and public areas."
    elif intent == "facilities":
        return "We offer a gym, pool, spa, in-room dining, and more. What would you like to know about?"
    elif intent == "complaint":
        print("[DEBUG] Handling complaint intent.")
        return "I'm very sorry to hear you're having a problem. Your feedback is important to us. Could you please provide more details so I can see how I can help, or would you like me to connect you with a member of our staff?"
    elif intent == "pet_policy":
        return "Yes, we are pet-friendly! Please let us know if you'll be bringing a pet so we can prepare accordingly."
    elif intent == "children_policy":
        return "Yes, children are welcome! Please let us know their ages for the best room options."
    elif intent == "gym":
        return "Yes, we have a fully equipped gym available for all guests from 6am to 10pm."
    elif intent == "pool":
        return "Yes, we have a swimming pool available for all guests from 6am to 10pm."
    elif intent == "parking":
        return "Yes, we offer free parking for our guests. Please let us know if you need a parking spot reserved."
    elif intent == "checkin_checkout":
        return "Our standard check-in time is 2:00 PM and check-out time is 12:00 PM. Early check-in and late check-out are subject to availability."
    elif intent == "place_suggestion":
        chat_state.waiting_for_place_suggestion_city = True
        return "Please specify the city you're interested in getting place suggestions for."
    elif intent == "feedback_suggestion":
        return "Thank you for wanting to provide feedback or a suggestion. Please tell me more about it."
    elif intent == "show_bookings":
        # Always show active bookings from chatbot.bookings
        active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
        if active_bookings:
            bookings = "\n".join([
                f"Booking ID: {bid}, City: {b['city'].title()}, Check-in: {b['check_in']}, Check-out: {b['check_out']}, Guests: {b['num_people']}" 
                for bid, b in active_bookings.items()
            ])
            return f"Here are your active bookings:\n{bookings}"
        else:
            return "You have no active bookings."
    elif intent == "cancel":
        # If waiting for booking ID from previous prompt
        if chat_state.pending_cancellation:
            booking_id = user_input.strip().upper()
            response = chatbot.cancel_booking(booking_id)
            chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
            chat_state.pending_cancellation = False
            return response
        # Try to extract booking ID from current message
        booking_id = extract_booking_id(user_input)
        if booking_id:
            response = chatbot.cancel_booking(booking_id)
            chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
            return response
        else:
            chat_state.pending_cancellation = True
            return "Please provide the Booking ID you want to cancel."
    elif intent == "modify":
        # If waiting for modification details
        if chat_state.pending_modification and chat_state.modification_type:
            booking_id = chat_state.pending_modification
            user_message = user_input.strip()
            # Handle if user just says 'guests' or 'dates'
            if user_message in ["guests", "number of guests"]:
                return "Please enter the new number of guests."
            if user_message in ["dates", "date", "check-in", "check-out"]:
                return "Please enter the new check-in and check-out dates in YYYY-MM-DD to YYYY-MM-DD format."
            changes = {}
            # If user enters just a number, treat as guests
            if user_message.isdigit():
                changes['num_people'] = int(user_message)
            # If user enters two dates (e.g., 2025-12-18 to 2025-12-22)
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            dates = re.findall(date_pattern, user_message)
            if len(dates) == 2:
                changes['check_in'] = dates[0]
                changes['check_out'] = dates[1]
            # Fallback to previous extraction logic
            details = extract_booking_details(user_message)
            if details:
                if details.get('check_in'):
                    changes['check_in'] = details['check_in']
                if details.get('check_out'):
                    changes['check_out'] = details['check_out']
                if details.get('num_people'):
                    changes['num_people'] = details['num_people']
            people_pattern = r'(\d+)\s*(?:people|persons|guests)'
            people_match = re.search(people_pattern, user_message.lower())
            if people_match:
                changes['num_people'] = int(people_match.group(1))
            if not changes:
                return "Please specify the new number of guests (e.g., '4') or the new dates (e.g., '2025-12-18 to 2025-12-22')."
            response = chatbot.modify_booking(booking_id, **changes)
            chat_state.active_bookings = {k: v for k, v in chatbot.bookings.items() if v['status'] == 'confirmed'}
            chat_state.pending_modification = None
            chat_state.modification_type = None
            return response
        # If waiting for booking ID
        if chat_state.pending_modification:
            booking_id = user_input.strip().upper()
            if booking_id in chatbot.bookings and chatbot.bookings[booking_id]['status'] == 'confirmed':
                chat_state.pending_modification = booking_id
                chat_state.modification_type = True
                return "What would you like to modify for this booking? (You can provide new dates or number of guests)"
            else:
                chat_state.pending_modification = None
                return "Booking ID not found or not active. Please provide a valid Booking ID."
        # Try to extract booking ID from current message
        booking_id = extract_booking_id(user_input)
        if booking_id and booking_id in chatbot.bookings and chatbot.bookings[booking_id]['status'] == 'confirmed':
            chat_state.pending_modification = booking_id
            chat_state.modification_type = True
            return "What would you like to modify for this booking? (You can provide new dates or number of guests)"
        else:
            chat_state.pending_modification = True
            chat_state.modification_type = None
            return "Please provide the Booking ID you want to modify."

    # 3. Fallback: Use neural model for complex queries or unmatched intents
    response = chatbot.process_message(user_input)

    # Handle emotion/feedback if no specific intent was matched and fallback didn't handle it
    feedback_keywords = [
        'happy', 'sad', 'angry', 'disgust', 'disgusting', 'confused', 'surprised', 'bad', 'good', 'excellent',
        'worst', 'best', 'love', 'hate', 'feedback', 'suggestion', 'improve', 'satisfied', 'unsatisfied',
        'disappointed', 'amazing', 'awesome', 'terrible', 'awful', 'frustrated', 'upset', 'thank you', 'thanks'
    ]
    if intent is None and any(word in user_input for word in feedback_keywords):
        emotion_result = analyze_emotion(user_input)
        if emotion_result:
            chat_state.conversation_emotions.append(emotion_result)
            emotion_response = get_emotion_based_response(emotion_result)
            negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
            if emotion_result.get('emotion') in negative_emotions and not chat_state.feedback_given:
                chat_state.feedback_given = True
                emotion_response += "\nI notice you seem to be experiencing some negative emotions. Would you like to provide feedback on how we can improve our service?"
            # If emotion is detected and no specific intent was matched, return emotion response
            return emotion_response

    # If no specific intent was matched and no emotion was detected (or emotion didn't trigger a specific response), return neural model response
    return response

def main():
    print("Hi, I am Questa, your friend in need for hotel assistant!")
    print("Type 'quit' to exit\n")
    print("You can:")
    print("1. Ask questions about hotel policies")
    print("2. Book a hotel room")
    print("3. Cancel or modify your booking")
    print("4. Get suggestions for places to visit")
    print("5. View your booking details\n")
    print("Example booking message:")
    print("I want to book a hotel in London for 2 people from 2024-03-01 to 2024-03-05")
    print("\nTo view your bookings, type 'show my bookings'")
    print("To cancel a booking, type 'cancel' followed by your booking ID")
    
    # Load model and vocabulary
    chatbot = load_model_and_vocabulary()
    if chatbot is None:
        print("\nError: Could not load model. Please train the model first using train_chatbot.py")
        return
    
    # Initialize chat state
    chat_state = ChatState()
    
    # Start chat loop
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Process user input and get response
        response = process_user_input(user_input, chatbot, chat_state)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    tf.random.set_seed(42)  # Set random seed for deterministic responses
    main() 