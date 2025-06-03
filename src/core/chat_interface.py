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
        self.waiting_for_booking_id = False  # New state to track when we're waiting for a booking ID

def generate_booking_id():
    """Generate a unique booking ID"""
    return str(uuid.uuid4())[:8].upper()  # Using first 8 characters for simplicity

def extract_booking_id(message):
    """Extract booking ID from user message"""
    # Match 8-character alphanumeric booking ID
    pattern = r'[A-Z0-9]{8}'
    match = re.search(pattern, message.upper())
    return match.group(0) if match else None

def process_user_input(user_input, chatbot, chat_state):
    """Process user input and generate appropriate response"""
    user_input = user_input.lower().strip()
    
    # Handle empty input
    if not user_input:
        return "Please enter a message."

    # 1. Place suggestions (MUST be first to take precedence)
    if any(word in user_input for word in ['suggest', 'recommend', 'visit', 'places', 'attractions', 'explore']) or \
       (chat_state.last_response and 'Which city would you like recommendations for?' in chat_state.last_response):
        # Extract city from input
        cities = ['london', 'paris', 'new york', 'tokyo', 'sydney', 'chennai', 'mumbai', 'delhi', 'bangalore']
        suggested_city = None
        
        # First check if a city is mentioned in the input
        for city in cities:
            if city in user_input.lower():
                suggested_city = city
                break
        
        if suggested_city:
            if suggested_city == 'london':
                chat_state.last_response = None
                return "Top attractions in London:\n" \
                       "1. Big Ben and Houses of Parliament\n" \
                       "2. Buckingham Palace\n" \
                       "3. British Museum\n" \
                       "4. Tower of London\n" \
                       "5. London Eye"
            elif suggested_city == 'paris':
                chat_state.last_response = None
                return "Must-visit places in Paris:\n" \
                       "1. Eiffel Tower\n" \
                       "2. Louvre Museum\n" \
                       "3. Notre-Dame Cathedral\n" \
                       "4. Champs-Élysées\n" \
                       "5. Montmartre"
            elif suggested_city == 'delhi':
                chat_state.last_response = None
                return "Top attractions in Delhi:\n" \
                       "1. Red Fort (Lal Qila)\n" \
                       "2. India Gate\n" \
                       "3. Qutub Minar\n" \
                       "4. Humayun's Tomb\n" \
                       "5. Lotus Temple\n" \
                       "6. Chandni Chowk (Old Delhi)\n" \
                       "7. Akshardham Temple\n" \
                       "8. National Museum\n" \
                       "9. Connaught Place\n" \
                       "10. Jama Masjid"
            elif suggested_city == 'mumbai':
                chat_state.last_response = None
                return "Must-visit places in Mumbai:\n" \
                       "1. Gateway of India\n" \
                       "2. Marine Drive\n" \
                       "3. Elephanta Caves\n" \
                       "4. Juhu Beach\n" \
                       "5. Colaba Causeway\n" \
                       "6. Bandra-Worli Sea Link\n" \
                       "7. Chhatrapati Shivaji Terminus\n" \
                       "8. Haji Ali Dargah\n" \
                       "9. National Museum of Indian Cinema\n" \
                       "10. Crawford Market"
            elif suggested_city == 'bangalore':
                chat_state.last_response = None
                return "Top attractions in Bangalore:\n" \
                       "1. Lalbagh Botanical Garden\n" \
                       "2. Cubbon Park\n" \
                       "3. Bangalore Palace\n" \
                       "4. Tipu Sultan's Summer Palace\n" \
                       "5. ISKCON Temple\n" \
                       "6. Vidhana Soudha\n" \
                       "7. Commercial Street\n" \
                       "8. MG Road\n" \
                       "9. Wonderla Amusement Park\n" \
                       "10. Bannerghatta National Park"
            elif suggested_city == 'chennai':
                chat_state.last_response = None
                return "Must-visit places in Chennai:\n" \
                       "1. Marina Beach\n" \
                       "2. Kapaleeshwarar Temple\n" \
                       "3. Fort St. George\n" \
                       "4. Santhome Cathedral\n" \
                       "5. Government Museum\n" \
                       "6. Elliot's Beach\n" \
                       "7. Guindy National Park\n" \
                       "8. Valluvar Kottam\n" \
                       "9. Birla Planetarium\n" \
                       "10. T Nagar Shopping District"
        else:
            response = "I can suggest places to visit in:\n" \
                      "- London\n" \
                      "- Paris\n" \
                      "- New York\n" \
                      "- Tokyo\n" \
                      "- Sydney\n" \
                      "- Chennai\n" \
                      "- Mumbai\n" \
                      "- Delhi\n" \
                      "- Bangalore\n" \
                      "Which city would you like recommendations for?"
            chat_state.last_response = response
            return response

    # Check if we're waiting for a booking ID
    if chat_state.waiting_for_booking_id:
        booking_id = user_input.upper()
        if re.match(r'^[A-Z0-9]{8}$', booking_id):
            if booking_id in chat_state.active_bookings:
                if chat_state.pending_cancellation:
                    chat_state.pending_cancellation = booking_id
                    chat_state.waiting_for_booking_id = False
                    return f"Are you sure you want to cancel booking {booking_id}?\n" \
                           f"Please confirm with 'yes' or 'no'"
                elif chat_state.pending_modification:
                    chat_state.pending_modification = booking_id
                    chat_state.waiting_for_booking_id = False
                    return f"What would you like to modify for booking {booking_id}?\n" \
                           f"Options: dates, number of guests, or city"
            else:
                chat_state.waiting_for_booking_id = False
                return f"Sorry, I couldn't find a booking with ID {booking_id}"
        else:
            chat_state.waiting_for_booking_id = False
            return "Please provide a valid booking ID (8 characters, letters and numbers only)."

    # Handle cancellation confirmation (state-dependent)
    if chat_state.pending_cancellation:
        if any(word in user_input for word in ['yes', 'confirm', 'proceed']):
            booking_id = chat_state.pending_cancellation
            if booking_id in chat_state.active_bookings:
                del chat_state.active_bookings[booking_id]
            chat_state.pending_cancellation = None
            return f"Your booking {booking_id} has been cancelled successfully."
        elif any(word in user_input for word in ['no', 'cancel', 'keep', 'maybe']):
            booking_id = chat_state.pending_cancellation
            chat_state.pending_cancellation = None
            return f"Cancellation cancelled. Your booking {booking_id} remains active."
        else:
            return "Please respond with 'yes' to confirm cancellation or 'no' to keep the booking."

    # Handle modification confirmation (state-dependent)
    if chat_state.pending_modification:
        booking_id = chat_state.pending_modification
        booking = chat_state.active_bookings.get(booking_id)
        
        if not booking:
            chat_state.pending_modification = None
            return f"Sorry, booking {booking_id} no longer exists."
            
        if 'dates' in user_input:
            dates = extract_booking_details(user_input)
            if dates and dates.get('check_in') and dates.get('check_out'):
                chat_state.active_bookings[booking_id]['check_in'] = dates['check_in']
                chat_state.active_bookings[booking_id]['check_out'] = dates['check_out']
                chat_state.pending_modification = None
                return f"Booking {booking_id} dates updated successfully:\n" \
                       f"New check-in: {dates['check_in']}\n" \
                       f"New check-out: {dates['check_out']}\n\n" \
                       f"Updated booking details:\n" \
                       f"- City: {booking['city'].title()}\n" \
                       f"- Check-in: {dates['check_in']}\n" \
                       f"- Check-out: {dates['check_out']}\n" \
                       f"- Number of guests: {booking['num_people']}"
            else:
                return "Please provide new dates in the format: YYYY-MM-DD to YYYY-MM-DD"
                
        elif 'guests' in user_input or 'people' in user_input:
            people_match = re.search(r'(\d+)\s*(?:people|persons|guests)?', user_input)
            if people_match:
                new_guest_count = int(people_match.group(1))
                chat_state.active_bookings[booking_id]['num_people'] = new_guest_count
                chat_state.pending_modification = None
                return f"Booking {booking_id} updated successfully:\n" \
                       f"New number of guests: {new_guest_count}\n\n" \
                       f"Updated booking details:\n" \
                       f"- City: {booking['city'].title()}\n" \
                       f"- Check-in: {booking['check_in']}\n" \
                       f"- Check-out: {booking['check_out']}\n" \
                       f"- Number of guests: {new_guest_count}"
            else:
                return "Please specify the new number of guests (e.g., 'change to 3 people')"
                
        elif 'city' in user_input or any(city in user_input.lower() for city in ['london', 'paris', 'new york', 'tokyo', 'sydney', 'chennai', 'mumbai', 'delhi', 'bangalore']):
            cities = ['london', 'paris', 'new york', 'tokyo', 'sydney', 'chennai', 'mumbai', 'delhi', 'bangalore']
            new_city = None
            for city in cities:
                if city in user_input.lower():
                    new_city = city
                    break
            if new_city:
                chat_state.active_bookings[booking_id]['city'] = new_city
                chat_state.pending_modification = None
                return f"Booking {booking_id} updated successfully:\n" \
                       f"New city: {new_city.title()}\n\n" \
                       f"Updated booking details:\n" \
                       f"- City: {new_city.title()}\n" \
                       f"- Check-in: {booking['check_in']}\n" \
                       f"- Check-out: {booking['check_out']}\n" \
                       f"- Number of guests: {booking['num_people']}"
            else:
                return "Please specify a valid city from our list:\n" \
                       "- London\n" \
                       "- Paris\n" \
                       "- New York\n" \
                       "- Tokyo\n" \
                       "- Sydney\n" \
                       "- Chennai\n" \
                       "- Mumbai\n" \
                       "- Delhi\n" \
                       "- Bangalore"
        elif any(word in user_input for word in ['show', 'view', 'list', 'my bookings']):
            chat_state.pending_modification = None  # Reset modification state
            if not chat_state.active_bookings:
                return "You don't have any active bookings."
            
            response = "Your active bookings:\n"
            for bid, b in chat_state.active_bookings.items():
                response += f"\nBooking ID: {bid}\n" \
                           f"- City: {b['city'].title()}\n" \
                           f"- Check-in: {b['check_in']}\n" \
                           f"- Check-out: {b['check_out']}\n" \
                           f"- Number of guests: {b['num_people']}\n"
            return response
        else:
            return "What would you like to modify?\n" \
                   "Options: dates, number of guests, or city"

    # Handle booking confirmation (state-dependent)
    if chat_state.pending_booking and any(word in user_input for word in ['yes', 'confirm', 'correct', 'proceed']):
        booking = chat_state.pending_booking
        booking_id = generate_booking_id()
        chat_state.active_bookings[booking_id] = booking
        chat_state.pending_booking = None
        return f"Booking confirmed!\n" \
               f"Your booking ID is: {booking_id}\n" \
               f"Please save this ID for future reference.\n\n" \
               f"Booking details:\n" \
               f"- City: {booking['city'].title()}\n" \
               f"- Check-in: {booking['check_in']}\n" \
               f"- Check-out: {booking['check_out']}\n" \
               f"- Number of guests: {booking['num_people']}"

    # 1. View bookings query
    if any(phrase in user_input for phrase in ['show my bookings', 'view my bookings', 'list my bookings', 
                                             'my bookings', 'show bookings', 'view bookings', 'list bookings',
                                             'show booking', 'view booking', 'list booking']):
        if not chat_state.active_bookings:
            return "You don't have any active bookings."
        
        response = "Your active bookings:\n"
        for booking_id, booking in chat_state.active_bookings.items():
            response += f"\nBooking ID: {booking_id}\n" \
                       f"- City: {booking['city'].title()}\n" \
                       f"- Check-in: {booking['check_in']}\n" \
                       f"- Check-out: {booking['check_out']}\n" \
                       f"- Number of guests: {booking['num_people']}\n"
        return response

    # 2. Cancellation queries (MUST be before booking queries)
    if any(word in user_input for word in ['cancel', 'cancellation', 'delete']):
        # First check if there's a booking ID in the input
        booking_id_match = re.search(r'[A-Z0-9]{8}', user_input.upper())
        if booking_id_match:
            booking_id = booking_id_match.group(0)
            if booking_id in chat_state.active_bookings:
                chat_state.pending_cancellation = booking_id
                return f"Are you sure you want to cancel booking {booking_id}?\n" \
                       f"Please confirm with 'yes' or 'no'"
            else:
                return f"Sorry, I couldn't find a booking with ID {booking_id}"
        else:
            chat_state.waiting_for_booking_id = True
            chat_state.pending_cancellation = True  # Set a flag to indicate we're waiting for cancellation
            return "Please provide your booking ID to cancel your reservation."

    # 3. Modification queries (MUST be before booking queries)
    if any(word in user_input for word in ['modify', 'change', 'update', 'edit']):
        # First check if there's a booking ID in the input
        booking_id_match = re.search(r'[A-Z0-9]{8}', user_input.upper())
        if booking_id_match:
            booking_id = booking_id_match.group(0)
            if booking_id in chat_state.active_bookings:
                chat_state.pending_modification = booking_id
                return f"What would you like to modify for booking {booking_id}?\n" \
                       f"Options: dates, number of guests, or city"
            else:
                return f"Sorry, I couldn't find a booking with ID {booking_id}"
        else:
            chat_state.waiting_for_booking_id = True
            chat_state.pending_modification = True  # Set a flag to indicate we're waiting for modification
            return "Please provide your booking ID to modify your reservation."

    # 4. Booking related queries
    if any(word in user_input for word in ['book', 'booking', 'reserve', 'reservation']):
        booking_details = extract_booking_details(user_input)
        if booking_details:
            chat_state.pending_booking = booking_details
            return f"I found these booking details:\n" \
                   f"- City: {booking_details['city'].title()}\n" \
                   f"- Check-in: {booking_details['check_in']}\n" \
                   f"- Check-out: {booking_details['check_out']}\n" \
                   f"- Number of guests: {booking_details['num_people']}\n\n" \
                   f"Would you like to confirm this booking?"
        else:
            return "To book a hotel, please provide:\n" \
                   "- City (e.g., London, Paris, New York)\n" \
                   "- Check-in date (YYYY-MM-DD)\n" \
                   "- Check-out date (YYYY-MM-DD)\n" \
                   "- Number of guests\n\n" \
                   "Example: I want to book a hotel in London for 2 people from 2024-03-01 to 2024-03-05"

    # 5. Children related queries
    if any(word in user_input for word in ['child', 'children', 'kid', 'kids', 'baby', 'babies']):
        if 'policy' in user_input or 'age' in user_input:
            return "Children under 12 stay free when sharing existing bedding. Extra beds are available for children aged 12-17 at an additional charge."
        elif 'play' in user_input or 'activity' in user_input:
            return "We offer various children's activities including:\n" \
                   "- Kids' play area\n" \
                   "- Swimming pool with children's section\n" \
                   "- Board games and video games\n" \
                   "- Daily supervised activities (ages 4-12)"
        else:
            return "For children, we offer:\n" \
                   "- Free stay for under 12s\n" \
                   "- Kids' play area\n" \
                   "- Children's menu\n" \
                   "- Babysitting services (additional charge)\n" \
                   "Would you like more specific information?"

    # 6. Amenities/Facilities queries
    if any(word in user_input for word in ['amenity', 'amenities', 'facility', 'facilities', 'wifi', 'pool', 'gym', 'spa']):
        if 'wifi' in user_input:
            return "Yes, we provide complimentary high-speed WiFi throughout the hotel, including all rooms and public areas."
        elif 'pool' in user_input:
            return "We have both indoor and outdoor swimming pools. The indoor pool is heated and available year-round."
        elif 'gym' in user_input:
            return "Our 24/7 fitness center includes:\n" \
                   "- Modern cardio equipment\n" \
                   "- Weight training area\n" \
                   "- Yoga studio\n" \
                   "- Personal training available"
        elif 'spa' in user_input:
            return "Our spa offers:\n" \
                   "- Massage treatments\n" \
                   "- Facial treatments\n" \
                   "- Sauna and steam room\n" \
                   "- Beauty services"
        else:
            return "Our hotel amenities include:\n" \
                   "- Free WiFi\n" \
                   "- Swimming pools (indoor & outdoor)\n" \
                   "- 24/7 Fitness center\n" \
                   "- Spa and wellness center\n" \
                   "- Restaurant and bar\n" \
                   "- Business center\n" \
                   "- Room service\n" \
                   "Would you like details about any specific amenity?"

    # 7. Hotel Policies
    if any(word in user_input for word in ['policy', 'policies', 'rules', 'terms', 'conditions']):
        if 'check' in user_input and ('in' in user_input or 'out' in user_input):
            return "Check-in and Check-out Policies:\n" \
                   "- Check-in time: 3:00 PM\n" \
                   "- Check-out time: 11:00 AM\n" \
                   "- Early check-in and late check-out available upon request (subject to availability)\n" \
                   "- Valid ID and credit card required at check-in"
        elif 'cancellation' in user_input:
            return "Cancellation Policy:\n" \
                   "- Free cancellation up to 24 hours before check-in\n" \
                   "- Late cancellation or no-show: One night's stay will be charged\n" \
                   "- Special rates and packages may have different cancellation policies"
        elif 'payment' in user_input:
            return "Payment Policies:\n" \
                   "- We accept all major credit cards\n" \
                   "- A valid credit card is required to guarantee your reservation\n" \
                   "- Pre-payment may be required for certain rates\n" \
                   "- Additional charges may apply for extra services"
        elif 'pet' in user_input:
            return "Pet Policy:\n" \
                   "- Pets are welcome in designated pet-friendly rooms\n" \
                   "- Pet fee: $50 per stay\n" \
                   "- Maximum 2 pets per room\n" \
                   "- Pets must be leashed in public areas"
        elif 'dining' in user_input or 'food' in user_input:
            return "Dining Policies:\n" \
                   "- 24/7 room service available\n" \
                   "- Breakfast served 6:00 AM - 10:00 AM\n" \
                   "- Restaurant open 11:00 AM - 11:00 PM\n" \
                   "- Special dietary requirements can be accommodated with advance notice"
        else:
            return "Our main hotel policies include:\n" \
                   "1. Check-in/Check-out\n" \
                   "2. Cancellation\n" \
                   "3. Payment\n" \
                   "4. Pet Policy\n" \
                   "5. Dining\n" \
                   "Which policy would you like to know more about?"

    # For all other cases, use the neural model's response
    neural_response = chatbot.process_message(user_input)
    return neural_response

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