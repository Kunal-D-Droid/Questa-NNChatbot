import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, models, preprocessing
import pickle
import json
import datetime
import uuid
import re
from typing import Dict, List, Tuple, Optional

class HotelBookingChatbot:
    def __init__(self):
        self.encoder_model = None
        self.decoder_model = None
        self.word2idx = {}
        self.idx2word = {}
        self.max_sequence_length = 30  # Increased for longer conversations
        self.embedding_dim = 256  # Increased for better representation
        self.lstm_units = 512  # Increased for better learning
        self.dropout_rate = 0.3
        self.bookings = {}
        self.faq_data = self._load_faq_data()
        
    def _load_faq_data(self) -> Dict:
        """Load FAQ data from JSON file"""
        return {
            "check_in_time": "Check-in time is 3:00 PM. Early check-in may be available upon request.",
            "check_out_time": "Check-out time is 11:00 AM. Late check-out may be available for an additional fee.",
            "cancellation_policy": "Free cancellation up to 24 hours before check-in. After that, a one-night charge may apply.",
            "payment_methods": "We accept credit cards, debit cards, and PayPal for payment.",
            "amenities": "Our hotel offers free WiFi, swimming pool, gym, breakfast, restaurant, spa, and business center.",
            "pet_policy": "Pets are welcome with a $50 cleaning fee per stay. Please inform us in advance.",
            "parking": "On-site parking is available for $20 per day.",
            "room_service": "24-hour room service is available.",
            "breakfast": "Breakfast is served daily from 6:30 AM to 10:00 AM in our main restaurant."
        }
    
    def build_model(self):
        """Build an enhanced encoder-decoder model with attention"""
        # Disable oneDNN operations
        tf.config.optimizer.set_jit(False)
        
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(self.max_sequence_length,), name='encoder_inputs')
        encoder_embedding = tf.keras.layers.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        
        # Bidirectional LSTM for encoder
        encoder_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                return_state=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='encoder_lstm'
            ),
            name='bidirectional_encoder'
        )
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        
        # Project encoder outputs to match decoder embedding dimension
        encoder_outputs = tf.keras.layers.Dense(
            self.embedding_dim,
            activation='tanh',
            name='encoder_projection'
        )(encoder_outputs)
        
        # Combine forward and backward states
        state_h = tf.keras.layers.Concatenate(name='state_h_concat')([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate(name='state_c_concat')([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = tf.keras.layers.Input(shape=(self.max_sequence_length,), name='decoder_inputs')
        decoder_embedding = tf.keras.layers.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        
        # Simple attention mechanism
        attention = tf.keras.layers.Dot(axes=[2, 2], name='attention_dot')([decoder_embedding, encoder_outputs])
        attention = tf.keras.layers.Softmax(axis=-1, name='attention_softmax')(attention)
        context = tf.keras.layers.Dot(axes=[2, 1], name='context_dot')([attention, encoder_outputs])
        
        # Concatenate decoder embedding with context vector
        decoder_concat = tf.keras.layers.Concatenate(name='decoder_concat')([decoder_embedding, context])
        
        # Decoder LSTM
        decoder_lstm = tf.keras.layers.LSTM(
            self.lstm_units * 2,  # Double size to match concatenated states
            return_sequences=True,
            return_state=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_concat,
            initial_state=encoder_states
        )
        
        # Additional dense layers with regularization
        decoder_dense1 = tf.keras.layers.Dense(
            self.lstm_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name='decoder_dense1'
        )
        decoder_dense2 = tf.keras.layers.Dense(
            len(self.word2idx),
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name='decoder_dense2'
        )
        
        # Apply dense layers
        x = decoder_dense1(decoder_outputs)
        x = tf.keras.layers.Dropout(0.2, name='decoder_dropout')(x)
        decoder_outputs = decoder_dense2(x)

        # Training model
        self.model = tf.keras.Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            name='encoder_decoder_model'
        )

        # Inference models
        self.encoder_model = tf.keras.Model(
            encoder_inputs,
            [encoder_outputs, encoder_states],
            name='encoder_model'
        )

        decoder_state_input_h = tf.keras.layers.Input(shape=(self.lstm_units * 2,), name='decoder_state_input_h')
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.lstm_units * 2,), name='decoder_state_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        # Simple attention for inference
        attention = tf.keras.layers.Dot(axes=[2, 2], name='inference_attention_dot')([decoder_embedding, encoder_outputs])
        attention = tf.keras.layers.Softmax(axis=-1, name='inference_attention_softmax')(attention)
        context = tf.keras.layers.Dot(axes=[2, 1], name='inference_context_dot')([attention, encoder_outputs])
        
        decoder_concat = tf.keras.layers.Concatenate(name='inference_decoder_concat')([decoder_embedding, context])
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_concat,
            initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        
        # Apply dense layers for inference
        x = decoder_dense1(decoder_outputs)
        x = tf.keras.layers.Dropout(0.2, name='inference_decoder_dropout')(x)
        decoder_outputs = decoder_dense2(x)
        
        self.decoder_model = tf.keras.Model(
            [decoder_inputs, encoder_outputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states,
            name='decoder_model'
        )

        # Compile model with custom learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model architecture: Enhanced Encoder-Decoder LSTM with Simple Attention")

    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to sequence of word indices"""
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

    def generate_response(self, input_text: str, temperature: float = 0.7) -> str:
        """Generate response using the encoder-decoder model"""
        # Preprocess input
        input_seq = self.preprocess_text(input_text)
        input_seq = preprocessing.sequence.pad_sequences([input_seq], maxlen=self.max_sequence_length, padding='post')
        
        # Get encoder states
        # states_value from encoder_model.predict is [encoder_outputs, [state_h, state_c]]
        encoder_outputs, states_value = self.encoder_model.predict(input_seq)

        # Initialize decoder input with start token
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx['<START>']

        # Generate response
        response = []
        for _ in range(self.max_sequence_length):
            # Get decoder output and new states
            # decoder_model expects [decoder_inputs, encoder_outputs] + decoder_states_inputs
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_outputs] + states_value
            )

            # Sample next token
            output_tokens = output_tokens[0, -1, :]
            output_tokens = output_tokens / temperature
            output_tokens = np.exp(output_tokens) / np.sum(np.exp(output_tokens))
            
            # Get next token
            sampled_token_index = np.argmax(output_tokens)
            sampled_word = self.idx2word[sampled_token_index]
            
            # Stop if we reach the end token
            if sampled_word == '<END>':
                break
                
            # Add word to response
            if sampled_word not in ['<PAD>', '<UNK>']:
                response.append(sampled_word)
            
            # Update target sequence and states for the next step
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c] # Update states for the next iteration
        
        return ' '.join(response)

    def handle_booking(self, check_in: str, check_out: str, city: str, num_people: int) -> str:
        """Handle hotel booking request"""
        try:
            check_in_date = datetime.datetime.strptime(check_in, '%Y-%m-%d')
            check_out_date = datetime.datetime.strptime(check_out, '%Y-%m-%d')
            current_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if check_in_date < current_date:
                return "Check-in date cannot be in the past"
            
            if check_in_date >= check_out_date:
                return "Check-out date must be after check-in date"
            
            if num_people < 1 or num_people > 10:
                return "Number of people must be between 1 and 10"
            
            booking_id = str(uuid.uuid4())[:8].upper()
            self.bookings[booking_id] = {
                'check_in': check_in,
                'check_out': check_out,
                'city': city,
                'num_people': num_people,
                'status': 'confirmed',
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return f"""Booking confirmed! Your booking ID is: {booking_id}

Booking details:
- City: {city}
- Check-in: {check_in}
- Check-out: {check_out}
- Number of guests: {num_people}

Please save your booking ID for future reference."""
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD format"

    def cancel_booking(self, booking_id: str) -> str:
        """Cancel a hotel booking"""
        if booking_id in self.bookings:
            if self.bookings[booking_id]['status'] == 'cancelled':
                return f"Booking {booking_id} is already cancelled"
            
            check_in_date = datetime.datetime.strptime(self.bookings[booking_id]['check_in'], '%Y-%m-%d')
            current_date = datetime.datetime.now()
            
            if (check_in_date - current_date).total_seconds() < 86400:  # 24 hours
                return "Cancellation is not possible within 24 hours of check-in"
            
            self.bookings[booking_id]['status'] = 'cancelled'
            return f"Booking {booking_id} has been cancelled successfully"
        return "Booking ID not found"

    def modify_booking(self, booking_id: str, **kwargs) -> str:
        """Modify an existing booking"""
        if booking_id not in self.bookings:
            return "Booking ID not found"
            
        booking = self.bookings[booking_id]
        if booking['status'] == 'cancelled':
            return "Cannot modify a cancelled booking"
        
        changes = {}
        for key, value in kwargs.items():
            if key in booking:
                if key in ['check_in', 'check_out']:
                    try:
                        new_date = datetime.datetime.strptime(value, '%Y-%m-%d')
                        if key == 'check_in':
                            current_date = datetime.datetime.now()
                            if new_date < current_date:
                                return "Check-in date cannot be in the past"
                            if new_date >= datetime.datetime.strptime(booking['check_out'], '%Y-%m-%d'):
                                return "Check-out date must be after check-in date"
                        else:  # check_out
                            if new_date <= datetime.datetime.strptime(booking['check_in'], '%Y-%m-%d'):
                                return "Check-out date must be after check-in date"
                    except ValueError:
                        return "Invalid date format. Please use YYYY-MM-DD format"
                elif key == 'num_people':
                    if not (1 <= value <= 10):
                        return "Number of people must be between 1 and 10"
                changes[key] = value
                
        if not changes:
            return "No valid changes provided"
            
        booking.update(changes)
        return f"""Booking {booking_id} has been modified successfully.

Updated booking details:
- City: {booking['city']}
- Check-in: {booking['check_in']}
- Check-out: {booking['check_out']}
- Number of guests: {booking['num_people']}
- Status: {booking['status']}"""

    def get_place_suggestions(self, city: str) -> List[str]:
        """Get place suggestions for a city"""
        suggestions = {
            'delhi': [
                'Red Fort',
                'India Gate',
                'Qutub Minar',
                'Lotus Temple',
                'Humayun\'s Tomb',
                'Chandni Chowk',
                'Akshardham Temple'
            ],
            'mumbai': [
                'Gateway of India',
                'Marine Drive',
                'Elephanta Caves',
                'Juhu Beach',
                'Colaba Causeway',
                'Haji Ali Dargah',
                'Sanjay Gandhi National Park'
            ],
            'bangalore': [
                'Lalbagh Botanical Garden',
                'Bangalore Palace',
                'Cubbon Park',
                'Vidhana Soudha',
                'ISKCON Temple',
                'UB City',
                'Wonderla Amusement Park'
            ],
            'london': [
                'Big Ben and Houses of Parliament',
                'London Eye',
                'Buckingham Palace',
                'Tower of London',
                'British Museum',
                'Tower Bridge',
                'Hyde Park'
            ],
            'paris': [
                'Eiffel Tower',
                'Louvre Museum',
                'Notre-Dame Cathedral',
                'Champs-Élysées',
                'Arc de Triomphe',
                'Montmartre',
                'Palace of Versailles'
            ]
        }
        return suggestions.get(city.lower(), ["No suggestions available for this city"])

    def _is_quality_response(self, response: str) -> bool:
        """Check if the neural model's response is of acceptable quality"""
        # Check for repetitive words
        words = response.lower().split()
        if len(words) < 3:  # Too short
            return False
            
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] >= 3:  # Word appears 3 or more times
                return False
                
        # Check for common incoherent patterns
        incoherent_patterns = [
            'need need',
            'can can',
            'Suite Suite',
            'details details',
            'dates dates'
        ]
        for pattern in incoherent_patterns:
            if pattern in response.lower():
                return False
                
        return True

    def process_message(self, message: str) -> str:
        """Process user message and generate response"""
        message = message.lower().strip()
        print("\n[DEBUG] Processing message:", message)
        
        # Handle greetings
        if self._is_greeting(message):
            return self._handle_greeting()
        
        # Handle FAQ queries first
        if self._is_faq_query(message):
            return self._handle_faq_query(message)
        
        # Handle booking status queries
        if self._is_booking_status_query(message):
            booking_id = self._extract_booking_id(message)
            if booking_id:
                return self._get_booking_status(booking_id)
            return "Please provide your booking ID to check the status."
        
        # Handle booking requests
        if self._is_booking_query(message):
            booking_info = self._extract_booking_info(message)
            if booking_info:
                return self._handle_booking_request(booking_info)
            return "To make a booking, please provide:\n" \
                   "- Check-in date (YYYY-MM-DD)\n" \
                   "- Check-out date (YYYY-MM-DD)\n" \
                   "- City\n" \
                   "- Number of people\n" \
                   "Example: I want to book a hotel in London for 2 people from 2024-03-01 to 2024-03-05"
        
        # Handle cancellation requests
        if self._is_cancellation_query(message):
            booking_id = self._extract_booking_id(message)
            if booking_id:
                return self.cancel_booking(booking_id)
            return "Please provide your booking ID to cancel the reservation. You can view your booking IDs by typing 'show my bookings'"
        
        # Handle modification requests
        if self._is_modification_query(message):
            booking_id = self._extract_booking_id(message)
            if booking_id:
                return self._handle_modification_request(booking_id, message)
            return "Please provide your booking ID to modify the reservation. You can view your booking IDs by typing 'show my bookings'"
        
        # Handle place suggestions
        if self._is_place_suggestion_query(message):
            city = self._extract_city(message)
            if city:
                suggestions = self.get_place_suggestions(city)
                return f"Here are some popular attractions in {city.title()}:\n" + "\n".join(f"- {s}" for s in suggestions)
            return "Please specify which city's attractions you'd like to know about."
        
        # For complex queries that don't match any specific intent,
        # use the neural model's response
        try:
            print("[DEBUG] Attempting neural response for complex query")
            response = self.generate_response(message)
            print(f"[DEBUG] Neural model raw response: {response}")
            
            if response and not response.isspace() and self._is_quality_response(response):
                print("[DEBUG] Using neural model response - passed quality check")
                return response
            else:
                print("[DEBUG] Neural response failed quality check - using fallback")
        except Exception as e:
            print(f"[DEBUG] Error generating neural response: {str(e)}")
        
        # Fallback response if neural model fails
        return "I'm not sure I understand. I can help you with:\n" \
               "1. Making hotel bookings\n" \
               "2. Cancelling or modifying bookings\n" \
               "3. Checking booking status\n" \
               "4. Providing information about hotel policies\n" \
               "5. Suggesting places to visit\n" \
               "Please let me know what you need help with."

    def _classify_intent(self, message: str) -> str:
        """Classify the intent of the message using the LSTM model"""
        # Preprocess the message
        input_seq = self.preprocess_text(message)
        input_seq = preprocessing.sequence.pad_sequences([input_seq], maxlen=self.max_sequence_length, padding='post')
        
        # Get the model's prediction
        _, h, c = self.encoder_model.predict(input_seq)
        
        # Generate response to understand intent
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx['<START>']
        
        response = []
        for _ in range(self.max_sequence_length):
            output_tokens, h, c = self.decoder_model.predict([target_seq, h, c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.idx2word[sampled_token_index]
            
            if sampled_word == '<END>':
                break
            
            response.append(sampled_word)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
        
        response_text = ' '.join(response)
        print("[DEBUG] Intent classification response:", response_text)
        
        # More strict intent classification
        if any(word in message.lower() for word in ['book', 'reservation', 'stay']) and \
           any(word in message.lower() for word in ['room', 'hotel', 'accommodation']):
            return 'booking'
        elif any(word in message.lower() for word in ['cancel', 'cancellation']) and \
             any(word in message.lower() for word in ['booking', 'reservation']):
            return 'cancellation'
        elif any(word in message.lower() for word in ['modify', 'change', 'update']) and \
             any(word in message.lower() for word in ['booking', 'reservation']):
            return 'modification'
        elif any(word in message.lower() for word in ['attraction', 'place', 'visit']) and \
             any(word in message.lower() for word in ['city', 'town', 'location']):
            return 'place_suggestions'
        
        return 'unknown'

    def _extract_booking_info(self, message: str) -> Dict:
        """Extract booking information from message"""
        info = {}
        
        # Extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, message)
        if len(dates) >= 2:
            info['check_in'] = dates[0]
            info['check_out'] = dates[1]
        
        # Extract number of people
        people_pattern = r'(\d+)\s*(?:people|guests|persons|person)'
        people_match = re.search(people_pattern, message)
        if people_match:
            info['num_people'] = int(people_match.group(1))
        
        # Extract city
        cities = ['delhi', 'mumbai', 'bangalore', 'london', 'paris', 'new york', 'tokyo', 'sydney', 'rome', 'berlin']
        for city in cities:
            if city in message:
                info['city'] = city
                break
        
        return info

    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        return any(greeting in message for greeting in greetings)

    def _handle_greeting(self) -> str:
        """Handle greeting messages"""
        return """Hello! I'm your hotel booking assistant. I can help you with:
1. Making hotel bookings
2. Cancelling or modifying bookings
3. Checking booking status
4. Providing information about hotel policies
5. Suggesting places to visit
How can I assist you today?"""

    def _is_booking_query(self, message: str) -> bool:
        """Check if message is about booking"""
        booking_keywords = ['book', 'booking', 'reserve', 'reservation', 'stay']
        return any(keyword in message for keyword in booking_keywords)

    def _handle_booking_request(self, booking_info: Dict) -> str:
        """Handle booking request with extracted information"""
        required_fields = ['check_in', 'check_out', 'city', 'num_people']
        missing_fields = [field for field in required_fields if field not in booking_info]
        
        if missing_fields:
            return f"Please provide the following information: {', '.join(missing_fields)}"
        
        return self.handle_booking(
            booking_info['check_in'],
            booking_info['check_out'],
            booking_info['city'],
            booking_info['num_people']
        )

    def _is_cancellation_query(self, message: str) -> bool:
        """Check if message is about cancellation"""
        cancel_keywords = ['cancel', 'cancellation', 'cancel my booking', 'cancel reservation']
        return any(keyword in message for keyword in cancel_keywords)

    def _is_modification_query(self, message: str) -> bool:
        """Check if message is about modification"""
        modify_keywords = ['modify', 'change', 'update', 'alter', 'edit']
        return any(keyword in message for keyword in modify_keywords)

    def _handle_modification_request(self, booking_id: str, message: str) -> str:
        """Handle booking modification request"""
        if not booking_id:
            return "Please provide your booking ID to modify the reservation. You can view your booking IDs by typing 'show my bookings'."
        
        changes = {}
        
        # Extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, message)
        if len(dates) >= 2:
            changes['check_in'] = dates[0]
            changes['check_out'] = dates[1]
        
        # Extract number of people
        people_pattern = r'(\d+)\s*(?:people|guests|persons|person)'
        people_match = re.search(people_pattern, message)
        if people_match:
            changes['num_people'] = int(people_match.group(1))
        
        # Extract city
        cities = ['delhi', 'mumbai', 'bangalore', 'london', 'paris', 'new york', 'tokyo', 'sydney', 'rome', 'berlin']
        for city in cities:
            if city in message:
                changes['city'] = city
                break
        
        if not changes:
            return f"""Please specify what changes you want to make to booking {booking_id}.
You can modify:
- Check-in date (YYYY-MM-DD)
- Check-out date (YYYY-MM-DD)
- Number of people
- City"""
        
        return self.modify_booking(booking_id, **changes)

    def _is_place_suggestion_query(self, message: str) -> bool:
        """Check if message is about place suggestions"""
        suggestion_keywords = ['suggest', 'places', 'attractions', 'visit', 'see', 'tourist', 'local attractions']
        return any(keyword in message for keyword in suggestion_keywords)

    def _extract_city(self, message: str) -> Optional[str]:
        """Extract city name from message"""
        cities = ['delhi', 'mumbai', 'bangalore', 'london', 'paris', 'new york', 'tokyo', 'sydney', 'rome', 'berlin']
        for city in cities:
            if city in message:
                return city
        return None

    def _is_faq_query(self, message: str) -> bool:
        """Check if message is a FAQ question"""
        faq_keywords = {
            'check_in': ['check-in', 'check in', 'arrival'],
            'check_out': ['check-out', 'check out', 'departure'],
            'cancellation': ['cancel', 'cancellation', 'refund'],
            'payment': ['payment', 'pay', 'credit card', 'debit card'],
            'amenities': ['amenities', 'facilities', 'features', 'services'],
            'pet': ['pet', 'pets', 'dog', 'dogs', 'cat', 'cats'],
            'parking': ['parking', 'park', 'car'],
            'room_service': ['room service', 'food delivery', 'in-room dining'],
            'breakfast': ['breakfast', 'morning meal', 'continental breakfast']
        }
        
        for category, keywords in faq_keywords.items():
            if any(keyword in message for keyword in keywords):
                return True
        return False

    def _handle_faq_query(self, message: str) -> str:
        """Handle FAQ questions"""
        if any(word in message for word in ['check-in', 'check in', 'arrival', 'check-out', 'check out', 'departure']):
            return f"{self.faq_data['check_in_time']}\n\n{self.faq_data['check_out_time']}"
        elif any(word in message for word in ['cancel', 'cancellation', 'refund']):
                return self.faq_data["cancellation_policy"]
        elif any(word in message for word in ['payment', 'pay', 'credit card', 'debit card']):
                return self.faq_data["payment_methods"]
        elif any(word in message for word in ['amenities', 'facilities', 'features', 'services']):
                return self.faq_data["amenities"]
        elif any(word in message for word in ['pet', 'pets', 'dog', 'dogs', 'cat', 'cats']):
            return self.faq_data["pet_policy"]
        elif any(word in message for word in ['parking', 'park', 'car']):
            return self.faq_data["parking"]
        elif any(word in message for word in ['room service', 'food delivery', 'in-room dining']):
            return self.faq_data["room_service"]
        elif any(word in message for word in ['breakfast', 'morning meal', 'continental breakfast']):
            return self.faq_data["breakfast"]
        return "I'm not sure about that. Please try asking about check-in/check-out times, cancellation policy, payment methods, or amenities."

    def _is_booking_status_query(self, message: str) -> bool:
        """Check if message is about booking status"""
        status_keywords = ['status', 'details', 'information', 'show booking', 'view booking']
        return any(keyword in message for keyword in status_keywords)

    def _get_booking_status(self, booking_id: str) -> str:
        """Get booking status"""
        if booking_id in self.bookings:
            booking = self.bookings[booking_id]
            return f"""Booking details:
- Booking ID: {booking_id}
- City: {booking['city']}
- Check-in: {booking['check_in']}
- Check-out: {booking['check_out']}
- Number of guests: {booking['num_people']}
- Status: {booking['status']}
- Created at: {booking['created_at']}"""
        return "Booking ID not found"

    def _extract_booking_id(self, message: str) -> Optional[str]:
        """Extract booking ID from message"""
        booking_id_pattern = r'[A-F0-9]{8}'
        match = re.search(booking_id_pattern, message)
        if match:
            return match.group(0)
        return None

    def get_training_data(self) -> List[Dict[str, str]]:
        """Get training data for the chatbot"""
        return [
            # Check-in/Check-out related queries
            {
                'input': 'What are the check-in and check-out times?',
                'output': 'Check-in time is 3:00 PM. Early check-in may be available upon request. Check-out time is 11:00 AM. Late check-out may be available for an additional fee.'
            },
            {
                'input': 'When can I check in?',
                'output': 'Check-in time is 3:00 PM. Early check-in may be available upon request.'
            },
            {
                'input': 'What time do I need to check out?',
                'output': 'Check-out time is 11:00 AM. Late check-out may be available for an additional fee.'
            },
            {
                'input': 'Can I check in early?',
                'output': 'Early check-in may be available upon request. Please contact our front desk to arrange early check-in.'
            },
            {
                'input': 'Is late check-out possible?',
                'output': 'Late check-out may be available for an additional fee. Please contact our front desk to arrange late check-out.'
            },
            {
                'input': 'What are your arrival and departure times?',
                'output': 'Check-in time is 3:00 PM. Early check-in may be available upon request. Check-out time is 11:00 AM. Late check-out may be available for an additional fee.'
            },
            {
                'input': 'Can I arrive before 3 PM?',
                'output': 'Early check-in may be available upon request. Please contact our front desk to arrange early check-in.'
            },
            {
                'input': 'What happens if I arrive late?',
                'output': 'Our front desk is staffed 24/7. You can check in at any time after 3:00 PM.'
            },
            
            # Booking related queries
            {
                'input': 'How do I book a room?',
                'output': 'To make a booking, please provide:\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- City\n- Number of people'
            },
            {
                'input': 'I want to book a hotel in London',
                'output': 'I can help you book a hotel in London. Please provide:\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- Number of people'
            },
            {
                'input': 'Can I book a room for 2 people?',
                'output': 'Yes, I can help you book a room for 2 people. Please provide:\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- City'
            },
            {
                'input': 'I need to make a reservation',
                'output': 'I can help you make a reservation. Please provide:\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- City\n- Number of people'
            },
            {
                'input': 'How do I make a hotel reservation?',
                'output': 'To make a reservation, please provide:\n- Check-in date (YYYY-MM-DD)\n- Check-out date (YYYY-MM-DD)\n- City\n- Number of people'
            },
            {
                'input': 'Can I book a room for next weekend?',
                'output': 'Yes, I can help you book a room for next weekend. Please provide the specific dates (YYYY-MM-DD) for check-in and check-out, and let me know how many people will be staying.'
            },
            
            # Cancellation related queries
            {
                'input': 'How do I cancel my booking?',
                'output': 'To cancel your booking, please provide your booking ID. You can view your booking IDs by typing "show my bookings".'
            },
            {
                'input': 'What is your cancellation policy?',
                'output': 'Free cancellation up to 24 hours before check-in. After that, a one-night charge may apply.'
            },
            {
                'input': 'Can I cancel my reservation?',
                'output': 'Yes, you can cancel your reservation. Please provide your booking ID. Free cancellation is available up to 24 hours before check-in.'
            },
            {
                'input': 'What happens if I cancel last minute?',
                'output': 'If you cancel within 24 hours of check-in, a one-night charge may apply. Please contact our front desk for assistance.'
            },
            {
                'input': 'Is there a cancellation fee?',
                'output': 'There is no cancellation fee if you cancel at least 24 hours before check-in. After that, a one-night charge may apply.'
            },
            
            # Amenities and services
            {
                'input': 'What amenities do you offer?',
                'output': 'Our hotel offers free WiFi, swimming pool, gym, breakfast, restaurant, spa, and business center.'
            },
            {
                'input': 'Do you have a swimming pool?',
                'output': 'Yes, we have a swimming pool available for all guests.'
            },
            {
                'input': 'Is breakfast included?',
                'output': 'Yes, breakfast is included and is served daily from 6:30 AM to 10:00 AM in our main restaurant.'
            },
            {
                'input': 'What facilities do you have?',
                'output': 'Our hotel offers free WiFi, swimming pool, gym, breakfast, restaurant, spa, and business center.'
            },
            {
                'input': 'Do you have a gym?',
                'output': 'Yes, we have a fully equipped gym available for all guests.'
            },
            {
                'input': 'Is there a spa?',
                'output': 'Yes, we have a spa offering various treatments and services.'
            },
            {
                'input': 'What dining options are available?',
                'output': 'We have a restaurant serving breakfast, lunch, and dinner. Room service is also available 24/7.'
            },
            
            # Pet policy
            {
                'input': 'Are pets allowed?',
                'output': 'Pets are welcome with a $50 cleaning fee per stay. Please inform us in advance.'
            },
            {
                'input': 'What is your pet policy?',
                'output': 'Pets are welcome with a $50 cleaning fee per stay. Please inform us in advance.'
            },
            {
                'input': 'Can I bring my dog?',
                'output': 'Yes, dogs are welcome with a $50 cleaning fee per stay. Please inform us in advance.'
            },
            {
                'input': 'Do you allow cats?',
                'output': 'Yes, cats are welcome with a $50 cleaning fee per stay. Please inform us in advance.'
            },
            
            # Parking
            {
                'input': 'Do you have parking?',
                'output': 'Yes, on-site parking is available for $20 per day.'
            },
            {
                'input': 'How much is parking?',
                'output': 'On-site parking is available for $20 per day.'
            },
            {
                'input': 'Is there free parking?',
                'output': 'Parking is available for $20 per day. There is no free parking option.'
            },
            {
                'input': 'Where can I park my car?',
                'output': 'We have on-site parking available for $20 per day. The parking area is located in the hotel basement.'
            },
            
            # Room service
            {
                'input': 'Is room service available?',
                'output': 'Yes, 24-hour room service is available.'
            },
            {
                'input': 'Can I order food to my room?',
                'output': 'Yes, 24-hour room service is available for food delivery to your room.'
            },
            {
                'input': 'What are the room service hours?',
                'output': 'Room service is available 24 hours a day, 7 days a week.'
            },
            {
                'input': 'Can I get food delivered to my room?',
                'output': 'Yes, 24-hour room service is available for food delivery to your room.'
            },
            
            # Special requests
            {
                'input': 'Do you have accessible rooms?',
                'output': 'Yes, we have accessible rooms available. Please let us know your specific requirements when booking.'
            },
            {
                'input': 'Can I request a specific room?',
                'output': 'Yes, you can request a specific room type or location. Please mention your preferences when booking.'
            },
            {
                'input': 'Do you have connecting rooms?',
                'output': 'Yes, we have connecting rooms available. Please request this when making your booking.'
            },
            {
                'input': 'Can I request a room with a view?',
                'output': 'Yes, you can request a room with a view. Please mention this preference when booking.'
            },
            
            # Payment and rates
            {
                'input': 'What payment methods do you accept?',
                'output': 'We accept credit cards, debit cards, and PayPal for payment.'
            },
            {
                'input': 'Do you accept credit cards?',
                'output': 'Yes, we accept all major credit cards.'
            },
            {
                'input': 'Can I pay with cash?',
                'output': 'We prefer credit card payments, but cash is accepted at check-in.'
            },
            {
                'input': 'Do you have special rates for long stays?',
                'output': 'Yes, we offer special rates for extended stays. Please contact our reservations team for details.'
            },
            
            # Location and transportation
            {
                'input': 'How far are you from the airport?',
                'output': 'We are located 20 minutes from the airport. We offer airport shuttle service for an additional fee.'
            },
            {
                'input': 'Do you have airport shuttle service?',
                'output': 'Yes, we offer airport shuttle service for an additional fee. Please arrange this in advance.'
            },
            {
                'input': 'Is there public transportation nearby?',
                'output': 'Yes, there is a bus stop and subway station within a 5-minute walk from the hotel.'
            },
            {
                'input': 'How do I get to the hotel?',
                'output': 'We are located in the city center. You can reach us by car, public transportation, or our airport shuttle service.'
            }
        ]

    def train_model(self, training_data: List[Dict[str, str]] = None):
        """Train the enhanced LSTM model with improved training procedures"""
        if training_data is None:
            training_data = self.get_training_data()
            
        print("\n[DEBUG] Starting model training...")
        print(f"[DEBUG] Number of training examples: {len(training_data)}")
        
        # Prepare training data
        input_texts = []
        target_texts = []
        
        # Enhanced data augmentation
        augmented_data = self._augment_training_data(training_data)
        print(f"[DEBUG] Number of training examples after augmentation: {len(augmented_data)}")
        
        for conversation in augmented_data:
            input_texts.append(conversation['input'])
            target_texts.append(conversation['output'])
        
        # Create vocabulary with subword tokenization
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        
        # Build vocabulary from training data
        for text in input_texts + target_texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.word2idx) - 1] = word
        
        print(f"[DEBUG] Vocabulary size: {len(self.word2idx)}")
        
        # Convert texts to sequences with better padding
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        
        for input_text, target_text in zip(input_texts, target_texts):
            encoder_input = self.preprocess_text(input_text)
            decoder_input = [self.word2idx['<START>']] + self.preprocess_text(target_text)
            decoder_target = self.preprocess_text(target_text) + [self.word2idx['<END>']]
            
            encoder_input_data.append(encoder_input)
            decoder_input_data.append(decoder_input)
            decoder_target_data.append(decoder_target)
        
        # Pad sequences with better handling
        encoder_input_data = preprocessing.sequence.pad_sequences(
            encoder_input_data,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        decoder_input_data = preprocessing.sequence.pad_sequences(
            decoder_input_data,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        decoder_target_data = preprocessing.sequence.pad_sequences(
            decoder_target_data,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        # Build and compile the model
        self.build_model()
        
        # Enhanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                min_delta=0.001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.CSVLogger('training_log.csv'),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        print("[DEBUG] Starting model training with enhanced callbacks...")
        
        # Train the model with validation split and class weights
        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=32,
            epochs=200,  # Increased epochs
            validation_split=0.2,
            callbacks=callbacks,
            class_weight='auto'  # Handle class imbalance
        )
        
        print("[DEBUG] Training completed!")
        print(f"[DEBUG] Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"[DEBUG] Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return history

    def _augment_training_data(self, training_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enhanced data augmentation with more variations and paraphrases"""
        augmented_data = training_data.copy()
        
        # Add variations of common phrases with more context
        variations = {
            'book': ['reserve', 'get a room', 'make a reservation', 'book a stay', 'arrange accommodation'],
            'cancel': ['cancel my booking', 'cancel reservation', 'cancel my room', 'cancel my stay', 'cancel accommodation'],
            'modify': ['change', 'update', 'alter', 'adjust', 'revise'],
            'check-in': ['arrival', 'check in', 'arrive', 'come in', 'enter'],
            'check-out': ['departure', 'check out', 'leave', 'exit', 'go out'],
            'price': ['cost', 'rate', 'fee', 'charge', 'amount'],
            'room': ['accommodation', 'suite', 'lodging', 'quarters', 'chamber'],
            'hotel': ['property', 'establishment', 'lodging', 'accommodation', 'inn']
        }
        
        # Add more sophisticated paraphrases
        paraphrases = [
            ('what is the check-in time', ['when can i check in', 'what time can i arrive', 'when is check-in available']),
            ('what is the check-out time', ['when do i need to check out', 'what time is check-out', 'when must i leave']),
            ('can i cancel my booking', ['how do i cancel my reservation', 'what is the cancellation process', 'how to cancel my stay']),
            ('what amenities do you offer', ['what facilities are available', 'what services do you provide', 'what features do you have']),
            ('do you have a pool', ['is there a swimming pool', 'do you offer pool facilities', 'is pool access available']),
            ('what is your pet policy', ['are pets allowed', 'can i bring my pet', 'do you accept pets']),
            ('what payment methods do you accept', ['how can i pay', 'what forms of payment do you take', 'what payment options are available'])
        ]
        
        # Add variations with different sentence structures
        for conversation in training_data:
            input_text = conversation['input'].lower()
            output_text = conversation['output']
            
            # Add variations
            for key, values in variations.items():
                if key in input_text:
                    for variation in values:
                        new_input = input_text.replace(key, variation)
                        augmented_data.append({
                            'input': new_input,
                            'output': output_text
                        })
                        
                        # Add question variations
                        if '?' not in new_input:
                            augmented_data.append({
                                'input': f"can you tell me {new_input}",
                                'output': output_text
                            })
                            augmented_data.append({
                                'input': f"i would like to know {new_input}",
                                'output': output_text
                            })
        
        # Add paraphrases with variations
        for original, paraphrase_list in paraphrases:
            for conversation in training_data:
                if original in conversation['input'].lower():
                    for paraphrase in paraphrase_list:
                        augmented_data.append({
                            'input': paraphrase,
                            'output': conversation['output']
                        })
        
        # Add variations with different word orders
        for conversation in augmented_data:
            words = conversation['input'].split()
            if len(words) > 3:
                # Create variations with different word orders
                for i in range(len(words) - 1):
                    new_words = words.copy()
                    new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
                    augmented_data.append({
                        'input': ' '.join(new_words),
                        'output': conversation['output']
                    })
        
        return augmented_data

    def load_model(self, model_path: str):
        """Load trained model and vocabulary"""
        try:
            # Load vocabulary
            with open('vocabulary.pkl', 'rb') as f:
                vocab_data = pickle.load(f)
                if isinstance(vocab_data, tuple):
                    self.word2idx, self.idx2word = vocab_data
                else:
                    self.word2idx = vocab_data
                    self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            
            # Build model architecture
            self.build_model()
            
            # Load trained weights
            self.model.load_weights(model_path)
            
            print("Model and vocabulary loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise 