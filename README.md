# Hotel Booking Chatbot using LSTM

A neural network-based chatbot for hotel bookings, built using LSTM (Long Short-Term Memory) architecture. The chatbot can handle hotel bookings, answer FAQs, provide place suggestions, and manage bookings.

## Features

1. Hotel Booking
   - Book hotels with date, city, and number of people
   - Generate unique booking IDs
   - View booking details
   - Modify bookings
   - Cancel bookings

2. FAQ Handling
   - Hotel policies
   - Check-in/check-out times
   - Payment methods
   - Amenities and facilities
   - Children and pet policies

3. Place Suggestions
   - City-specific attractions
   - Historical sites
   - Cultural places
   - Shopping areas
   - Entertainment options

## Project Structure

```
hotel-booking-chatbot/
├── src/
│   ├── core/
│   │   ├── chat_interface.py      # Main interface for users
│   │   ├── hotel_chatbot.py       # LSTM model implementation
│   │   └── __init__.py
│   ├── training/
│   │   ├── train_chatbot.py       # Training script
│   │   ├── augment_training_data.py # Data augmentation
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── data/
│   ├── processed/
│   │   └── training_data.json     # Training data
│   └── models/                    # Saved models and vocabulary
├── logs/                          # Training logs
├── visualizations/                # Training visualizations
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hotel-booking-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python -m src.training.train_chatbot
```
This will:
- Load training data from training_data.json
- Train the LSTM model
- Save the model and vocabulary
- Generate training history visualization

2. Run the chat interface:
```bash
python -m src.core.chat_interface
```

## Example Interactions

1. Booking a Hotel:
```
You: I want to book a hotel in London for 2 people from 2024-03-01 to 2024-03-05
Chatbot: [Shows booking details and asks for confirmation]
```

2. Viewing Bookings:
```
You: show my bookings
Chatbot: [Shows all active bookings with details]
```

3. Modifying a Booking:
```
You: modify [BOOKING_ID]
Chatbot: What would you like to modify?
1. Check-in date
2. Check-out date
3. Number of guests
4. City
```

4. Asking about Amenities:
```
You: tell me about your restaurants
Chatbot: [Shows detailed dining options with timings]
```

5. Getting Place Suggestions:
```
You: what are the attractions in London?
Chatbot: [Shows categorized attractions in London]
```

## Model Architecture

The chatbot uses an LSTM-based Encoder-Decoder architecture:
- Encoder: Processes input text
- Decoder: Generates responses
- Attention mechanism for better context understanding
- Dropout and batch normalization for regularization

## Training

The model is trained on hotel booking conversations with:
- Custom training data
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Learning rate reduction on plateau

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
