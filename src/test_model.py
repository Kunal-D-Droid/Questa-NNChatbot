import tensorflow as tf
import pickle
from src.core.hotel_chatbot import HotelBookingChatbot

def test_model():
    print("Loading model and vocabulary...")
    try:
        # Load vocabulary
        with open('data/models/vocabulary.pkl', 'rb') as f:
            word2idx, idx2word = pickle.load(f)
        
        # Initialize chatbot
        chatbot = HotelBookingChatbot()
        chatbot.word2idx = word2idx
        chatbot.idx2word = idx2word
        
        # Build model and load weights
        print("Building model and loading weights...")
        chatbot.build_model()
        chatbot.model.load_weights('data/models/best_model.weights.h5')
        print("Model loaded successfully!")
        
        # Test some example inputs
        test_inputs = [
            "hello",
            "what are your room rates?",
            "do you have a swimming pool?",
            "can i book a room for next week?",
            "what time is check in?"
        ]
        
        print("\nTesting model responses:")
        print("-" * 50)
        for input_text in test_inputs:
            print(f"\nInput: {input_text}")
            response = chatbot.generate_response(input_text)
            print(f"Response: {response}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model() 