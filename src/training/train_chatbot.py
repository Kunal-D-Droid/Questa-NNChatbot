import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing
import json
import pickle
import matplotlib.pyplot as plt
from ..core.hotel_chatbot import HotelBookingChatbot

# Configure GPU
print("Configuring GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TF from taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s): {gpus}")
        print("GPU acceleration enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

def augment_data(conversations):
    """Create variations of training data"""
    augmented = []
    
    # Add original conversations
    augmented.extend(conversations)
    
    # Create variations
    for conv in conversations:
        try:
            input_text = conv.get("input", "").lower()
            output_text = conv.get("output", "")
            
            if not input_text or not output_text:
                continue
            
            # Greeting variations
            if "hello" in input_text or "hi" in input_text:
                variations = [
                    "hey there",
                    "good morning",
                    "good afternoon",
                    "good evening",
                    "greetings"
                ]
                for var in variations:
                    augmented.append({"input": var, "output": output_text})
            
            # Question variations
            if "?" in input_text:
                # Remove question mark and add different question formats
                base = input_text.replace("?", "").strip()
                variations = [
                    f"can you tell me {base}",
                    f"i would like to know {base}",
                    f"could you inform me about {base}",
                    f"please tell me {base}"
                ]
                for var in variations:
                    augmented.append({"input": var, "output": output_text})
            
            # Polite request variations
            if "can" in input_text or "could" in input_text:
                base = input_text.replace("can", "").replace("could", "").strip()
                variations = [
                    f"i would like to {base}",
                    f"i want to {base}",
                    f"please {base}",
                    f"is it possible to {base}"
                ]
                for var in variations:
                    augmented.append({"input": var, "output": output_text})
        except Exception as e:
            print(f"Error processing conversation: {e}")
            continue
    
    return augmented

def load_training_data():
    """Load and prepare training data"""
    print("Loading and preparing training data...")
    
    try:
        # Load conversations from JSON file
        with open('data/processed/training_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            conversations = data.get('conversations', [])
        
        if not conversations:
            raise ValueError("No conversations found in training data")
        
        # Augment the data
        augmented_conversations = augment_data(conversations)
        
        print(f"Original conversations: {len(conversations)}")
        print(f"Augmented conversations: {len(augmented_conversations)}")
        
        return augmented_conversations
    except FileNotFoundError:
        print("Error: training_data_augmented.json file not found")
        raise
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in training_data_augmented.json")
        raise
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise

def prepare_training_data(conversations, chatbot):
    """Prepare training data for the model"""
    # Create vocabulary
    words = set()
    valid_conversations = []
    
    # First pass: collect words and validate conversations
    for conv in conversations:
        try:
            input_text = conv.get("input", "").lower()
            output_text = conv.get("output", "")
            
            if not input_text or not output_text:
                continue
                
            # Add words to vocabulary
            words.update(input_text.split())
            words.update(output_text.split())
            valid_conversations.append(conv)
            
        except Exception as e:
            print(f"Error processing conversation for vocabulary: {e}")
            continue
    
    if not valid_conversations:
        raise ValueError("No valid conversations found after filtering")
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    words.update(special_tokens)
    
    # Create word to index mapping
    chatbot.word2idx = {word: idx for idx, word in enumerate(words)}
    chatbot.idx2word = {idx: word for word, idx in chatbot.word2idx.items()}
    
    print(f"Vocabulary size: {len(words)}")
    print(f"Valid conversations for training: {len(valid_conversations)}")
    
    # Prepare input and target sequences
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []
    
    # Second pass: create sequences
    for conv in valid_conversations:
        try:
            input_text = conv.get("input", "").lower()
            output_text = conv.get("output", "")
            
            # Convert input text to sequence
            encoder_input = [chatbot.word2idx.get(word, chatbot.word2idx['<UNK>']) 
                           for word in input_text.split()]
            encoder_input_data.append(encoder_input)
            
            # Convert output text to sequence
            decoder_input = [chatbot.word2idx['<START>']] + [
                chatbot.word2idx.get(word, chatbot.word2idx['<UNK>']) 
                for word in output_text.split()
            ]
            decoder_input_data.append(decoder_input)
            
            # Target sequence is decoder input shifted by one
            decoder_target = decoder_input[1:] + [chatbot.word2idx['<END>']]
            decoder_target_data.append(decoder_target)
            
        except Exception as e:
            print(f"Error creating sequences for conversation: {e}")
            continue
    
    if not encoder_input_data:
        raise ValueError("No valid sequences could be created from the conversations")
    
    # Pad sequences
    encoder_input_data = preprocessing.sequence.pad_sequences(
        encoder_input_data, maxlen=chatbot.max_sequence_length, padding='post'
    )
    decoder_input_data = preprocessing.sequence.pad_sequences(
        decoder_input_data, maxlen=chatbot.max_sequence_length, padding='post'
    )
    decoder_target_data = preprocessing.sequence.pad_sequences(
        decoder_target_data, maxlen=chatbot.max_sequence_length, padding='post'
    )
    
    print(f"Final number of training examples: {len(encoder_input_data)}")
    
    return encoder_input_data, decoder_input_data, decoder_target_data

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_history.png')
    plt.close()

def train_model():
    """Train the chatbot model"""
    # Load training data
    conversations = load_training_data()
    
    # Initialize chatbot
    chatbot = HotelBookingChatbot()
    
    # Prepare training data
    encoder_input_data, decoder_input_data, decoder_target_data = prepare_training_data(conversations, chatbot)
    
    # Build model
    print("\nBuilding LSTM Encoder-Decoder model...")
    chatbot.build_model()
    
    # Load previous best weights if they exist
    try:
        print("\nLoading previous best weights...")
        # First try loading the full model
        try:
            chatbot.model = tf.keras.models.load_model('data/models/best_model.keras')
            print("Successfully loaded full model")
        except:
            # If that fails, try loading just the weights
            chatbot.model.load_weights('data/models/best_model.weights.h5')
            print("Successfully loaded model weights")
    except Exception as e:
        print(f"Could not load previous weights: {e}")
        print("Starting training from scratch")
    
    # Print model summary
    chatbot.model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'data/models/best_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('logs/training_log.csv'),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = chatbot.model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and vocabulary
    print("\nSaving model and vocabulary...")
    try:
        # Save model weights
        print("Saving model weights...")
        chatbot.model.save_weights('data/models/best_model.weights.h5')
        print("Model weights saved successfully as 'best_model.weights.h5'")
        
        # Save vocabulary
        with open('data/models/vocabulary.pkl', 'wb') as f:
            pickle.dump((chatbot.word2idx, chatbot.idx2word), f)
        print("Vocabulary saved as 'vocabulary.pkl'")
        
        print("Training completed successfully!")
        print("Training history plot saved as 'visualizations/training_history.png'")
        print("Training logs saved in 'logs/training_log.csv'")
        print("TensorBoard logs saved in './logs' directory")
    except Exception as e:
        print(f"Error in model saving process: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 