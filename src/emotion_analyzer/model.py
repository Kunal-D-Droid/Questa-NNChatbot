import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pickle

class EmotionAnalyzer:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        
    def build_model(self):
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_text(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        return padded
    
    def predict_emotion(self, text):
        if not self.model:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict(processed_text)
        emotion_idx = np.argmax(prediction[0])
        emotion = self.emotions[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': dict(zip(self.emotions, prediction[0].tolist()))
        }
    
    def save_model(self, model_path, tokenizer_path):
        if self.model:
            self.model.save(model_path)
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
    
    def load_model(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f) 