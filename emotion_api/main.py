from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from emotion_analyzer.model import EmotionAnalyzer

app = FastAPI(title="Emotion Analysis API")

# Initialize the emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# Load the model and tokenizer
try:
    emotion_analyzer.load_model(
        model_path='data/models/best_model.weights.h5',
        tokenizer_path='data/models/vocabulary.pkl'
    )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotion(input_data: TextInput):
    try:
        result = emotion_analyzer.predict_emotion(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Emotion Analysis API is running"} 