import joblib
import re
import string
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- Preprocessing Function (from Notebook) ---
def clean_text(text: str) -> str:
    """
    Performs basic text cleaning used during model training.
    """
    text = text.lower() # Convert to lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove digits
    text = text.strip() # Remove leading/trailing whitespace
    return text


MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / 'intent_classifier_pipeline.pkl'
ENCODER_PATH = MODEL_DIR / 'intent_label_encoder.pkl'

class IntentClassifier:
    """Handles loading and inference for the intent classification model."""

    def __init__(self):
        self.pipeline = None
        self.label_encoder = None
        self.is_loaded = False

    def load_model(self):
        """Loads the model pipeline and label encoder from disk."""
        if not self.is_loaded:
            print("Loading ML model artifacts...")
            try:
                self.pipeline = joblib.load(MODEL_PATH)
                self.label_encoder = joblib.load(ENCODER_PATH)
                self.is_loaded = True
                print("Model artifacts loaded successfully.")
            except FileNotFoundError as e:
                # Critical failure: API will be in a degraded state
                raise FileNotFoundError(
                    f"Required model file not found: {e}. "
                    "Ensure .pkl files are placed in the 'ml/' directory."
                )

    def classify_single(self, text: str) -> Tuple[str, float]:
        """Classify a single query and return intent and confidence."""
        # 1. Preprocess the text
        processed_text = clean_text(text)
        
        # 2. Predict probability
        proba = self.pipeline.predict_proba([processed_text])[0]
        
        # 3. Get the intent and confidence
        pred_index = proba.argmax()
        predicted_intent = self.label_encoder.inverse_transform([pred_index])[0]
        confidence = proba[pred_index]
        
        return predicted_intent, float(confidence)

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple queries and return a list of results."""
        # 1. Preprocess all texts
        processed_texts = [clean_text(t) for t in texts]

        # 2. Predict probability
        probas = self.pipeline.predict_proba(processed_texts)
        
        # 3. Compile results
        predictions = []
        for text, proba in zip(texts, probas):
            pred_index = proba.argmax()
            predicted_intent = self.label_encoder.inverse_transform([pred_index])[0]
            confidence = proba[pred_index]
            
            predictions.append({
                "text": text,
                "intent": predicted_intent,
                "confidence": float(confidence)
            })
            
        return predictions

# Global instance for model caching
model_cache = IntentClassifier()

# Dependency injector helper
def get_model_loader() -> IntentClassifier:
    return model_cache