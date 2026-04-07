import pickle
import numpy as np
import os
from collections import Counter

class SignLanguagePredictor:
    def __init__(self, model_path="backend/models/model.pkl", le_path="backend/models/labels.pkl", buffer_size=10):
        self.model = None
        self.le = None
        self.buffer = []
        self.buffer_size = buffer_size
        
        if os.path.exists(model_path) and os.path.exists(le_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(le_path, 'rb') as f:
                self.le = pickle.load(f)
            print("Model and labels loaded successfully.")
        else:
            print(f"Model or labels not found at {model_path} or {le_path}")

    def predict(self, landmarks):
        if self.model is None or self.le is None:
            return None, 0.0
        
        if landmarks is None:
            return None, 0.0

        # Predict
        prediction_prob = self.model.predict_proba([landmarks])[0]
        prediction_idx = np.argmax(prediction_prob)
        confidence = prediction_prob[prediction_idx]
        
        label = self.le.inverse_transform([prediction_idx])[0]
        
        # Temporal smoothing
        self.buffer.append(label)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Get most common prediction in buffer
        most_common_label, count = Counter(self.buffer).most_common(1)[0]
        
        # Only return if the most common label has high enough frequency in the buffer
        if count >= self.buffer_size // 2:
            return most_common_label, float(confidence)
        
        return None, float(confidence)

    def reset_buffer(self):
        self.buffer = []
