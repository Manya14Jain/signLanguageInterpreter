"""
Inference wrapper around GestureEngine.
Returns (letter, confidence, finger_states) for each frame.
"""

import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

from gesture_engine import GestureEngine


class SignLanguagePredictor:
    def __init__(self, buffer_size=15):
        self.engine = GestureEngine(buffer_size=buffer_size)

    def predict(self, landmarks):
        """
        landmarks: list of 21 [x, y, z] floats from MediaPipe, or None.
        Returns (letter_str_or_None, confidence_float, finger_states_dict).
        """
        letter, confidence, finger_states = self.engine.classify(landmarks)
        return letter, confidence, finger_states

    def reset(self):
        self.engine.reset()
