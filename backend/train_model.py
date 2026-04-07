import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_dnn(csv_path="data/landmarks.csv", model_dir="backend/models"):
    if not os.path.exists(csv_path):
        print(f"Landmarks dataset not found at {csv_path}")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load data
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values.astype('float32')
    y = df.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(f"Training DNN with {num_classes} classes...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Save
    model.save(os.path.join(model_dir, "sign_language_dnn.h5"))
    with open(os.path.join(model_dir, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(le, f)

    print("Model and Label Encoder saved successfully.")

if __name__ == "__main__":
    train_dnn()
