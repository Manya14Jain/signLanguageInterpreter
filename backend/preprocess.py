import cv2
import os
import pandas as pd
import numpy as np
import tqdm
from utils.hand_tracker import HandTracker
from dataset_loader import get_dataset_info

def extract_landmarks(dataset_path="data/asl_digits", output_csv="data/landmarks.csv"):
    dataset_info = get_dataset_info(dataset_path)
    if not dataset_info:
        print("No dataset found.")
        return

    tracker = HandTracker(max_hands=1)
    data = []
    
    print(f"Extracting landmarks from {len(dataset_info)} classes...")
    
    for label, image_paths in dataset_info.items():
        print(f"Processing class: {label}")
        count = 0
        for img_path in tqdm.tqdm(image_paths):
            img = cv2.imread(img_path)
            if img is None: continue
            
            # The Digits dataset images might be small or have specific backgrounds.
            # We use HandTracker to extract 21 landmarks.
            landmarks = tracker.get_flattened_landmarks(img)
            
            if landmarks:
                landmarks.append(label)
                data.append(landmarks)
                count += 1
                
        print(f"Captured {count}/{len(image_paths)} for {label}.")

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False, header=False)
        print(f"Dataset saved to {output_csv}")
    else:
        print("No landmarks extracted. Check if images are clear.")

if __name__ == "__main__":
    extract_landmarks()
