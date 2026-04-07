import cv2
import os
import pandas as pd
import numpy as np
import time
from utils.hand_tracker import HandTracker

def collect_data(label, num_samples=100, save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    csv_path = os.path.join(save_dir, "landmarks.csv")
    
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_hands=1)
    
    count = 0
    print(f"Collecting data for label: {label}")
    print("Keep your hand in frame. Captures will happen automatically.")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    data = []
    
    while count < num_samples:
        success, img = cap.read()
        if not success: break
            
        img = cv2.flip(img, 1)
        img = tracker.find_hands(img)
        landmarks = tracker.get_flattened_landmarks(img)
        
        if landmarks:
            landmarks.append(label)
            data.append(landmarks)
            count += 1
            cv2.putText(img, f"Captured: {count}/{num_samples}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No hand detected!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Use streamlit if possible, but for standalone script we try imshow
        try:
            cv2.imshow("Data Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        except cv2.error:
            # If imshow fails (e.g. on server/streamlit environment), we just print progress
            if count % 10 == 0:
                print(f"Progress: {count}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        df = pd.DataFrame(data)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"Saved {len(data)} samples for '{label}' to {csv_path}")

if __name__ == "__main__":
    label = input("Enter label: ")
    collect_data(label)
