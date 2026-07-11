import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

frames_path = r"C:\Users\P.KARTHIKEYA REDDY\Desktop\sign_language_project\dataset"
output_csv = "landmarks.csv"

mp_hands = mp.solutions.hands

data = []

frame_files = []
for root, dirs, files in os.walk(frames_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            frame_files.append(os.path.join(root, file))

print("Total frames:", len(frame_files))

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:

    for path in tqdm(frame_files, desc="Extracting landmarks"):
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            row = []
            for p in lm.landmark:
                row.extend([p.x, p.y, p.z])
            label = os.path.basename(os.path.dirname(path))
            row.append(label)
            data.append(row)

if data:
    cols = []
    for i in range(21):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])
    cols.append("label")
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
else:
    print("No landmarks extracted.")
