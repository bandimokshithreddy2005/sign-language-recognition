import cv2
import mediapipe as mp
import numpy as np
import os

gesture = input("Enter gesture name: ")

path = f"dataset/{gesture}"
os.makedirs(path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    cv2.putText(frame, f"Collecting: {gesture} - {count}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collect", frame)

    key = cv2.waitKey(1)
    if key == ord('s') or key == ord(' '):
        if results.multi_hand_landmarks:
            data = []
            for lm in results.multi_hand_landmarks[0].landmark:
                data.append(lm.x)
                data.append(lm.y)
            np.save(os.path.join(path, f"{count}.npy"), np.array(data))
            count += 1
            print(f"Saved {count}")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. Saved {count} samples for {gesture}")
