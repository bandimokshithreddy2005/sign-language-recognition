import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading

model = joblib.load("model/gesture_model.pkl")

engine = pyttsx3.init('sapi5')
engine.setProperty('rate',150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

last_word = ""

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            data = []

            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            data = np.array(data).reshape(1,-1)

            prediction = model.predict(data)[0]

            if prediction != last_word:
                threading.Thread(target=speak, args=(prediction,)).start()
                last_word = prediction

            cv2.putText(frame, prediction, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Gesture", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
