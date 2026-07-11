import cv2
import numpy as np
import pickle
import pyttsx3
import threading
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

class TTSWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.is_speaking = False

    def _speak(self, text):
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False

    def say(self, text):
        if not self.is_speaking:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

class SignDetector:
    def __init__(self):
        # Load model & labels
        self.model = load_model("models/sign_lstm_model.h5")
        with open("models/labels.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.sequence = deque(maxlen=20)
        self.stability_buffer = deque(maxlen=10)
        self.last_spoken_word = ""
        self.last_speech_time = 0
        self.CONFIDENCE_THRESHOLD = 0.85
        self.SPEECH_COOLDOWN = 1.0
        self.tts = TTSWorker()
        self.current_gesture = "..."

        self.cap = cv2.VideoCapture(0)

    def extract_normalized_landmarks(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        ref_point = hand_landmarks.landmark[9]
        scale = np.sqrt((wrist.x - ref_point.x)**2 + (wrist.y - ref_point.y)**2 + (wrist.z - ref_point.z)**2)
        if scale == 0: scale = 1
        normalized = []
        for lm in hand_landmarks.landmark:
            normalized.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
        return normalized

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None, "..."

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks_63 = self.extract_normalized_landmarks(hand_landmarks)
                self.sequence.append(landmarks_63)

                if len(self.sequence) == 20:
                    input_data = np.expand_dims(list(self.sequence), axis=0)
                    predictions = self.model.predict(input_data, verbose=0)[0]
                    class_id = np.argmax(predictions)
                    conf = predictions[class_id]
                    self.stability_buffer.append(class_id)
                    if len(self.stability_buffer) == 10:
                        stable_id = max(set(self.stability_buffer), key=list(self.stability_buffer).count)
                        if self.stability_buffer.count(stable_id) >= 7 and conf > self.CONFIDENCE_THRESHOLD:
                            word = self.label_encoder.inverse_transform([stable_id])[0]
                            now = time.time()
                            if (word != self.last_spoken_word) or (now - self.last_speech_time > self.SPEECH_COOLDOWN):
                                self.tts.say(word)
                                self.last_spoken_word = word
                                self.last_speech_time = now
                            self.current_gesture = word
                            self.sequence.clear()
        return frame, self.current_gesture
