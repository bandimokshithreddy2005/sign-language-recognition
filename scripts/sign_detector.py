import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from collections import deque

class SignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.current_gesture = "..."
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        self.stability_buffer = deque(maxlen=10)
        self.last_spoken = ""
        self.SPEECH_COOLDOWN = 1.5

    def stop(self):
        self.running = False
        self.cap.release()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            gesture_detected = "..."

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    # Simple example: detect open/closed hand using landmarks
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]

                    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) -
                                              np.array([index_tip.x, index_tip.y]))

                    if distance > 0.1:
                        gesture_detected = "Hello"
                    else:
                        gesture_detected = "Stop"

                    # Stability buffer to reduce flickering
                    self.stability_buffer.append(gesture_detected)
                    if len(self.stability_buffer) == 10:
                        stable = max(set(self.stability_buffer), key=self.stability_buffer.count)
                        gesture_detected = stable

            self.current_gesture = gesture_detected

            # Speak gesture if changed
            if gesture_detected != self.last_spoken:
                now = time.time()
                self.tts.say(gesture_detected)
                self.tts.runAndWait()
                self.last_spoken = gesture_detected

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            self.frame = buffer.tobytes()

    def get_frame(self):
        return getattr(self, "frame", None), self.current_gesture