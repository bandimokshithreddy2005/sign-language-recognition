from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading

app = Flask(__name__)

# Load trained model
model = joblib.load("model/gesture_model.pkl")

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate',150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera object
cap = None

# Gesture tracking
current_gesture = ""
last_spoken = ""

def generate_frames():
    global cap, current_gesture, last_spoken
    while cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
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
                current_gesture = prediction

                # Speak only when gesture changes
                if current_gesture != last_spoken:
                    threading.Thread(target=speak,args=(current_gesture,)).start()
                    last_spoken = current_gesture

                cv2.putText(frame,current_gesture,(50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/gesture')
def gesture():
    return jsonify({"gesture": current_gesture})

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return jsonify({"status":"started"})

@app.route('/stop_camera')
def stop_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
        cap = None
    return jsonify({"status":"stopped"})

if __name__ == "__main__":
    app.run(debug=True)