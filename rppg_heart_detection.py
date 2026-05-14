import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# rPPG
green_signal = deque(maxlen=300)
timestamps = deque(maxlen=300)

# BPM
bpm_history = deque(maxlen=10)
stable_bpm = 0
last_bpm_time = time.time()

# Breathing
mouth_distances = deque(maxlen=100)
breathing_rate = 0
last_breath_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # -------- FOREHEAD --------
            forehead_indices = [10, 338, 297, 332, 284]
            pts = []

            for idx in forehead_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                pts.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            x, y, w_box, h_box = cv2.boundingRect(np.array(pts))
            forehead_roi = frame[y:y+h_box, x:x+w_box]

            if forehead_roi.size > 0:
                green = np.mean(forehead_roi[:, :, 1])
                green_signal.append(green)
                timestamps.append(time.time())

            # -------- BPM --------
            if len(green_signal) > 60 and (time.time() - last_bpm_time > 2):

                signal = np.array(green_signal)
                signal = signal - np.mean(signal)
                signal = np.convolve(signal, np.ones(5)/5, mode='valid')

                fft = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), d=1/30)

                valid = np.where((freqs >= 0.8) & (freqs <= 2.5))

                if len(valid[0]) > 0:
                    peak = freqs[valid][np.argmax(np.abs(fft[valid]))]
                    raw_bpm = peak * 60

                    if 55 < raw_bpm < 120:
                        if stable_bpm == 0 or abs(raw_bpm - stable_bpm) < 15:
                            bpm_history.append(raw_bpm)
                            stable_bpm = int(np.mean(bpm_history))

                last_bpm_time = time.time()

            # -------- BREATHING (MOUTH DISTANCE) --------
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            x1, y1 = int(upper_lip.x * w), int(upper_lip.y * h)
            x2, y2 = int(lower_lip.x * w), int(lower_lip.y * h)

            cv2.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 3, (255, 0, 0), -1)

            distance = abs(y2 - y1)
            mouth_distances.append(distance)

            # -------- BREATH RATE --------
            if len(mouth_distances) > 30 and (time.time() - last_breath_time > 3):

                signal = np.array(mouth_distances)
                signal = signal - np.mean(signal)

                peaks = np.where(signal > np.std(signal))[0]

                if len(peaks) > 1:
                    breaths = len(peaks) / 3  # approx in 3 sec window
                    breathing_rate = int(breaths * 20)  # scale to per minute

                last_breath_time = time.time()

    # -------- DISPLAY --------
    cv2.putText(frame, f"Heart Rate: {stable_bpm} BPM", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Breathing: {breathing_rate} bpm", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Cardiac Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
