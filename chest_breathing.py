import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# -------- SIGNAL STORAGE --------
chest_signal = deque(maxlen=100)
timestamps = deque(maxlen=100)

breath_count = 0
prev_state = "DOWN"
breathing_rate = 0
last_calc_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)
    h, w, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # -------- SHOULDER POINTS --------
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        y_left = int(left_shoulder.y * h)
        y_right = int(right_shoulder.y * h)

        # -------- CHEST LEVEL (average shoulder height) --------
        chest_y = (y_left + y_right) // 2

        chest_signal.append(chest_y)
        timestamps.append(time.time())

        # Draw points
        cv2.circle(frame, (int(left_shoulder.x * w), y_left),
                   5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_shoulder.x * w),
                   y_right), 5, (0, 255, 0), -1)

        # -------- BREATH DETECTION --------
        if len(chest_signal) > 10:
            smooth = np.mean(list(chest_signal)[-5:])

            if smooth < np.mean(chest_signal):
                current_state = "UP"   # inhale
            else:
                current_state = "DOWN"  # exhale

            if prev_state == "UP" and current_state == "DOWN":
                breath_count += 1

            prev_state = current_state

    # -------- CALCULATE BREATH RATE --------
    if time.time() - last_calc_time > 10:
        breathing_rate = breath_count * 6
        breath_count = 0
        last_calc_time = time.time()

    # -------- DISPLAY --------
    cv2.putText(frame, f"Breathing Rate: {breathing_rate} bpm",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Chest Breathing Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
