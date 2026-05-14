import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime
from report_generator import generate_report

# -------- PATIENT --------


def read_patient():
    try:
        with open("patient.txt", "r") as f:
            return f.read().split(",")
    except:
        return "Unknown", "N/A", "N/A"


# -------- INIT --------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# -------- HEART --------
green_signal = deque(maxlen=300)
bpm_history = deque(maxlen=10)
stable_bpm = 0
last_bpm_time = time.time()

# -------- CHEST BREATH --------
chest_signal = deque(maxlen=100)
breath_count = 0
prev_state = "DOWN"
breathing_rate = 0
last_breath_time = time.time()

# -------- MOUTH BREATH --------
mouth_count = 0
mouth_prev = "CLOSED"
mouth_rate = 0
last_mouth_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_mesh.process(rgb)
    pose_res = pose.process(rgb)

    h, w, _ = frame.shape

    # ================= HEART + MOUTH =================
    if face_res.multi_face_landmarks:
        for face in face_res.multi_face_landmarks:

            # ---- HEART (rPPG) ----
            pts = []
            for i in [10, 338, 297, 332, 284]:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                pts.append((x, y))

            x, y, w_box, h_box = cv2.boundingRect(np.array(pts))
            roi = frame[y:y+h_box, x:x+w_box]

            if roi.size > 0:
                green_signal.append(np.mean(roi[:, :, 1]))

            if len(green_signal) > 60 and time.time() - last_bpm_time > 2:
                signal = np.array(green_signal)
                signal -= np.mean(signal)

                fft = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), d=1/30)

                idx = np.where((freqs >= 0.8) & (freqs <= 2.5))

                if len(idx[0]) > 0:
                    peak = freqs[idx][np.argmax(np.abs(fft[idx]))]
                    bpm = peak * 60

                    if 55 < bpm < 120:
                        bpm_history.append(bpm)
                        stable_bpm = int(np.mean(bpm_history))

                last_bpm_time = time.time()

            # ---- MOUTH BREATH ----
            upper = face.landmark[13]
            lower = face.landmark[14]
            nose = face.landmark[1]
            chin = face.landmark[152]

            ratio = abs(lower.y - upper.y) / abs(chin.y - nose.y)

            current = "OPEN" if ratio > 0.02 else "CLOSED"

            if mouth_prev == "OPEN" and current == "CLOSED":
                mouth_count += 1

            mouth_prev = current

    # ================= CHEST BREATH =================
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark

        left = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        y_left = int(left.y * h)
        y_right = int(right.y * h)

        chest_y = (y_left + y_right) // 2
        chest_signal.append(chest_y)

        if len(chest_signal) > 10:
            smooth = np.mean(list(chest_signal)[-5:])

            if smooth < np.mean(chest_signal):
                current_state = "UP"
            else:
                current_state = "DOWN"

            if prev_state == "UP" and current_state == "DOWN":
                breath_count += 1

            prev_state = current_state

    # -------- CALCULATIONS --------
    if time.time() - last_breath_time > 10:
        breathing_rate = breath_count * 6
        breath_count = 0
        last_breath_time = time.time()

    if time.time() - last_mouth_time > 10:
        mouth_rate = mouth_count * 6
        mouth_count = 0
        last_mouth_time = time.time()

    # ================= RISK =================
    if stable_bpm > 120:
        risk = "HIGH"
        name, age, sex = read_patient()

        pdf = generate_report(name, age, sex, stable_bpm, breathing_rate, risk)

        with open("latest_pdf.txt", "w") as f:
            f.write(pdf)

        with open("report.txt", "a") as f:
            f.write(f"{datetime.now()} | {name} | HIGH | {stable_bpm}\n")

    elif stable_bpm > 90:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # -------- SEND TO UI --------
    with open("data.txt", "w") as f:
        f.write(f"{stable_bpm},{breathing_rate},{mouth_rate},{risk}")

    # -------- DISPLAY --------
    cv2.putText(frame, f"Heart: {stable_bpm}", (30, 50), 0, 1, (0, 0, 255), 2)
    cv2.putText(
        frame, f"Chest Breath: {breathing_rate}", (30, 100), 0, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Mouth Breath: {mouth_rate}",
                (30, 150), 0, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Risk: {risk}", (30, 200), 0, 1, (0, 255, 0), 2)

    cv2.imshow("Cardiac Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
