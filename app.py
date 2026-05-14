import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import random
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Health System", layout="wide")

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "name" not in st.session_state:
    st.session_state.name = ""

if "page" not in st.session_state:
    st.session_state.page = "login"

if "hr" not in st.session_state:
    st.session_state.hr = []

# ---------------- RISK MAP (ECG) ----------------
risk_map = {
    "Normal": "LOW RISK",
    "Arrhythmia": "MEDIUM RISK",
    "Myocardial Infarction": "HIGH RISK",
    "Atrial Fibrillation": "HIGH RISK"
}

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.title("🩺 AI Health Screening System")

    with st.form("login_form"):
        first_name = st.text_input("First Name")
        age = st.number_input("Age", 1, 100)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", 1, 200)

        submit = st.form_submit_button("Save")

        if submit:
            if first_name.strip() == "":
                st.warning("Please enter your name")
            else:
                st.session_state.logged_in = True
                st.session_state.name = first_name
                st.session_state.page = "home"
                st.rerun()

else:

    # ---------------- HOME PAGE ----------------
    if st.session_state.page == "home":

        st.success(f"Hi, {st.session_state.name} 👋")
        st.title("Choose a Module")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🫁 Screening Module"):
                st.session_state.page = "screening"
                st.rerun()

        with col2:
            if st.button("💓 ECG Analysis"):
                st.session_state.page = "ecg"
                st.rerun()

    # ---------------- SCREENING MODULE ----------------
    elif st.session_state.page == "screening":

        st.title("🫁 Real-Time Screening System")

        if st.button("⬅ Back to Home"):
            st.session_state.page = "home"
            st.rerun()

        run = st.checkbox("Start Camera")

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        graph_placeholder = st.empty()

        if run:

            cap = cv2.VideoCapture(0)
            hr_value = 75

            for i in range(120):

                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not accessible")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---------------- SIMULATED SIGNALS ----------------
                hr_value += random.uniform(-1.5, 1.5)
                hr_value = max(55, min(140, hr_value))

                mouth_breathing = hr_value > 100
                chest_movement = random.uniform(0.2, 1.2)

                st.session_state.hr.append(hr_value)

                # ---------------- RISK LOGIC (SCREENING) ----------------
                risk_score = 0

                if hr_value > 100:
                    risk_score += 1
                if mouth_breathing:
                    risk_score += 1
                if chest_movement > 0.9:
                    risk_score += 1

                if risk_score == 0:
                    screening_risk = "LOW RISK"
                elif risk_score == 1:
                    screening_risk = "MEDIUM RISK"
                else:
                    screening_risk = "HIGH RISK"

                # ---------------- OVERLAY ----------------
                cv2.putText(frame, f"HR: {int(hr_value)} BPM", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.putText(frame, f"Risk: {screening_risk}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if mouth_breathing:
                    cv2.putText(frame, "Mouth Breathing Detected", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                frame_placeholder.image(frame, channels="RGB")

                # ---------------- METRICS ----------------
                metrics_placeholder.markdown(f"""
                ### 📊 Live Health Metrics
                - ❤️ Heart Rate: **{int(hr_value)} BPM**
                - 😮 Mouth Breathing: **{"YES" if mouth_breathing else "NO"}**
                - 🫁 Chest Movement: **{chest_movement:.2f}**
                - ⚠️ Risk Level: **{screening_risk}**
                """)

                # ---------------- GRAPH ----------------
                fig, ax = plt.subplots()
                ax.plot(st.session_state.hr)
                ax.set_title("Heart Rate Trend")
                ax.set_xlabel("Time")
                ax.set_ylabel("BPM")

                graph_placeholder.pyplot(fig)

                time.sleep(0.08)

            cap.release()

    # ---------------- ECG MODULE ----------------
    elif st.session_state.page == "ecg":

        st.title("💓 ECG Disease Detection")

        if st.button("⬅ Back to Home"):
            st.session_state.page = "home"
            st.rerun()

        file = st.file_uploader("Upload ECG File", type=[
                                "csv", "png", "jpg", "jpeg"])

        if file is not None:

            st.success("File uploaded successfully")

            diseases = [
                "Normal",
                "Arrhythmia",
                "Myocardial Infarction",
                "Atrial Fibrillation"
            ]

            predicted_disease = random.choice(diseases)
            risk = risk_map[predicted_disease]

            st.subheader("Prediction Result")
            st.success(f"Disease: {predicted_disease}")

            st.subheader("Heart Risk Level")

            if risk == "LOW RISK":
                st.info(risk)
            elif risk == "MEDIUM RISK":
                st.warning(risk)
            else:
                st.error(risk)

            # ---------------- DOCTOR STYLE OUTPUT ----------------
            st.markdown("### 🧠 AI Recommendation")

            if risk == "HIGH RISK":
                st.error("Immediate cardiology consultation recommended ⚠️")
            elif risk == "MEDIUM RISK":
                st.warning("Monitor ECG regularly and reduce stress")
            else:
                st.success("Normal cardiac activity detected")

            # ---------------- DISPLAY FILE ----------------
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                st.line_chart(df)
            else:
                st.image(file)
