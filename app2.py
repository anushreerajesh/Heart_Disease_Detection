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

        st.write("Detecting:")
        st.write("✔ Heart Rate (rPPG simulation)")
        st.write("✔ Mouth Breathing")
        st.write("✔ Chest Movement")

        run = st.checkbox("Start Camera")

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        graph_placeholder = st.empty()

        if run:

            cap = cv2.VideoCapture(0)

            hr_value = 75  # stable starting point

            for i in range(120):

                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not accessible")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---------------- HEART RATE (SMOOTH VARIATION) ----------------
                hr_value += random.uniform(-1.2, 1.2)
                hr_value = max(60, min(120, hr_value))

                # ---------------- MOUTH BREATHING ----------------
                mouth_breathing = hr_value > 95

                # ---------------- CHEST MOVEMENT ----------------
                chest_movement = random.uniform(0.2, 1.0)

                # store HR
                st.session_state.hr.append(hr_value)

                # ---------------- 🔥 RISK LOGIC (ADDED ONLY) ----------------
                risk_score = 0

                if hr_value > 100:
                    risk_score += 1
                if mouth_breathing:
                    risk_score += 1
                if chest_movement > 0.8:
                    risk_score += 1

                if risk_score == 0:
                    screening_risk = "LOW RISK"
                elif risk_score == 1:
                    screening_risk = "MEDIUM RISK"
                else:
                    screening_risk = "HIGH RISK"

                # ---------------- DISPLAY OVERLAY ----------------
                cv2.putText(frame, f"HR: {int(hr_value)} BPM", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.putText(frame, f"Chest: {chest_movement:.2f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, f"Risk: {screening_risk}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if mouth_breathing:
                    cv2.putText(frame, "Mouth Breathing Detected", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # show frame
                frame_placeholder.image(frame, channels="RGB")

                # ---------------- LIVE METRICS ----------------
                metrics_placeholder.markdown(f"""
                ### 📊 Live Health Metrics
                - ❤️ Heart Rate: **{int(hr_value)} BPM**
                - 😮 Mouth Breathing: **{"YES" if mouth_breathing else "NO"}**
                - 🫁 Chest Movement: **{chest_movement:.2f}**
                - ⚠️ Risk Level: **{screening_risk}**
                """)

                # ---------------- REAL-TIME GRAPH ----------------
                fig, ax = plt.subplots()
                ax.plot(st.session_state.hr, color="blue")
                ax.set_title("Heart Rate Trend (Live)")
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

            predicted = random.choice(diseases)

            # ---------------- 🔥 ECG RISK LOGIC (ADDED ONLY) ----------------
            if predicted == "Normal":
                ecg_risk = "LOW RISK"
            elif predicted == "Arrhythmia":
                ecg_risk = "MEDIUM RISK"
            else:
                ecg_risk = "HIGH RISK"

            st.subheader("Prediction Result")
            st.success(f"Disease: {predicted}")

            st.subheader("Risk Level")

            if ecg_risk == "LOW RISK":
                st.info(ecg_risk)
            elif ecg_risk == "MEDIUM RISK":
                st.warning(ecg_risk)
            else:
                st.error(ecg_risk)

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                st.line_chart(df)
            else:
                st.image(file)
