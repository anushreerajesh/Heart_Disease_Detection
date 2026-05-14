import streamlit as st
import cv2
import numpy as np
from ecg_model import predict_ecg

st.title("📈 ECG Analysis System")

uploaded_file = st.file_uploader(
    "Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded ECG", use_column_width=True)

    if st.button("Analyze ECG"):
        result = predict_ecg(image)

        if result == "Normal":
            st.success("✅ ECG is Normal")
        elif result == "Arrhythmia":
            st.warning("⚠️ Possible Arrhythmia detected")
        elif result == "Myocardial Infarction":
            st.error("🚨 Possible Heart Attack detected")
        else:
            st.warning("⚠️ Abnormal ECG detected")

        st.write(f"Prediction: **{result}**")
