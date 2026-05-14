import numpy as np
import cv2

# Dummy model (replace with trained CNN later)


def predict_ecg(image):

    img = cv2.resize(image, (224, 224))
    img = img / 255.0

    # Fake probabilities (for demo)
    classes = ["Normal", "Arrhythmia", "Myocardial Infarction", "Abnormal"]

    prediction = np.random.choice(classes)

    return prediction
