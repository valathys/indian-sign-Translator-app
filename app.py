import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import load_model

# Load ASL Alphabet Model
asl_model = load_model('ISL_CSLRT_Corpus\Frames_Word_Level')
asl_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Load ISL Model & Gloss CSV
isl_model = load_model('sign_language_model.h5')
gloss_df = pd.read_csv("ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv")
isl_labels = gloss_df["Label"].tolist()

# Preprocess function
def preprocess_frame(frame, size=(64, 64)):
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Start Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for hand detection
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for both models
    preprocessed = preprocess_frame(roi)

    # ASL Prediction
    asl_pred = asl_model.predict(preprocessed)
    asl_result = asl_labels[np.argmax(asl_pred)]

    # ISL Prediction
    isl_pred = isl_model.predict(preprocessed)
    isl_result = isl_labels[np.argmax(isl_pred)]

    # Draw ROI on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Display predictions
    cv2.putText(frame, f"ASL: {asl_result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"ISL: {isl_result}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Sign Language Translator", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
