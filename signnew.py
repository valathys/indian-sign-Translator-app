import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

# Paths and Configs
DATASET_DIR = "npy_data"
CACHE_FILE = "cached_dataset.npz"
SEQUENCE_LENGTH = 30
SIMILARITY_THRESHOLD = 0.5

# Load or cache dataset
def load_trained_data():
    if os.path.exists(CACHE_FILE):
        print("⚡ Loading cached dataset...")
        data_npz = np.load(CACHE_FILE, allow_pickle=True)
        return data_npz["data"].item()

    print("⏳ Loading raw dataset and creating cache...")
    data = {}
    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue
        sequences = []
        for seq in os.listdir(label_path):
            seq_path = os.path.join(label_path, seq)
            if not os.path.isdir(seq_path):
                continue
            frames = []
            for i in range(1, SEQUENCE_LENGTH + 1):
                npy_path = os.path.join(seq_path, f"{i:02}.npy")
                if os.path.exists(npy_path):
                    frames.append(np.load(npy_path))
            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(np.array(frames))
        if sequences:
            data[label] = np.mean(sequences, axis=0)  # Averaging sequences for each label
    np.savez_compressed(CACHE_FILE, data=data)
    return data

# Extract keypoints from frame
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Predict label
def predict_label(input_seq, trained_data):
    input_flat = input_seq.reshape(1, -1)
    best_label = "Unknown"
    best_score = -1

    for label, sequence in trained_data.items():
        train_flat = sequence.reshape(1, -1)
        score = cosine_similarity(input_flat, train_flat)[0][0]
        if score > best_score:
            best_score = score
            best_label = label

    return best_label if best_score > SIMILARITY_THRESHOLD else "Unknown"

# Real-time detection
def run_webcam_prediction():
    trained_data = load_trained_data()
    print("✅ Dataset loaded and ready!")

    cap = cv2.VideoCapture(0)
    sequence = []

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.array(sequence)
                prediction = predict_label(input_seq, trained_data)
                print(f"🔤 Predicted Sign: {prediction}")
                cv2.putText(image, prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow("Sign Language Prediction", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_prediction()
