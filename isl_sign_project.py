import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
DATASET_PATH = "ISL_CSLRT_Corpus/Frames_Sentence_Level"  # Update this path
MODEL_PATH = "sign_language_model.h5"
SEQUENCE_LENGTH = 16  # Matches your 16-frame sequences
THRESHOLD = 0.7
KEYPOINTS_SHAPE = 1662  # 33*3 (pose) + 21*3 (left hand) + 21*3 (right hand)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    """Extract pose, left and right hand keypoints with consistent shape"""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([pose, lh, rh])
    
    # Ensure consistent shape
    if len(keypoints) != KEYPOINTS_SHAPE:
        print(f"⚠ Warning: Expected {KEYPOINTS_SHAPE} keypoints, got {len(keypoints)}")
        return np.zeros(KEYPOINTS_SHAPE)
    return keypoints

def load_dataset():
    """Load dataset with strict shape enforcement"""
    print(f"\nLoading dataset from: {os.path.abspath(DATASET_PATH)}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset path does not exist")
        return None, None, None
    
    actions = []
    sequences = []
    labels = []
    
    # Process each action folder
    for label_idx, action_folder in enumerate(sorted(os.listdir(DATASET_PATH))):
        action_path = os.path.join(DATASET_PATH, action_folder)
        
        if not os.path.isdir(action_path):
            continue
            
        action_name = " ".join(action_folder.split("_"))
        actions.append(action_name)
        print(f"Processing: {action_name}")
        
        # Get sequence folders (1, 2, etc.)
        sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
        
        for seq_folder in sorted(sequence_folders):
            seq_path = os.path.join(action_path, seq_folder)
            frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])
            
            sequence = []
            for frame_file in frame_files[:SEQUENCE_LENGTH]:
                frame_path = os.path.join(seq_path, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"⚠ Could not read {frame_file}")
                    continue
                    
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            
            # Only add complete sequences
            if len(sequence) == SEQUENCE_LENGTH:
                sequences.append(sequence)
                labels.append(label_idx)
    
    if not sequences:
        print("❌ Error: No valid sequences created")
        return None, None, None
    
    # Convert to numpy arrays with strict shape checking
    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(actions))
    
    print(f"\nDataset loaded successfully:")
    print(f"- Actions: {len(actions)}")
    print(f"- Sequences: {len(sequences)}")
    print(f"- Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"- Keypoints per frame: {KEYPOINTS_SHAPE}")
    
    return X, y, actions

def create_model(input_shape, num_classes):
    """Create LSTM model with correct input shape"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(X, y, actions):
    """Train and save the model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    model = create_model((SEQUENCE_LENGTH, KEYPOINTS_SHAPE), len(actions))
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return load_model(MODEL_PATH)

def real_time_detection(model, actions):
    """Run real-time sign detection with shape validation"""
    sequence = []
    predictions = []
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]
            
            # Pad sequence if needed
            if len(sequence) < SEQUENCE_LENGTH:
                display_len = len(sequence)
                padding = [np.zeros(KEYPOINTS_SHAPE)] * (SEQUENCE_LENGTH - len(sequence))
                sequence = padding + sequence
            else:
                display_len = SEQUENCE_LENGTH
            
            # Only predict when we have a full sequence
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    # Reshape input to match model expectations
                    input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                    res = model.predict(input_data, verbose=0)[0]
                    confidence = np.max(res)
                    
                    if confidence > THRESHOLD:
                        predicted_action = actions[np.argmax(res)]
                        if not predictions or predicted_action != predictions[-1]:
                            predictions.append(predicted_action)
                            print(f"Detected: {predicted_action} ({confidence:.2f} confidence)")
                        
                        # Keep last 3 unique predictions
                        predictions = list(dict.fromkeys(predictions[-3:]))
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue
            
            # Display info
            cv2.putText(frame, f"Frames: {display_len}/{SEQUENCE_LENGTH}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if predictions:
                cv2.putText(frame, " | ".join(predictions), 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        print("\nLoading pre-trained model...")
        try:
            model = load_model(MODEL_PATH)
            # Need to get the actions list
            print("Loading dataset to get action labels...")
            _, _, actions = load_dataset()
            if not actions:
                raise ValueError("Couldn't load action labels")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train new model...")
            model = None
    else:
        model = None
    
    # Train if no model exists
    if model is None:
        print("\nLoading dataset for training...")
        X, y, actions = load_dataset()
        
        if X is None:
            print("❌ Cannot proceed without valid dataset")
            return
        
        print("\nTraining model...")
        model = train_model(X, y, actions)
    
    print("\nStarting real-time detection...")
    print("Press 'q' to quit")
    real_time_detection(model, actions)

if __name__ == "__main__":
    main()