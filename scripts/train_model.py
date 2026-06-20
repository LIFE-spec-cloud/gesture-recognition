import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.callbacks import ReduceLROnPlateau, EarlyStopping

LANDMARKS_DIR = f"dataset/landmarks"
MODELS_DIR = "models"
GESTURE_LIST = ["Hello", "Yes", "No", "Stop", "Wait", "Come_Here", "Go_Away", "Thank_You", "Help", "Clap"]
SAMPLES_PER_GESTURES = 200

def augment_sequence(sequence, noise_level=0.005, scale_range=(0.95, 1.05), shift_range=(-0.02, 0.02)):
    augmented = sequence.copy()
    
    scale_factor = np.random.uniform(scale_range[0]. scale_range[1])
    augmented = augmented * scale_factor
    
    reshaped = augmented.reshape(30, 42, 3)
    X_shift = np.random.uniform(shift_range[0], shift_range[1])
    y_shift = np.random.uniform(shift_range[0], shift_range[1])
    
    reshaped[:, :, 0] += X_shift
    reshaped[:, :, 1] += y_shift
    
    noise = np.random.normal(0, noise, reshaped.shape)
    reshaped += noise
    
    return reshaped.reshape(30, 126)

X_raw = []
y_raw = []

for idx, gesture in enumerate(GESTURE_LIST):
    folder = os.path.join(LANDMARKS_DIR, gesture)

    for f in os.listdir(folder):
        if f.endswith(".npy"):
            d = np.load(os.path.join(folder, f))
            X_raw.append(d)
            y_raw.append(idx)
        
X_raw = np.array(X_raw, dtype = np.float32)
y_raw = np.array(y_raw, dtype = np.int64)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42)

X_train_expanded,  y_train_expanded = [], []

samples_per_sign = len(X_raw) // len(GESTURE_LIST)
mutation_needed = int(SAMPLES_PER_GESTURES / samples_per_sign)

for seq, label in zip(X_train, y_train):
    X_train_expanded.append(seq)
    y_train_expanded.append(label)
    
    for i in range(mutation_needed - 1):
        mutated_seq = augment_sequence(X_train)
        X_train_expanded.append(mutated_seq)
        y_train_expanded.append(label)
        
X_train = np.array(X_train_expanded, dtype = np.float32)
y_train = np.array(y_train_expanded, dtype = np.int64)
