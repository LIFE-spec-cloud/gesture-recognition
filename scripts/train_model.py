import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

LANDMARKS_DIR = f"dataset/landmarks"
MODELS_DIR = "models/gesture_classifier"
GESTURE_LIST = ["Hello", "Yes", "No", "Stop", "Wait", "Come_Here", "Go_Away", "Thank_You", "Help", "Clap"]
SAMPLES_PER_GESTURES = 50

def augment_sequence(sequence, noise_level=0.005, scale_range=(0.95, 1.05), shift_range=(-0.02, 0.02)):
    augmented = sequence.copy()
    
    scale_factor = np.random.uniform(scale_range[0]. scale_range[1])
    augmented = augmented * scale_factor
    
    reshaped = augmented.reshape(30, 42, 3)
    X_shift = np.random.uniform(shift_range[0], shift_range[1])
    y_shift = np.random.uniform(shift_range[0], shift_range[1])
    
    reshaped[:, :, 0] += X_shift
    reshaped[:, :, 1] += y_shift
    
    noise = np.random.normal(0, noise_level, reshaped.shape)
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
        mutated_seq = augment_sequence(seq)
        X_train_expanded.append(mutated_seq)
        y_train_expanded.append(label)
        
X_train = np.array(X_train_expanded, dtype = np.float32)
y_train = np.array(y_train_expanded, dtype = np.int64)
indices = np.arange(X_train.shape[0]) 
np.random.shuffle(indices) 
X_train = X_train[indices] 
y_train = y_train[indices]

model = keras.Sequential([
    keras.Input(shape=(30, 126)),
    
    keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv1D(256, kernel_size=5, padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.3),
    
    keras.layers.GlobalAveragePooling1D(),
    
    keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(len(GESTURE_LIST), activation="softmax")
])

model.compile(
    optimizer = keras.optimizers.ADAM(learning_rate = 0.001),
    loss = 'sparse_categorical_entropy',
    metrics = ['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose = 1, patience = 2, min_lr=0.00001)
EPOCHS = 70
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train,
    validation_split = 0.10,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [early_stop, reduce_lr]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n ACCURACY: {test_acc*100:.2f}%")

os.makedirs(MODELS_DIR, exist_ok=True)
model.save(os.path.join(MODELS_DIR, 'action_model.h5'))
np.save(os.path.join(MODELS_DIR, 'labels.npy'), np.array(GESTURE_LIST))