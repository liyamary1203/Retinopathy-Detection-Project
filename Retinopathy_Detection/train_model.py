import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ── Image parameters ──────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"

# ── Data generators ───────────────────────────────────────────────────────────
# EfficientNet has its own internal preprocessing (expects 0-255 input)
# We use preprocess_input via preprocessing_function
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Validation: EfficientNet preprocessing ONLY — no augmentation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Class labels:", train_data.class_indices)

# ── Class weights (handle imbalance) ──────────────────────────────────────────
counter     = dict(zip(*np.unique(train_data.classes, return_counts=True)))
total       = sum(counter.values())
class_weight = {cls: total / (len(counter) * count) for cls, count in counter.items()}
print("Class weights:", class_weight)

# ── Build model ───────────────────────────────────────────────────────────────
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # Phase 1: freeze entire base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    "retinopathy_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# ── PHASE 1: Train classification head ────────────────────────────────────────
print("\n===== PHASE 1: Training classification head (base frozen) =====\n")

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop_p1 = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stop_p1]
)

# ── PHASE 2: Fine-tune top layers ─────────────────────────────────────────────
print("\n===== PHASE 2: Fine-tuning top 40 layers of EfficientNetB0 =====\n")

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-8,
    verbose=1
)

early_stop_p2 = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, reduce_lr, early_stop_p2]
)

print("\nTraining finished and best model saved to retinopathy_model.h5")

# ── Visualization ─────────────────────────────────────────────────────────────
print("\nGenerating training history visualizations...")

# Ensure output directory exists
output_dir = "static/training"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Helper function to combine nested history dictionaries
def combine_history(h1, h2):
    combined = {}
    for key in h1.history.keys():
        combined[key] = h1.history[key] + h2.history[key]
    return combined

try:
    full_history = combine_history(history1, history2)
    epochs = range(1, len(full_history['accuracy']) + 1)

    # 🔹 Accuracy Plot Only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, full_history['accuracy'], 'b-', marker='o', label='Training Accuracy')
    plt.plot(epochs, full_history['val_accuracy'], 'r-', marker='s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('accuracy_plot.png')
    print("Accuracy plot successfully saved as 'accuracy_plot.png'")
    # Note: plt.show() removed to prevent script blocking; use the saved file.

except Exception as e:
    print(f"Failed to generate accuracy graph: {e}")