import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = "retinopathy_model.h5"
TEST_DIR   = "dataset/test"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16

print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file {MODEL_PATH} not found!")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

# ── Data Preparation ──────────────────────────────────────────────────────────
print(f"Preparing test data from {TEST_DIR}...")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # CRITICAL: Do not shuffle for evaluation
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating model performance...")
predictions = model.predict(test_data, verbose=1)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_data.classes

target_names = list(test_data.class_indices.keys())

print("\n" + "="*50)
print("             CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n" + "="*50)
print("           CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=target_names))

# Print individual metrics clearly
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*50)
print("Summary Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*50)

# ── Visualization ─────────────────────────────────────────────────────────────
print("\nGenerating performance visualizations...")

# Ensure output directory exists
output_dir = "static/evaluation"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names)
plt.yticks(tick_marks, target_names)

# Add annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# 2. Performance Summary Bar Chart
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['#4361ee', '#3f37c9', '#4895ef', '#4cc9f0'])
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('Performance Metrics Summary')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'summary_metrics.png'))
plt.close()

print(f"Visualizations saved to {output_dir}/")
