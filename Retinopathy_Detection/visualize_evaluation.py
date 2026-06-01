import matplotlib.pyplot as plt
import numpy as np
import os

# ── Metrics Data (extracted from the photo) ──────────────────────────────────
# Summary Metrics
metrics = {
    'Accuracy': 0.9875,
    'Precision': 0.9756,
    'Recall': 1.0000,
    'F1-Score': 0.9877
}

# Confusion Matrix: [[TN, FP], [FN, TP]] where DR=0, No_DR=1
# From photo: [[78, 2], [0, 80]]
cm = np.array([[78, 2], [0, 80]])
labels = ['DR', 'No_DR']

# ── Ensure output directory exists ───────────────────────────────────────────
output_dir = "static/plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ── 1. Accuracy Graph (Bar Chart) ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics.keys(), metrics.values(), color=['#4361ee', '#3f37c9', '#4895ef', '#4cc9f0'])
plt.ylim(0.9, 1.01)  # Focus on the high-performance range
plt.ylabel('Score')
plt.title('Model Performance Metrics (Summary)')

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'summary_metrics.png'))
print(f"Summary metrics graph saved to {os.path.join(output_dir, 'summary_metrics.png')}")
plt.close()

# ── 2. Confusion Matrix Heatmap ────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

# Add annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=14, fontweight='bold')

plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
print(f"Confusion matrix graph saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
plt.close()

print("\nAll visualizations generated successfully!")
