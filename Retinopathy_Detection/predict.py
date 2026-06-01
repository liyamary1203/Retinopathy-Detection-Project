import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load trained model
model = tf.keras.models.load_model("retinopathy_model.h5")

# Path of image you want to test
img_path = "test_image.png"

# Load and preprocess image — must match training preprocessing
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)          # shape: (224, 224, 3), range 0-255
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)       # EfficientNet-specific preprocessing

# Predict
prediction = model.predict(img_array)

# Class indices: {'DR': 0, 'No_DR': 1}
# sigmoid output close to 0 → DR, close to 1 → No_DR
print(f"Raw prediction value: {prediction[0][0]:.4f}")
if prediction[0][0] < 0.5:
    print("Prediction: DR (Diabetic Retinopathy detected)")
    print("UnHealthy eye")
else:
    print("Prediction: No_DR ")
    print("Healthy eye")