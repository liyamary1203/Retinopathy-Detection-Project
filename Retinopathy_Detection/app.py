import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Configurations
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model 
model_path = os.path.join(os.path.dirname(__file__), "retinopathy_model.h5")
print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded on the server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess image — must match training preprocessing
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)          
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)       
            
            # Predict
            prediction = model.predict(img_array)
            raw_value = float(prediction[0][0])
            
            # Class indices: {'DR': 0, 'No_DR': 1}
            # sigmoid output close to 0 → DR, close to 1 → No_DR
            if raw_value < 0.5:
                result = "DR (Diabetic Retinopathy detected)"
                status = "Unhealthy eye"
            else:
                result = "No DR"
                status = "Healthy eye"
                
            return jsonify({
                'success': True,
                'raw_value': round(raw_value, 4),
                'prediction': result,
                'status': status
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Failed to delete {filepath}: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
