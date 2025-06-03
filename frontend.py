import os
import numpy as np
import io
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = './sugar_level_model.keras'  # Use relative path
model = load_model(model_path)

# Parameters for image processing
image_size = (64, 64)  # Resize all images to 64x64

@app.route('/')
def index():
    return render_template('index.html')  # HTML file for the frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Convert FileStorage to a file-like object
        file_stream = io.BytesIO(file.read())

        # Load and preprocess the image
        img = load_img(file_stream, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array).flatten()[0]
        return jsonify({'predicted_sugar_level': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
