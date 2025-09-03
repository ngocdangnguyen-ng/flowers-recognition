from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load model

# Dynamic model and class names loading
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_PATH = None
model = None
class_names = None


def load_model_and_classes():
    global model, class_names, MODEL_PATH
    try:
        # Find best model file
        model_files = []
        if os.path.exists(MODELS_PATH):
            for file in os.listdir(MODELS_PATH):
                if file.endswith('_best.h5'):
                    model_files.append(file)
        if not model_files:
            # Fallback: any .h5 file
            model_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.h5')]
        if not model_files:
            raise FileNotFoundError("No trained model found! Please run training first.")
        MODEL_PATH = os.path.join(MODELS_PATH, model_files[0])
        print(f"Loading model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load class names from processed/test
        test_path = os.path.join(PROCESSED_PATH, 'test')
        if os.path.exists(test_path):
            class_names = sorted([name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))])
        else:
            class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        print(f"Model loaded successfully! Classes: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        predictions = model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': class_names[idx],
                'confidence': float(predictions[idx]),
                'percentage': float(predictions[idx] * 100)
            })
        
        return jsonify({'success': True, 'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_and_classes()
    app.run(debug=True, host='0.0.0.0', port=5000)