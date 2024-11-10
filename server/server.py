from flask_cors import CORS  # Import CORS
from flask import Flask, request, jsonify, send_file
import os
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tempfile

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Define the base directory for models
BASE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Dictionary to map model names to file paths
model_paths = {
    "alzheimer": os.path.join(BASE_MODEL_DIR, "AlzheimerResNet152.keras"),
    "malaria": os.path.join(BASE_MODEL_DIR, "MalariaResnet50.keras"),
    "covid": os.path.join(BASE_MODEL_DIR, "CovidResNet152.keras"),
    "retinal-imaging": os.path.join(BASE_MODEL_DIR, "retinal_imaging_resnet152.keras"),
    "kidney-cancer": os.path.join(BASE_MODEL_DIR, "KidneyCancerResnet50.keras"),
}

# Function to load the selected model
def load_selected_model(model_name):
    try:
        logging.info(f"Attempting to load model: {model_name}")
        if model_name in model_paths:
            model = load_model(model_paths[model_name])
            logging.info(f"Model {model_name} loaded successfully.")
            return model
        else:
            raise ValueError(f"Invalid model name provided: {model_name}")
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise RuntimeError(f"Error loading model {model_name}: {str(e)}")

# Function to generate Grad-CAM heatmap
def generate_grad_cam(model, img_array, last_conv_layer_name="conv5_block3_out", pred_index=None):
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_model(img_array)
        tape.watch(last_conv_layer_output)
        predictions = classifier_model(last_conv_layer_output)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Function to overlay the heatmap on the image
def overlay_heatmap(heatmap, img_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

# Route for handling POST requests
@app.route('/process_image', methods=['POST'])
def process_image():
    # Retrieve model name from form data
    model_name = request.form.get('model_name', "kidney-cancer")
    if model_name not in model_paths:
        return jsonify({"error": "Invalid model name"}), 400

    # Retrieve the image file from request files
    if 'image_file' not in request.files:
        return jsonify({"error": "Image file is missing"}), 400
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    image_file.save(temp_file.name)
    img_path = temp_file.name

    # Load the selected model
    try:
        model = load_selected_model(model_name)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Get the top prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    prediction_confidence = predictions[0][pred_index]

    # Generate Grad-CAM heatmap
    heatmap = generate_grad_cam(model, img_array)
    processed_img = overlay_heatmap(heatmap, img_path)

    # Save the processed image temporarily
    output_file = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    output_path = output_file.name
    cv2.imwrite(output_path, processed_img)

    # Send processed image as response
    return send_file(output_path, mimetype='image/jpeg', as_attachment=True, download_name='processed_image.jpeg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
