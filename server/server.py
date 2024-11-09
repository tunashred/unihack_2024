from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)

# Define the base directory for models
BASE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Dictionary to map model names to file paths
model_paths = {
    "AlzheimerResNet152": os.path.join(BASE_MODEL_DIR, "AlzheimerResNet152.keras"),
    "Malaria": os.path.join(BASE_MODEL_DIR, "MalariaResnet50.keras"),
    "Covid": os.path.join(BASE_MODEL_DIR, "CovidResNet152.keras"),
    "RetinalImagingResnet152": os.path.join(BASE_MODEL_DIR, "retinal_imaging_resnet152.keras"),
    "KidneyCancer": os.path.join(BASE_MODEL_DIR, "KidneyCancerResnet50.keras")
}

# Function to load the selected model
def load_selected_model(model_name):
    if model_name in model_paths:
        return load_model(model_paths[model_name])
    else:
        raise ValueError("Invalid model name provided")

# Load the default model
model = load_model(model_paths["KidneyCancer"])

# Function to generate Grad-CAM heatmap using tf.GradientTape
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

# Function to overlay the heatmap on the image and save the result
def overlay_heatmap(heatmap, img_path, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

# Route for handling GET requests
@app.route('/get_image', methods=['GET'])
def get_image():
    img_path = request.args.get('image_path')
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Invalid or missing image path"}), 400
    return jsonify({"message": "Image path received", "image_path": img_path})

# Route for handling POST requests
@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    model_name = data.get('model_name', "KidneyCancer")  # Default to "KidneyCancer" if not provided
    img_path = data.get('image_path')
    output_path = data.get('output_path', './processed_image.jpeg')  # Default output path

    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Invalid or missing image path"}), 400

    if model_name not in model_paths:
        return jsonify({"error": "Invalid model name"}), 400

    # Load the selected model
    model = load_selected_model(model_name)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Get the top prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])

    # Generate Grad-CAM heatmap
    heatmap = generate_grad_cam(model, img_array)

    # Overlay the heatmap on the image and save it
    overlay_heatmap(heatmap, img_path, output_path)

    return jsonify({"message": "Image processed successfully", "output_path": output_path})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
