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
import tifffile  # For reading TIFF images
import matplotlib.pyplot as plt
import imageio
import matplotlib

matplotlib.use('Agg')

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This disables all GPUs

# Check if GPU is available (it should now show no GPU devices)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected but disabled, running on CPU.")
else:
    print("Running on CPU.")


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
    "segmentation": os.path.join(BASE_MODEL_DIR, "Segmentation.keras")
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


# Function to handle segmentation
def process_segmentation():
    # Load the Segmentation model
    model = load_model(r"C:\Users\Viorel\Desktop\web\unihack_2024\models\Segmentation.keras")

    # Load data function
    def load_data():
        # Load the TIFF images
        custom_img = tifffile.imread(r"C:\Users\Viorel\Desktop\web\unihack_2024\models\testing.tif")
        custom_labels = tifffile.imread(r"C:\Users\Viorel\Desktop\web\unihack_2024\models\testing_groundtruth.tif")

        # Normalize the images to [0, 1]
        custom_img = custom_img.astype(np.float32) / np.max(custom_img)
        # Ensure labels are binary (0 or 1)
        custom_labels = (custom_labels > 0).astype(np.float32)

        # Add channel dimension for compatibility with Keras (depth, height, width, channels)
        custom_img = np.expand_dims(custom_img, axis=-1)
        custom_labels = np.expand_dims(custom_labels, axis=-1)

        return custom_img, custom_labels

    # Load the data
    current_img, current_labels = load_data()

    # Resize images to (256, 256) and add a channel dimension for grayscale (1 channel)
    current_img = tf.image.resize(current_img, (256, 256))
    current_labels = tf.image.resize(current_labels, (256, 256))

    current_img = np.expand_dims(current_img, axis=-1)  # Adds the channel dimension

    # Ensure labels are clipped between 0 and 1
    current_labels = np.clip(current_labels, 0, 1)

    # Check if the shapes are now correct
    print(current_img.shape)  # Should be (num_samples, 256, 256, 1)
    print(current_labels.shape)  # Should be (num_samples, 256, 256, 1)

    # Predict on test images
    predicted_labels = model.predict(current_img)
    # Save each slice as a frame in a GIF
    frames = []
    for i in range(predicted_labels.shape[0]):  # Loop through the images in the batch
        fig, ax = plt.subplots()
        ax.imshow(predicted_labels[i, :, :, 0], cmap='gray')  # Adjust the indices based on shape
        ax.axis('off')

        # Save frame to a file-like object for creating the GIF
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Create a GIF
    gif_path = 'segmentation_resultsfinal2.gif'
    imageio.mimsave(gif_path, frames, fps=5)

    print(f"GIF saved at: {gif_path}")

    return gif_path


# Route for handling POST requests
@app.route('/process_image', methods=['POST'])
def process_image():
    # Retrieve model name from form data
    model_name = request.form.get('model_name', "kidney-cancer")
    if model_name not in model_paths:
        return jsonify({"error": "Invalid model name"}), 400

    # If the model is 'segmentation', directly process the segmentation
    if model_name == "segmentation":
        try:
            gif_path = process_segmentation()  # Call the segmentation process directly
            return send_file(gif_path, mimetype='image/gif', as_attachment=True,
                             download_name='segmentation_resultsfinal2.gif')
        except Exception as e:
            logging.error(f"Error during segmentation: {str(e)}")
            return jsonify({"error": "Error during segmentation"}), 500

    # For other models, handle image file upload
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

    return send_file(output_path, mimetype='image/jpeg', as_attachment=True,
                     download_name=f"{model_name}_gradcam_result.jpg")


if __name__ == "__main__":
    app.run(debug=True)
