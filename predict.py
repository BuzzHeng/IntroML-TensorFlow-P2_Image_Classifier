# predict.py

# Basic usage:
# python predict.py ./test_images/orange_dahlia.jpg my_model.h5

# Return the top 3 most likely classes:
# python predict.py ./test_images/orange_dahlia.jpg my_model.h5 --top_k 3

# Use a label_map.json file to map labels to flower names:
# python predict.py ./test_images/orange_dahlia.jpg my_model.h5 --category_names label_map.json

import os
import warnings
# Set environment variables to suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

import tensorflow_hub as hub
import tf_keras as keras
import argparse
import json
import numpy as np
from PIL import Image

# Load the model
def load_model(model_path):
    model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model
    
# Preprocess image
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()
    
# Map labels to names
def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names
    
def predict(image_path, model, top_k):
    # Load and preproces image
    image = Image.open(image_path)
    image = np.expand_dims(process_image(np.array(image)), axis=0)
    # Predict
    predictions = model.predict(image)
    probs, class_indices = tf.math.top_k(predictions, k=top_k)
    probs = probs.numpy().flatten()
    class_indices = class_indices.numpy().flatten()
    return probs, class_indices
    
def main():
    parser = argparse.ArgumentParser(description="Predict flower name from image.")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to flower names")
    args = parser.parse_args()
    
    # Load the model from the checkpoint
    model = load_model(args.model_path)

    # Predict
    probs, class_indices = predict(args.image_path, model, args.top_k)
    
    # Map indices to class names if category_names is provided
    if args.category_names:
        class_names = load_class_names(args.category_names)
        classes = [class_names.get(str(index), "Unknown") for index in class_indices]
    else: 
        classes = class_indices  

    # Display the results
    print("Top K predictions: ")
    for prob, class_name in zip(probs, classes):
        print(f"Flower Class: {class_name}, Probability: {prob:.3f}")

if __name__ == '__main__':
    main()