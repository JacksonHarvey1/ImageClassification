#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classify a single image using the trained multi-class model
Usage: python classify_image.py path/to/image.jpg
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import custom layers
from multiclass_classifier import Patches, PatchEncoder

def classify_image(image_path):
    """Classify a single image using the trained model"""
    try:
        print(f"Classifying image: {image_path}")
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found.")
            return
        
        # Load the trained model
        model_path = 'models/multiclass_transformer.h5'
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return
        
        # Define class names (adjust based on your model)
        class_names = ['augmented_transformers', 'happy', 'sad']
        
        # Create a custom object scope with our custom layers
        custom_objects = {
            'Patches': Patches,
            'PatchEncoder': PatchEncoder
        }
        
        # Load the model with custom objects
        print("Loading model...")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load and preprocess the image
        print("Loading image...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the original image
        print("Displaying image...")
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        
        # Resize and preprocess
        resize = tf.image.resize(img, (256, 256))
        plt.subplot(1, 2, 2)
        plt.imshow(resize.numpy().astype(int))
        plt.title("Resized Image (256x256)")
        plt.tight_layout()
        plt.savefig('image_display.png')  # Save the figure in case display fails
        plt.show()
        
        # Make prediction
        print("Making prediction...")
        yhat = model.predict(np.expand_dims(resize/255, 0), verbose=1)
        print("Prediction complete!")
        
        # Get the predicted class
        predicted_class_idx = np.argmax(yhat[0])
        
        # Print prediction array for debugging
        print(f"Prediction array: {yhat[0]}")
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Number of classes in model output: {len(yhat[0])}")
        print(f"Number of class names: {len(class_names)}")
        
        # Check if the prediction array matches the number of classes
        if len(yhat[0]) != len(class_names):
            print(f"Warning: Mismatch between class_names ({len(class_names)}) and prediction array ({len(yhat[0])})")
            # Create generic class names for the prediction array
            generic_names = [f"Class {i}" for i in range(len(yhat[0]))]
            
            # Show all class probabilities with generic names
            plt.figure(figsize=(10, 5))
            plt.bar(generic_names, yhat[0])
            plt.title("Class Probabilities")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('class_probabilities.png')  # Save the figure in case display fails
            plt.show()
            
            # Map the predicted class index to a class name if possible
            if predicted_class_idx < len(class_names):
                predicted_class = class_names[predicted_class_idx]
            else:
                predicted_class = f"Class {predicted_class_idx}"
        else:
            # Show all class probabilities with actual class names
            plt.figure(figsize=(10, 5))
            plt.bar(class_names, yhat[0])
            plt.title("Class Probabilities")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('class_probabilities.png')  # Save the figure in case display fails
            plt.show()
            
            predicted_class = class_names[predicted_class_idx]
        
        confidence = yhat[0][predicted_class_idx]
        
        # Print the result
        print("\n" + "="*50)
        print(f"PREDICTION: {predicted_class}")
        print(f"CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
        print("="*50)
        
        # Determine if it's a transformer
        is_transformer = predicted_class in ['transformers', 'augmented_transformers']
        if is_transformer:
            print("\nThis image is classified as a TRANSFORMER.")
        else:
            print("\nThis image is NOT classified as a transformer.")
            
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py path/to/image.jpg")
        return
    
    # Get the image path from the command line argument
    image_path = sys.argv[1]
    
    # Classify the image
    classify_image(image_path)

if __name__ == "__main__":
    main()
