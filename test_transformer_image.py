#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test a transformer image using the trained multi-class model
"""

import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import functions from multiclass_classifier
from multiclass_classifier import (
    load_and_prepare_multiclass_data, test_multiclass_on_image,
    Patches, PatchEncoder  # Import custom layers
)

def main():
    """Main function to test a transformer image"""
    # Get class names from the data directory
    _, _, _, class_names = load_and_prepare_multiclass_data()
    
    # Load the trained model
    model_path = 'models/multiclass_transformer.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please train the model first using run_multiclass_classifier.py")
        return
    
    # Create a custom object scope with our custom layers
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
    
    # Load the model with custom objects
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    # Test image path
    test_image = 'data/transformers/166338adsa5399687_scale1.png'
    if not os.path.exists(test_image):
        print(f"Error: Test image {test_image} not found.")
        return
    
    # Test the image
    print(f"Testing image: {test_image}")
    test_multiclass_on_image(model, test_image, class_names)

if __name__ == "__main__":
    main()
