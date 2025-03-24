#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Classification using TensorFlow
This script implements a binary image classifier to categorize images as 'happy' or 'sad'
"""

import os
import cv2
import numpy as np
import imghdr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

def setup_gpu():
    """Configure GPU for optimal use"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

def clean_image_data(data_dir='data', image_exts=['jpeg', 'jpg', 'bmp', 'png']):
    """Remove corrupted or invalid images from the dataset"""
    print("Cleaning image data...")
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f'Image not in ext list {image_path}')
                    os.remove(image_path)
            except Exception as e:
                print(f'Issue with image {image_path}: {e}')
                # os.remove(image_path)

def load_and_prepare_data(data_dir='data'):
    """Load, scale, and split the image data"""
    print("Loading and preparing data...")
    # Load data
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    
    # Scale data
    data = data.map(lambda x, y: (x/255, y))
    
    # Split data
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1)
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test

def build_cnn_model():
    """Build a CNN model for image classification"""
    print("Building CNN model...")
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()
    
    return model

def build_transformer_model():
    """
    Build a transformer-based model for image classification
    This imports the Vision Transformer (ViT) implementation from transformer_classifier.py
    """
    try:
        from transformer_classifier import build_vit_model
        return build_vit_model()
    except ImportError as e:
        print(f"Error importing transformer model: {e}")
        print("Falling back to CNN model.")
        return build_cnn_model()

def train_model(model, train_data, val_data, epochs=20, is_transformer=False):
    """Train the model"""
    print(f"Training model for {epochs} epochs...")
    
    if is_transformer:
        try:
            from transformer_classifier import train_transformer_model
            return train_transformer_model(model, train_data, val_data, epochs)
        except ImportError:
            print("Could not import transformer training function. Using standard training.")
    
    # Standard training for CNN model
    logdir = 'logs/cnn'
    os.makedirs(logdir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    history = model.fit(
        train_data, 
        epochs=epochs, 
        validation_data=val_data, 
        callbacks=[tensorboard_callback]
    )
    
    return history

def plot_performance(history):
    """Plot training and validation metrics"""
    print("Plotting performance metrics...")
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='teal', label='loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    plt.title('Loss', fontsize=15)
    plt.legend(loc="upper right")
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    plt.title('Accuracy', fontsize=15)
    plt.legend(loc="upper left")
    
    plt.show()

def evaluate_model(model, test_data):
    """Evaluate model performance on test data"""
    print("Evaluating model...")
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()
    
    for batch in test_data.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        accuracy.update_state(y, yhat)
    
    print(f"Precision: {precision.result().numpy():.4f}")
    print(f"Recall: {recall.result().numpy():.4f}")
    print(f"Accuracy: {accuracy.result().numpy():.4f}")

def test_on_image(model, image_path):
    """Test the model on a single image"""
    print(f"Testing model on image: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    
    # Resize and preprocess
    resize = tf.image.resize(img, (256, 256))
    plt.subplot(1, 2, 2)
    plt.imshow(resize.numpy().astype(int))
    plt.title("Resized Image (256x256)")
    plt.show()
    
    # Make prediction
    yhat = model.predict(np.expand_dims(resize/255, 0))
    
    if yhat > 0.5:
        prediction = "Sad"
    else:
        prediction = "Happy"
    
    print(f"Prediction: {prediction} (confidence: {abs(yhat[0][0] - 0.5) * 2:.2f})")
    return prediction

def save_model(model, path='models/imageclassifier.h5'):
    """Save the trained model"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def load_saved_model(path='models/imageclassifier.h5'):
    """Load a saved model"""
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model

def main():
    """Main function to run the image classification pipeline"""
    # Setup
    setup_gpu()
    
    # Choose model type (CNN or Transformer)
    use_transformer = False  # Set to True to use transformer model
    
    # Clean data (optional, can be commented out if data is already clean)
    clean_image_data()
    
    # Load and prepare data
    train_data, val_data, test_data = load_and_prepare_data()
    
    # Build model
    if use_transformer:
        model = build_transformer_model()
    else:
        model = build_cnn_model()
    
    # Train model
    history = train_model(model, train_data, val_data, epochs=20, is_transformer=use_transformer)
    
    # Plot performance
    plot_performance(history)
    
    # Evaluate model
    evaluate_model(model, test_data)
    
    # Test on a sample image
    test_on_image(model, '154006829.jpg')
    
    # Save model
    save_model(model)
    
    print("Image classification pipeline completed successfully!")

if __name__ == "__main__":
    main()
