#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class Image Classification using TensorFlow
This script extends the binary classifier to handle multiple classes including 'transformers'
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.metrics import Precision, Recall, CategoricalAccuracy
import matplotlib.pyplot as plt

# Import shared functions from the base classifier
from image_classifier import (
    setup_gpu, clean_image_data, plot_performance, test_on_image
)

def load_and_prepare_multiclass_data(data_dir='data'):
    """Load, scale, and split the image data for multi-class classification"""
    print("Loading and preparing multi-class data...")
    # Load data
    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical'  # Use categorical labels for multi-class
    )
    
    # Get class names
    class_names = data.class_names
    print(f"Classes found: {class_names}")
    
    # Scale data
    data = data.map(lambda x, y: (x/255, y))
    
    # Split data
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1)
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test, class_names

def build_multiclass_cnn_model(num_classes):
    """Build a CNN model for multi-class image classification"""
    print(f"Building CNN model for {num_classes} classes...")
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # softmax for multi-class
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()
    
    return model

# Custom Patches layer for Keras
class Patches(keras.layers.Layer):
    """Split images into patches"""
    def __init__(self, patch_size, trainable=True, dtype=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.reshape = None
        self.cropping = None
        
    def build(self, input_shape):
        self.cropping = keras.layers.Cropping2D((0, 0))
        self.reshape = keras.layers.Reshape((-1, self.patch_size * self.patch_size * 3))
        super().build(input_shape)
        
    def call(self, images):
        # Use Keras operations instead of direct TF operations
        patches = self.reshape(self.cropping(images))
        return patches
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

# Custom PatchEncoder layer for Keras
class PatchEncoder(keras.layers.Layer):
    """Encode patches with positional embeddings"""
    def __init__(self, num_patches, projection_dim, trainable=True, dtype=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = None
        self.position_embedding = None
        
    def build(self, input_shape):
        self.projection = keras.layers.Dense(units=self.projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )
        super().build(input_shape)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

def build_multiclass_transformer_model(num_classes, input_shape=(256, 256, 3)):
    """Build a Vision Transformer model for multi-class classification"""
    print(f"Building Vision Transformer model for {num_classes} classes...")
    
    # Use the same architecture but modify the output layer for multi-class
    patch_size = 16
    projection_dim = 64
    transformer_layers = 4
    num_heads = 4
    transformer_units = [128, 64]
    mlp_head_units = [256, 128]
    
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Create input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Extract patches
    reshaped = keras.layers.Reshape(
        (input_shape[0] // patch_size, input_shape[1] // patch_size, 
         patch_size * patch_size * input_shape[2])
    )(inputs)
    patches = keras.layers.Reshape(
        ((input_shape[0] // patch_size) * (input_shape[1] // patch_size), 
         patch_size * patch_size * input_shape[2])
    )(reshaped)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = keras.layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = keras.layers.Dropout(0.1)(x3)
        x3 = keras.layers.Dense(transformer_units[1], activation=tf.nn.gelu)(x3)
        x3 = keras.layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = keras.layers.Add()([x3, x2])

    # Layer normalization
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Global average pooling
    representation = keras.layers.GlobalAveragePooling1D()(representation)
    
    # MLP head
    for units in mlp_head_units:
        representation = keras.layers.Dense(units, activation=tf.nn.gelu)(representation)
        representation = keras.layers.Dropout(0.1)(representation)
    
    # Final classification layer (softmax for multi-class classification)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(representation)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    
    model.summary()
    
    return model

def train_multiclass_model(model, train_data, val_data, epochs=20, is_transformer=False):
    """Train the multi-class model"""
    print(f"Training multi-class model for {epochs} epochs...")
    
    if is_transformer:
        # Create callbacks for transformer model
        logdir = 'logs/multiclass_transformer'
        os.makedirs(logdir, exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=logdir),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
    else:
        # Standard callbacks for CNN model
        logdir = 'logs/multiclass_cnn'
        os.makedirs(logdir, exist_ok=True)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir)]
    
    # Train model
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )
    
    return history

def evaluate_multiclass_model(model, test_data, class_names):
    """Evaluate multi-class model performance on test data"""
    print("Evaluating multi-class model...")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions for confusion matrix
    all_images = []
    all_labels = []
    all_predictions = []
    
    for images, labels in test_data:
        predictions = model.predict(images)
        all_images.extend(images.numpy())
        all_labels.extend(np.argmax(labels.numpy(), axis=1))
        all_predictions.extend(np.argmax(predictions, axis=1))
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    f1 = f1_score(all_labels, all_predictions, average=None)
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")
    
    # Calculate overall metrics
    print("\nOverall metrics:")
    print(f"Precision (macro): {precision_score(all_labels, all_predictions, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(all_labels, all_predictions, average='macro'):.4f}")
    print(f"F1-score (macro): {f1_score(all_labels, all_predictions, average='macro'):.4f}")

def test_multiclass_on_image(model, image_path, class_names):
    """Test the multi-class model on a single image"""
    print(f"Testing multi-class model on image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
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
    
    # Get the predicted class
    predicted_class_idx = np.argmax(yhat[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = yhat[0][predicted_class_idx]
    
    print(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
    
    # Show all class probabilities
    plt.figure(figsize=(10, 5))
    
    # Check if the number of classes matches the prediction array
    if len(class_names) == len(yhat[0]):
        plt.bar(class_names, yhat[0])
    else:
        print(f"Warning: Mismatch between class_names ({len(class_names)}) and prediction array ({len(yhat[0])})")
        # Create generic class names if needed
        generic_names = [f"Class {i}" for i in range(len(yhat[0]))]
        plt.bar(generic_names, yhat[0])
        # Map the predicted class index to the actual class name
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class {predicted_class_idx}"
    
    plt.title("Class Probabilities")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return predicted_class

def save_multiclass_model(model, path='models/multiclass_classifier.h5'):
    """Save the trained multi-class model"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def load_saved_multiclass_model(path='models/multiclass_classifier.h5'):
    """Load a saved multi-class model"""
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model

def main():
    """Main function to run the multi-class image classification pipeline"""
    # Setup
    setup_gpu()
    
    # Choose model type (CNN or Transformer)
    use_transformer = True  # Set to True to use transformer model
    
    # Clean data (optional, can be commented out if data is already clean)
    # clean_image_data()
    
    # Load and prepare data
    train_data, val_data, test_data, class_names = load_and_prepare_multiclass_data()
    
    # Build model
    num_classes = len(class_names)
    if use_transformer:
        model = build_multiclass_transformer_model(num_classes)
    else:
        model = build_multiclass_cnn_model(num_classes)
    
    # Train model
    history = train_multiclass_model(model, train_data, val_data, epochs=30, is_transformer=use_transformer)
    
    # Plot performance
    plot_performance(history)
    
    # Evaluate model
    evaluate_multiclass_model(model, test_data, class_names)
    
    # Test on a sample image (if available)
    sample_images = {
        'happy': 'data/happy/154006829.jpg',
        'sad': 'data/sad/sad-people.jpg',
        'transformers': 'data/transformers/166338adsa5399687_scale1.png'
    }
    
    for class_name, image_path in sample_images.items():
        if os.path.exists(image_path):
            print(f"\nTesting on {class_name} image:")
            test_multiclass_on_image(model, image_path, class_names)
    
    # Save model
    model_path = 'models/multiclass_transformer.h5' if use_transformer else 'models/multiclass_cnn.h5'
    save_multiclass_model(model, model_path)
    
    print("Multi-class image classification pipeline completed successfully!")

if __name__ == "__main__":
    main()
