#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer-based Image Classification using TensorFlow
This script implements a Vision Transformer (ViT) model for image classification
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt

# Import shared functions from the base classifier
from image_classifier import (
    setup_gpu, clean_image_data, load_and_prepare_data,
    plot_performance, evaluate_model, test_on_image, save_model, load_saved_model
)

class Patches(layers.Layer):
    """Split images into patches"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    """Encode patches with positional embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def build_vit_model(
    input_shape=(256, 256, 3),
    patch_size=16,
    num_classes=1,  # Binary classification
    projection_dim=64,
    transformer_layers=4,
    num_heads=4,
    transformer_units=[128, 64],
    mlp_head_units=[256, 128],
):
    """Build a Vision Transformer (ViT) model"""
    print("Building Vision Transformer model...")
    
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    patches = Patches(patch_size)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(transformer_units[1], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Global average pooling
    representation = layers.GlobalAveragePooling1D()(representation)
    
    # MLP head
    for units in mlp_head_units:
        representation = layers.Dense(units, activation=tf.nn.gelu)(representation)
        representation = layers.Dropout(0.1)(representation)
    
    # Final classification layer (sigmoid for binary classification)
    outputs = layers.Dense(num_classes, activation="sigmoid")(representation)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    
    model.summary()
    
    return model

def train_transformer_model(model, train_data, val_data, epochs=20):
    """Train the transformer model with learning rate scheduling"""
    print(f"Training transformer model for {epochs} epochs...")
    
    # Create callbacks
    logdir = 'logs/transformer'
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
    
    # Train model
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )
    
    return history

def main():
    """Main function to run the transformer-based image classification pipeline"""
    # Setup
    setup_gpu()
    
    # Clean data (optional, can be commented out if data is already clean)
    # clean_image_data()
    
    # Load and prepare data
    train_data, val_data, test_data = load_and_prepare_data()
    
    # Build transformer model
    model = build_vit_model()
    
    # Train model
    history = train_transformer_model(model, train_data, val_data, epochs=30)
    
    # Plot performance
    plot_performance(history)
    
    # Evaluate model
    evaluate_model(model, test_data)
    
    # Test on a sample image
    test_on_image(model, '154006829.jpg')
    
    # Save model
    save_model(model, 'models/transformer_classifier.h5')
    
    print("Transformer-based image classification pipeline completed successfully!")

if __name__ == "__main__":
    main()
