#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Classification Runner Script
This script provides a command-line interface to run either the CNN-based or 
transformer-based image classification models.
"""

import argparse
import os
import sys

def main():
    """Main function to parse arguments and run the appropriate classifier"""
    parser = argparse.ArgumentParser(description='Run image classification models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer'],
                        help='Model type to use (cnn or transformer)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training and load saved model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the image data')
    parser.add_argument('--clean-data', action='store_true',
                        help='Run data cleaning step')
    
    # Testing parameters
    parser.add_argument('--test-image', type=str, default='154006829.jpg',
                        help='Path to an image for testing')
    
    # Model saving/loading
    parser.add_argument('--model-path', type=str, 
                        help='Path to save/load model (default depends on model type)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default model path if not provided
    if not args.model_path:
        if args.model == 'cnn':
            args.model_path = 'models/imageclassifier.h5'
        else:
            args.model_path = 'models/transformer_classifier.h5'
    
    # Import appropriate modules
    if args.model == 'cnn':
        print("Using CNN-based image classifier")
        from image_classifier import (
            setup_gpu, clean_image_data, load_and_prepare_data, build_cnn_model,
            train_model, plot_performance, evaluate_model, test_on_image,
            save_model, load_saved_model
        )
        build_model_fn = build_cnn_model
        is_transformer = False
    else:
        print("Using transformer-based image classifier")
        from transformer_classifier import (
            setup_gpu, clean_image_data, load_and_prepare_data, build_vit_model,
            train_transformer_model, plot_performance, evaluate_model, test_on_image,
            save_model, load_saved_model
        )
        build_model_fn = build_vit_model
        is_transformer = True
    
    # Setup GPU
    setup_gpu()
    
    # Clean data if requested
    if args.clean_data:
        clean_image_data(data_dir=args.data_dir)
    
    # Load and prepare data
    train_data, val_data, test_data = load_and_prepare_data(data_dir=args.data_dir)
    
    # Either load or train model
    if args.no_train and os.path.exists(args.model_path):
        print(f"Loading saved model from {args.model_path}")
        model = load_saved_model(args.model_path)
    else:
        # Build model
        model = build_model_fn()
        
        # Train model
        if is_transformer:
            history = train_transformer_model(model, train_data, val_data, epochs=args.epochs)
        else:
            history = train_model(model, train_data, val_data, epochs=args.epochs, is_transformer=is_transformer)
        
        # Plot performance
        plot_performance(history)
        
        # Save model
        save_model(model, args.model_path)
    
    # Evaluate model
    evaluate_model(model, test_data)
    
    # Test on a sample image
    if os.path.exists(args.test_image):
        test_on_image(model, args.test_image)
    else:
        print(f"Test image {args.test_image} not found. Skipping test.")
    
    print("Image classification completed successfully!")

if __name__ == "__main__":
    main()
