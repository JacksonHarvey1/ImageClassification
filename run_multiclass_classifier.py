#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class Image Classification Runner Script
This script provides a command-line interface to run the multi-class image classification model.
"""

import argparse
import os
import sys

def main():
    """Main function to parse arguments and run the multi-class classifier"""
    parser = argparse.ArgumentParser(description='Run multi-class image classification model')
    
    # Model selection
    parser.add_argument('--model', type=str, default='transformer', choices=['cnn', 'transformer'],
                        help='Model type to use (cnn or transformer)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
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
    parser.add_argument('--test-image', type=str,
                        help='Path to an image for testing')
    
    # Model saving/loading
    parser.add_argument('--model-path', type=str, 
                        help='Path to save/load model (default depends on model type)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default model path if not provided
    if not args.model_path:
        if args.model == 'cnn':
            args.model_path = 'models/multiclass_cnn.h5'
        else:
            args.model_path = 'models/multiclass_transformer.h5'
    
    # Import appropriate modules
    from multiclass_classifier import (
        setup_gpu, clean_image_data, load_and_prepare_multiclass_data,
        build_multiclass_cnn_model, build_multiclass_transformer_model,
        train_multiclass_model, plot_performance, evaluate_multiclass_model,
        test_multiclass_on_image, save_multiclass_model, load_saved_multiclass_model
    )
    
    # Setup GPU
    setup_gpu()
    
    # Clean data if requested
    if args.clean_data:
        clean_image_data(data_dir=args.data_dir)
    
    # Load and prepare data
    train_data, val_data, test_data, class_names = load_and_prepare_multiclass_data(data_dir=args.data_dir)
    
    # Either load or train model
    if args.no_train and os.path.exists(args.model_path):
        print(f"Loading saved model from {args.model_path}")
        model = load_saved_multiclass_model(args.model_path)
    else:
        # Build model
        num_classes = len(class_names)
        if args.model == 'cnn':
            print("Using CNN-based multi-class classifier")
            model = build_multiclass_cnn_model(num_classes)
            is_transformer = False
        else:
            print("Using transformer-based multi-class classifier")
            model = build_multiclass_transformer_model(num_classes)
            is_transformer = True
        
        # Train model
        history = train_multiclass_model(
            model, train_data, val_data, 
            epochs=args.epochs, 
            is_transformer=is_transformer
        )
        
        # Plot performance
        plot_performance(history)
        
        # Save model
        print(f"Saving model to {args.model_path}...")
        save_multiclass_model(model, args.model_path)
        
        # Verify model was saved
        if os.path.exists(args.model_path):
            print(f"Model successfully saved to {args.model_path}")
        else:
            print(f"Error: Failed to save model to {args.model_path}")
    
    # Evaluate model
    evaluate_multiclass_model(model, test_data, class_names)
    
    # Test on a sample image if provided
    if args.test_image and os.path.exists(args.test_image):
        print(f"\nTesting on provided image: {args.test_image}")
        test_multiclass_on_image(model, args.test_image, class_names)
    else:
        # Test specifically on a transformer image
        transformer_dir = os.path.join(args.data_dir, 'transformers')
        if os.path.exists(transformer_dir):
            files = os.listdir(transformer_dir)
            if files:
                transformer_image = os.path.join(transformer_dir, files[0])
                print(f"\nTesting on transformer image: {transformer_image}")
                test_multiclass_on_image(model, transformer_image, class_names)
                
        # Test on sample images from each class
        print("\nTesting on sample images from each class:")
        for class_name in class_names:
            # Skip transformers since we already tested it
            if class_name == 'transformers':
                continue
                
            # Try to find a sample image for this class
            class_dir = os.path.join(args.data_dir, class_name)
            if os.path.exists(class_dir):
                files = os.listdir(class_dir)
                if files:
                    sample_image = os.path.join(class_dir, files[0])
                    print(f"\nTesting on {class_name} image: {sample_image}")
                    test_multiclass_on_image(model, sample_image, class_names)
    
    print("Multi-class image classification completed successfully!")

if __name__ == "__main__":
    main()
