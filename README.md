# Image Classification with TensorFlow

This repository contains Python scripts for image classification using both CNN and Transformer-based models in TensorFlow. The project is designed to classify images into "happy" or "sad" categories.

## Project Structure

- `image_classifier.py`: CNN-based image classification implementation
- `transformer_classifier.py`: Vision Transformer (ViT) based image classification implementation
- `run_classifier.py`: Command-line interface to run either model
- `data/`: Directory containing training images
  - `happy/`: Images of happy people/scenes
  - `sad/`: Images of sad people/scenes
- `models/`: Directory for saved models
  - `imageclassifier.h5`: Saved CNN model
  - `transformer_classifier.h5`: Saved transformer model

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

You can install the required packages with:

```bash
pip install tensorflow tensorflow-gpu opencv-python matplotlib
```

## Usage

### Running the Classifier

The main script `run_classifier.py` provides a command-line interface to run either the CNN or transformer-based model:

```bash
# Run the CNN model (default)
python run_classifier.py

# Run the transformer model
python run_classifier.py --model transformer

# Run with custom parameters
python run_classifier.py --model transformer --epochs 30 --clean-data --test-image my_test_image.jpg
```

### Command-line Arguments

- `--model`: Model type to use (`cnn` or `transformer`, default: `cnn`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training (default: 32)
- `--no-train`: Skip training and load saved model
- `--data-dir`: Directory containing the image data (default: 'data')
- `--clean-data`: Run data cleaning step
- `--test-image`: Path to an image for testing (default: '154006829.jpg')
- `--model-path`: Path to save/load model (default depends on model type)

### Using Individual Scripts

You can also run the individual scripts directly:

```bash
# Run CNN-based classifier
python image_classifier.py

# Run transformer-based classifier
python transformer_classifier.py
```

## Model Details

### CNN Model

The CNN model is a simple convolutional neural network with the following architecture:
- 3 convolutional layers with max pooling
- Flatten layer
- Dense layer with 256 units
- Output layer with sigmoid activation for binary classification

### Transformer Model

The transformer model is based on the Vision Transformer (ViT) architecture:
- Splits images into patches
- Applies positional embeddings
- Processes through transformer blocks with multi-head attention
- Uses a classification head for binary prediction

## Data Preparation

The scripts expect data to be organized in the following structure:
```
data/
  happy/
    image1.jpg
    image2.jpg
    ...
  sad/
    image1.jpg
    image2.jpg
    ...
```

The data cleaning step removes any corrupted or invalid images from the dataset.

## Model Training and Evaluation

Both models are trained with similar pipelines:
1. Load and preprocess the data
2. Split into training, validation, and test sets
3. Train the model
4. Evaluate performance on the test set
5. Test on a sample image
6. Save the trained model

## Performance Visualization

During training, the scripts will display plots of:
- Training and validation loss
- Training and validation accuracy

After training, the model is evaluated on the test set, reporting:
- Precision
- Recall
- Accuracy

## Testing on New Images

You can test the trained models on new images using the `--test-image` parameter:

```bash
python run_classifier.py --model transformer --no-train --test-image path/to/your/image.jpg
```

This will load the saved model and make a prediction on the specified image.
