# Age and Gender Classification using CNNs

This repository contains implementations of convolutional neural networks (CNNs) for age and gender classification using facial images from the UTKFace dataset.

## Models

### Age Classification Model
- Uses a CNN architecture with separable convolutions
- Predicts age as a continuous value using regression
- Features:
  - Input image size: 200x200x3
  - 6 convolutional blocks with batch normalization and LeakyReLU
  - Dense layers with dropout for regularization 
  - Outputs lower and upper age bounds
  - Mean Absolute Error loss function

### Gender Classification Model
- CNN for binary classification of gender (male/female)
- Architecture:
  - Input image size: 200x200x3
  - 5 convolutional blocks with batch normalization
  - Dense layers with dropout
  - Softmax output layer
  - Categorical crossentropy loss

## Dataset
- Uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/)
- Images are preprocessed to 200x200 pixels
- Training/validation split: 70%/30%
- Data augmentation with random flips and rotations

## Training
- Models trained using Adam optimizer
- Early stopping and model checkpointing
- TensorBoard logging for monitoring training
- Learning rate scheduling

## Results
The models achieve:
- Age prediction: MAE of ~5.5 years
- Gender classification: ~90% accuracy

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

## Usage
The Jupyter notebooks contain the complete implementation:
- `age_class_model.ipynb`: Age regression model
- `gender_class_model(UTK_Crop).ipynb`: Gender classification model

Models can be exported to TFLite format for mobile/edge deployment.

