#Image Classification using Modified ResNet50V2

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project implements a state-of-the-art image classification model using transfer learning with the ResNet50V2 architecture. The model is trained on a custom dataset with four classes, demonstrating high accuracy and robustness in image classification tasks.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Project Structure](#project-structure)

## Features

- Transfer learning using ResNet50V2 pre-trained on ImageNet
- Custom dataset handling and preprocessing
- Data augmentation for improved model generalization
- Fine-tuning of the pre-trained model
- Detailed visualization of training progress and results
- Confusion matrix and classification report generation

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- seaborn
## Dataset

The dataset used in this project is located in the `Dataset_3/Case3_th0.1_Qscaled` directory. It contains images divided into four classes:

1. Class A
2. Class B
3. Class C
4. Class D

The dataset is automatically split into training (80%) and validation (20%) sets during the execution of the script.

## Usage

To run the image classification script:

```bash
python image_classification.py
```

This will:
1. Load and preprocess the data
2. Create and train the model
3. Fine-tune the model
4. Evaluate the model and display results

You can modify the hyperparameters and model architecture in the `image_classification.py` file to experiment with different configurations.

## Model Architecture

The model uses transfer learning with the ResNet50V2 architecture:

1. Data Augmentation Layer
   - Random horizontal flip
   - Random rotation (±10%)
   - Random zoom (±10%)
2. Pre-trained ResNet50V2 base (weights from ImageNet)
3. Global Average Pooling
4. Dense layer (256 units, ReLU activation)
5. Dropout (0.5)
6. Output Dense layer (4 units, softmax activation)

## Training Process

The training process consists of two phases:

1. Initial training (10 epochs)
   - Base model is frozen
   - Only top layers are trained

2. Fine-tuning (25 epochs)
   - Base model is partially unfrozen
   - Entire model is trained with a lower learning rate

Both phases implement early stopping and learning rate reduction to prevent overfitting.

## Results

The script generates and displays:

- Training and validation accuracy/loss plots
- Confusion matrix
- Detailed classification report

These results help in understanding the model's performance and identifying areas for improvement.

## Project Structure

```
image-classification-project/
│
├── image_classification.py   # Main script
├── README.md                 # Project documentation
├── requirements.txt          # Required Python packages
│
└── Dataset_3/
    └── Case3_th0.1_Qscaled/  # Dataset directory
        ├── Class A/
        ├── Class B/
        ├── Class C/
        └── Class D/
```

---

