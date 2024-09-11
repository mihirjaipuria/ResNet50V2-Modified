# 🖼️ Advanced Image Classification using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project implements a state-of-the-art image classification model using transfer learning with the ResNet50V2 architecture. The model is trained on a custom dataset with four classes, demonstrating high accuracy and robustness in image classification tasks.

## 📋 Table of Contents
- [🚀 Features](#-features)
- [🛠️ Prerequisites](#️-prerequisites)
- [⚙️ Installation](#️-installation)
- [📊 Dataset](#-dataset)
- [📝 Usage](#-usage)
- [🏗️ Model Architecture](#️-model-architecture)
- [🔄 Training Process](#-training-process)
- [📈 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Features

- Transfer learning using ResNet50V2 pre-trained on ImageNet
- Custom dataset handling and preprocessing
- Data augmentation for improved model generalization
- Fine-tuning of the pre-trained model
- Detailed visualization of training progress and results
- Confusion matrix and classification report generation

## 🛠️ Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- seaborn

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-classification-project.git
   cd image-classification-project
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The dataset used in this project is located in the `Dataset_3/Case3_th0.1_Qscaled` directory. It contains images divided into four classes:

1. Class A
2. Class B
3. Class C
4. Class D

The dataset is automatically split into training (80%) and validation (20%) sets during the execution of the script.

## 📝 Usage

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

## 🏗️ Model Architecture

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

## 🔄 Training Process

The training process consists of two phases:

1. Initial training (10 epochs)
   - Base model is frozen
   - Only top layers are trained

2. Fine-tuning (25 epochs)
   - Base model is partially unfrozen
   - Entire model is trained with a lower learning rate

Both phases implement early stopping and learning rate reduction to prevent overfitting.

## 📈 Results

The script generates and displays:

- Training and validation accuracy/loss plots
- Confusion matrix
- Detailed classification report

These results help in understanding the model's performance and identifying areas for improvement.

## 📁 Project Structure

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by [Your Name]

For any questions or suggestions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).
