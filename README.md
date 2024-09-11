This project implements a state-of-the-art image classification model using transfer learning with the ResNet50V2 architecture. The model is trained on a custom dataset with four classes, demonstrating high accuracy and robustness in image classification tasks.
📋 Table of Contents

🚀 Features
🛠️ Prerequisites
⚙️ Installation
📊 Dataset
📝 Usage
🏗️ Model Architecture
🔄 Training Process
📈 Results
📁 Project Structure
🤝 Contributing
📄 License

🚀 Features

Transfer learning using ResNet50V2 pre-trained on ImageNet
Custom dataset handling and preprocessing
Data augmentation for improved model generalization
Fine-tuning of the pre-trained model
Detailed visualization of training progress and results
Confusion matrix and classification report generation

🛠️ Prerequisites

Python 3.7+
TensorFlow 2.x
NumPy
Matplotlib
scikit-learn
seaborn

⚙️ Installation

Clone this repository:
bashCopygit clone https://github.com/yourusername/image-classification-project.git
cd image-classification-project

Create a virtual environment (optional but recommended):
bashCopypython -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:
bashCopypip install -r requirements.txt


📊 Dataset
The dataset used in this project is located in the Dataset_3/Case3_th0.1_Qscaled directory. It contains images divided into four classes:

Class A
Class B
Class C
Class D

The dataset is automatically split into training (80%) and validation (20%) sets during the execution of the script.
📝 Usage
To run the image classification script:
bashCopypython image_classification.py
This will:

Load and preprocess the data
Create and train the model
Fine-tune the model
Evaluate the model and display results

You can modify the hyperparameters and model architecture in the image_classification.py file to experiment with different configurations.
🏗️ Model Architecture
The model uses transfer learning with the ResNet50V2 architecture:

Data Augmentation Layer

Random horizontal flip
Random rotation (±10%)
Random zoom (±10%)


Pre-trained ResNet50V2 base (weights from ImageNet)
Global Average Pooling
Dense layer (256 units, ReLU activation)
Dropout (0.5)
Output Dense layer (4 units, softmax activation)

🔄 Training Process
The training process consists of two phases:

Initial training (10 epochs)

Base model is frozen
Only top layers are trained


Fine-tuning (25 epochs)

Base model is partially unfrozen
Entire model is trained with a lower learning rate



Both phases implement early stopping and learning rate reduction to prevent overfitting.
📈 Results
The script generates and displays:

Training and validation accuracy/loss plots
Confusion matrix
Detailed classification report

These results help in understanding the model's performance and identifying areas for improvement.
📁 Project Structure
Copyimage-classification-project/
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
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
