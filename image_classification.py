import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset path
data_dir = 'Dataset_3/Case3_th0.1_Qscaled'
img_height, img_width = 224, 224  # Standard input size for many pre-trained models
batch_size = 32
num_classes = 4  # As specified, there are 4 classes

def load_and_preprocess_data(data_dir):
    image_files = []
    labels = []
    class_names = os.listdir(data_dir)
    
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for image_file in os.listdir(class_dir):
            image_files.append(os.path.join(class_dir, image_file))
            labels.append(class_index)
    
    return np.array(image_files), np.array(labels), class_names

image_files, labels, class_names = load_and_preprocess_data(data_dir)
print(f"Total images: {len(image_files)}")
print(f"Classes: {class_names}")

# Split the data
train_files, val_files, train_labels, val_labels = train_test_split(
    image_files, labels, test_size=0.2, stratify=labels, random_state=42
)

def preprocess_image(image_file):
    img = tf.io.read_file(image_file)
    img = tf.image.decode_png(img, channels=3)  # Assuming PNG format, change if different
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img

train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_dataset = train_dataset.map(lambda x, y: (preprocess_image(x), y))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_dataset = val_dataset.map(lambda x, y: (preprocess_image(x), y))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Load pre-trained ResNet50V2 model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
base_model.trainable = False

# Create the model
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),  # Add this line to define input shape
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
    ]
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_dataset)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)
plot_history(history_fine)

# Save the model
model.save('case_3')

# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Get predictions
val_predictions = model.predict(val_dataset)
val_pred_classes = np.argmax(val_predictions, axis=1)
val_true_classes = np.concatenate([y for x, y in val_dataset], axis=0)

# Plot confusion matrix
plot_confusion_matrix(val_true_classes, val_pred_classes, class_names)

# Print classification report
print(classification_report(val_true_classes, val_pred_classes, target_names=class_names))
