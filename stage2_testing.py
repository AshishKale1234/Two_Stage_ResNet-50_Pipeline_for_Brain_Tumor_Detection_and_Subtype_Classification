import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score

# Configuration (Must match training setup)
MODEL_PATH = 'stage2_best_model.h5' # Ensure this file exists after training
TEST_DIR = "stage2_yes/test"       # Path to your test data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Setup Test Data Generator

# Use the same preprocessing function as during training
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create the generator. IMPORTANT: shuffle=False to map predictions to true labels
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get the class names in the order used by the generator (essential for the plot labels)
class_labels = list(test_generator.class_indices.keys())
print(f"Loaded {test_generator.samples} test images across {len(class_labels)} classes: {class_labels}")

# 2. Load Model and Evaluate Performance
try:
    model = load_model(MODEL_PATH)
except:
    print(f"Error: Could not load model from {MODEL_PATH}. Check file path.")
    exit()

# Evaluate standard test accuracy/loss (for reference)
print("\n--- Running Standard Model Evaluation ---")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 3. Generate Predictions and True Labels

# Reset generator to ensure predictions and labels align (essential step)
test_generator.reset()

# Predict on the test set
# steps ensures we predict all images, including the last partial batch
steps = test_generator.samples // test_generator.batch_size + 1 
predictions = model.predict(test_generator, steps=steps)

# Convert probabilities to class indices (0, 1, 2, 3...)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels from the generator (these are the true class indices)
true_classes = test_generator.classes

# Ensure true_classes length matches predictions length (can sometimes differ due to batching)
# We truncate predictions and true_classes to the smallest common length
min_len = min(len(true_classes), len(predicted_classes))
true_classes = true_classes[:min_len]
predicted_classes = predicted_classes[:min_len]

# 4. Calculate and Plot Confusion Matrix
print("\n--- Generating Confusion Matrix ---")

# Calculate the Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the Confusion Matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,              # Annotate cells with numerical values
    fmt='d',                 # Format as integers
    cmap='Blues',            # Color map
    xticklabels=class_labels, # Predicted labels
    yticklabels=class_labels  # Actual labels
)
plt.title('Stage 2: Confusion Matrix for Tumor Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Optional: Print classification metrics for more detail
from sklearn.metrics import classification_report
print("\n--- Classification Report (Detailed Metrics per Class) ---")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))