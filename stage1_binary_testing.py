import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load trained Stage 1 model
model = load_model('stage1_best_model.h5')

# Paths to test data
test_dir = "stage1_binary/test"  # contains 'no' and 'yes' folders
classes = ['no', 'yes']

# Collect image paths and labels
image_paths = []
labels = []

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(test_dir, cls)
    for img_file in os.listdir(cls_folder):
        img_path = os.path.join(cls_folder, img_file)
        image_paths.append(img_path)
        labels.append(idx)  # 0 = no tumor, 1 = yes tumor

# Predict function
predictions = []

for img_path in image_paths:
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]
    pred_class = 1 if pred > 0.5 else 0
    predictions.append(pred_class)

    print(f"{os.path.basename(img_path)} | Actual: {classes[labels[image_paths.index(img_path)]]} | Predicted: {classes[pred_class]} | Confidence: {pred:.2f}")

# Evaluation metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)

print("\n--- Stage 1 Evaluation ---")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
