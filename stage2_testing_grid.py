import os
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Paths and model
MODEL_PATH = "stage2_best_model.h5"
TEST_DIR = "stage2_yes/test"
IMG_SIZE = (224, 224)

# Load model
model = load_model(MODEL_PATH)

# Get class names from folder names
class_names = sorted(os.listdir(TEST_DIR))
print("Classes:", class_names)

# Collect all test images
test_images = []
for cls in class_names:
    cls_path = os.path.join(TEST_DIR, cls)
    imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    test_images.extend([(img_path, cls) for img_path in imgs])

# Randomly select 12 images
selected_images = random.sample(test_images, 12)

# Plotting grid with color-coded titles
plt.figure(figsize=(15, 10))

for i, (img_path, true_label) in enumerate(selected_images):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prediction
    preds = model.predict(x)
    pred_label = class_names[np.argmax(preds)]

    # Color: green if correct, red if wrong
    color = 'green' if pred_label == true_label else 'red'

    # Plot
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Actual: {true_label}\nPred: {pred_label}", fontsize=10, color=color)

plt.tight_layout()
plt.show()
