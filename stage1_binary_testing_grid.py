import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load trained Stage 1 model
model = load_model('stage1_best_model.h5')

# Paths to test data
test_dir = "stage1_binary/test"
classes = ['no', 'yes']

# Choose number of images to display
NUM_IMAGES = 12  # total images to show in grid

# Collect all image paths with labels
all_images = []
for cls in classes:
    cls_folder = os.path.join(test_dir, cls)
    for img_file in os.listdir(cls_folder):
        all_images.append((os.path.join(cls_folder, img_file), cls))

# Randomly select NUM_IMAGES
images_to_show = random.sample(all_images, NUM_IMAGES)

# Plot images with actual/predicted labels
plt.figure(figsize=(20, 8))

for i, (img_path, actual_label) in enumerate(images_to_show):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict using Stage 1 model
    pred_prob = model.predict(img_array)[0][0]
    pred_label = 'yes' if pred_prob > 0.5 else 'no'
    
    # Display image
    plt.subplot(3, 4, i+1)  # 3 rows, 4 columns for 12 images
    plt.imshow((img_array[0] + 1)/2)  # undo preprocess_input scaling for display
    plt.title(f"Actual: {actual_label}\nPredicted: {pred_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()
