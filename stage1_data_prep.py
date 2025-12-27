import os
import shutil
import random
from pathlib import Path

# Configuration
source_dir = "archive/2 classes"  # your original dataset
output_dir = "stage1_binary"      # folder where train/val/test will be created
split_ratio = (0.8, 0.1, 0.1)    # train, val, test

# Helper function to flatten 'yes' folder
def gather_yes_images(yes_folder):
    """
    Move all images from 4 classes subfolders into parent yes folder
    """
    subfolders = [f for f in os.listdir(yes_folder) if os.path.isdir(os.path.join(yes_folder, f))]
    for sub in subfolders:
        sub_path = os.path.join(yes_folder, sub)
        for file in os.listdir(sub_path):
            src = os.path.join(sub_path, file)
            dst = os.path.join(yes_folder, file)
            shutil.copy(src, dst)  # copy instead of move to preserve original
        # Optional: remove empty folder
        # shutil.rmtree(sub_path)

# Flatten yes folder
gather_yes_images(os.path.join(source_dir, "yes"))

# Function to split dataset
def split_data(source_folder, output_base, split_ratio=(0.8,0.1,0.1)):
    classes = os.listdir(source_folder)
    for cls in classes:
        cls_path = os.path.join(source_folder, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(split_ratio[0]*n_total)
        n_val = int(split_ratio[1]*n_total)

        # Create train/val/test folders
        for folder in ["train", "validation", "test"]:
            os.makedirs(os.path.join(output_base, folder, cls), exist_ok=True)

        # Copy images to respective folders
        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            if i < n_train:
                dst = os.path.join(output_base, "train", cls, img)
            elif i < n_train + n_val:
                dst = os.path.join(output_base, "validation", cls, img)
            else:
                dst = os.path.join(output_base, "test", cls, img)
            shutil.copy(src, dst)

# Split the data
split_data(source_dir, output_dir, split_ratio)
print("Data preprocessing and splitting complete!")
