import os
import shutil
import random

# Configuration
source_dir = r"archive/2 classes/yes"   # Only the 'yes' folder
output_dir = "stage2_yes"               # Stage 2 output
split_ratio = (0.8, 0.1, 0.1)           # train, validation, test

# Function to split dataset
def split_data(source_folder, output_base, split_ratio=(0.8,0.1,0.1)):
    # List all subfolders (classes)
    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_folder, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)
        n_total = len(images)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        # Create train/val/test folders for each class
        for folder in ["train", "validation", "test"]:
            os.makedirs(os.path.join(output_base, folder, cls), exist_ok=True)

        # Copy images to the respective folders
        for i, img in enumerate(images):
            src = os.path.join(cls_path, img)
            if i < n_train:
                dst = os.path.join(output_base, "train", cls, img)
            elif i < n_train + n_val:
                dst = os.path.join(output_base, "validation", cls, img)
            else:
                dst = os.path.join(output_base, "test", cls, img)
            shutil.copy(src, dst)

# Run the split
if __name__ == "__main__":
    split_data(source_dir, output_dir, split_ratio)
    print("Stage 2 dataset creation complete!")
