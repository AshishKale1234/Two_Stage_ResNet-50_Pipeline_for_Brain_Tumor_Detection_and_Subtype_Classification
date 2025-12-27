import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ------------------------------
# Configuration & Paths
# ------------------------------
STAGE1_MODEL_PATH = 'stage1_best_model.h5'
STAGE2_MODEL_PATH = 'stage2_best_model.h5'
IMG_SIZE = (224, 224)
BINARY_THRESHOLD = 0.5 # Threshold for Stage 1 (0.5 is default)

# Class names must match the order of your stage 2 generator/folders
# Assuming your folders are ordered: ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']
STAGE2_CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

# ------------------------------
# Model Loading
# ------------------------------
try:
    stage1_model = load_model(STAGE1_MODEL_PATH)
    stage2_model = load_model(STAGE2_MODEL_PATH)
    print("Both Stage 1 and Stage 2 models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# ------------------------------
# Utility Function for Preprocessing
# ------------------------------
def preprocess_image(img_path):
    """Loads and preprocesses a single image for the ResNet50 model."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")
        
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ------------------------------
# Core Prediction Pipeline
# ------------------------------
def predict_mri(img_path):
    """Runs the two-stage diagnostic pipeline on a single MRI image."""
    
    try:
        preprocessed_img = preprocess_image(img_path)
    except FileNotFoundError as e:
        return str(e)

    # --- Stage 1: Binary Detection (Tumor vs. No Tumor) ---
    stage1_pred = stage1_model.predict(preprocessed_img, verbose=0)[0][0]
    
    if stage1_pred > BINARY_THRESHOLD:
        # Tumor Detected. Proceed to Stage 2.
        
        # --- Stage 2: Multi-Class Classification ---
        stage2_preds = stage2_model.predict(preprocessed_img, verbose=0)[0]
        
        # Find the highest probability class
        tumor_type_index = np.argmax(stage2_preds)
        tumor_type = STAGE2_CLASS_NAMES[tumor_type_index]
        confidence = stage2_preds[tumor_type_index]
        
        return {
            "result": "Tumor Detected",
            "type": tumor_type,
            "confidence": f"{confidence * 100:.2f}%",
            "stage1_confidence": f"{stage1_pred * 100:.2f}%"
        }
        
    else:
        # No Tumor Detected.
        confidence_no_tumor = 1.0 - stage1_pred
        return {
            "result": "No Tumor Detected",
            "confidence": f"{confidence_no_tumor * 100:.2f}%",
            "stage1_confidence": f"{stage1_pred * 100:.2f}%"
        }

# ------------------------------
# Example Usage (You must change these paths to test files)
# ------------------------------
if __name__ == "__main__":
    # --- Example 1: Use a known tumor image (e.g., a Glioma) ---
    # NOTE: Replace with a real path from your 'stage2_yes/test/glioma_tumor' folder
    tumor_example_path = 'stage2_yes/test/glioma_tumor/G_001.jpg' 
    
    # --- Example 2: Use a known non-tumor image ---
    # NOTE: Replace with a real path from your 'stage1_binary/test/no' folder
    no_tumor_example_path = 'stage1_binary/test/no/N_001.jpg'
    
    print("\n--- Running Example 1 (Tumor Case) ---")
    result_tumor = predict_mri(tumor_example_path)
    print(result_tumor)
    
    print("\n--- Running Example 2 (No Tumor Case) ---")
    result_no_tumor = predict_mri(no_tumor_example_path)
    print(result_no_tumor)