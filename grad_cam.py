import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Configuration
MODEL_PATH = "stage2_best_model.h5"
IMG_SIZE = (224, 224)

LAST_CONV_LAYER_NAME = 'conv5_block3_out' 

# Class names must match the order of your stage 2 generator/folders
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

# 1. Grad-CAM Generation Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes and returns a Grad-CAM heatmap.
    """
    # 1. Create a model that maps the input image to the activations of the 
    # last convolutional layer AND the final output predictions.
    grad_model = Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Record operations for automatic differentiation (gradient calculation)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # If no specific index is provided, use the class with the highest prediction
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # Get the score for the predicted class
        class_channel = preds[:, pred_index]

    # 3. Compute the gradient of the predicted class score with respect to 
    # the output of the last convolutional layer.
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Compute the mean intensity of the gradient over all spatial dimensions (width/height).
    # This gives the "importance weight" for each feature map channel.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply the feature map by the importance weights
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU to discard negative contributions (focus only on positive evidence)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

# 2. Image Processing and Visualization Function
def display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image and displays it.
    """
    # Load the original image using OpenCV
    # We use cv2.IMREAD_COLOR to ensure the image is loaded with 3 color channels
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    
    # Resize the image to the model input size (224x224)
    img = cv2.resize(img, IMG_SIZE)
    
    # Check for successful image load 
    if img is None:
        raise FileNotFoundError(f"OpenCV could not load image at: {img_path}")
        
    # Heatmap Processing 
    # Rescale heatmap to 0-255 and convert to 8-bit unsigned integer (required by cv2 functions)
    heatmap = np.uint8(255 * heatmap)
    
    # Resize the heatmap to the size of the image (224x224)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    
    # Apply a jet color map for visualization. This creates a 3-channel BGR image.
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the original image. 
    # Both 'img' and 'heatmap' are now 3-channel (BGR) arrays of size (224, 224, 3), 
    # resolving the 'Sizes of input arguments do not match' error.
    superimposed_img = cv2.addWeighted(img, 1.0 - alpha, heatmap, alpha, 0)
    
    # Convert back to RGB for Matplotlib display (OpenCV loads images as BGR)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB) 
    
    return superimposed_img

# 3. Main Execution Block for Testing
if __name__ == "__main__":
    
    # Load Model 
    try:
        model = load_model(MODEL_PATH)
        print("Stage 2 model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Select Image for Testing
        TEST_IMAGE_PATH = 'stage2_yes/test/glioma_tumor/G_23.jpg' 
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\nERROR: Test image not found at {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH to a valid file path in your test set.")
    else:
        # Preprocessing 
        original_img = image.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
        img_array = image.img_to_array(original_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Run Prediction
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds[0])
        predicted_label = CLASS_NAMES[predicted_class_index]
        confidence = preds[0][predicted_class_index]

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            img_array, model, LAST_CONV_LAYER_NAME, pred_index=predicted_class_index
        )
        
        # Visualize 
        gradcam_image = display_gradcam(TEST_IMAGE_PATH, heatmap)
        
        plt.figure(figsize=(10, 5))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Original MRI\n(Actual: Unknown)")
        plt.axis('off')

        # Grad-CAM Heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_image)
        plt.title(f"Grad-CAM Result\nPredicted: {predicted_label} ({confidence*100:.2f}%)")
        plt.axis('off')
        plt.suptitle("Model Interpretability: Grad-CAM Visualization")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        