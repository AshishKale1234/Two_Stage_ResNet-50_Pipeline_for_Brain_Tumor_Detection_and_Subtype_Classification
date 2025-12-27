import streamlit as st
import os
import tempfile
import time
import numpy as np

# TF / Keras imports for Grad-CAM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Grad-CAM helpers (from your grad_cam.py)
from grad_cam import make_gradcam_heatmap, display_gradcam

# Import the core prediction function from the full_pipeline script
try:
    from full_pipeline import predict_mri
except ImportError:
    st.error("Error: Could not import predict_mri from full_pipeline.py. Make sure the file exists.")
    st.stop()

# --------------------------
# Grad-CAM configuration
# --------------------------
STAGE2_MODEL_PATH = "stage2_best_model.h5"
LAST_CONV_LAYER_NAME = "conv5_block3_out"   # final conv layer of ResNet50 base
IMG_SIZE = (224, 224)

@st.cache_resource
def load_stage2_model():
    """Load Stage 2 model once and cache it for reuse."""
    return load_model(STAGE2_MODEL_PATH)

stage2_model = load_stage2_model()

# --------------------------
# Streamlit App Configuration
# --------------------------
st.set_page_config(
    page_title="Automated Brain Tumor Diagnosis",
    layout="wide",
    initial_sidebar_state="auto"
)

# --------------------------
# Helper: Display results
# --------------------------
def display_results(result_dict):
    """Displays the diagnosis based on the result dictionary."""
    st.subheader("Final Diagnostic Result")
    
    col1, col2 = st.columns(2)

    # 1. Tumor Detected Case
    if result_dict['result'] == "Tumor Detected":
        tumor_type = result_dict['type'].replace('_', ' ').title()
        
        col1.error(f"**DIAGNOSIS:** {result_dict['result']}")
        col2.metric(label="Predicted Tumor Type", value=tumor_type)

        st.markdown("---")
        st.info(f"**Classification Confidence (Stage 2):** {result_dict['confidence']}")
        st.caption(f"Initial Binary Detection (Stage 1) Confidence: {result_dict['stage1_confidence']}")

    # 2. No Tumor Detected Case
    elif result_dict['result'] == "No Tumor Detected":
        col1.success(f"**DIAGNOSIS:** {result_dict['result']}")
        col2.metric(label="No Tumor Confidence", value=result_dict['confidence'])
        
        st.markdown("---")
        st.caption("The system stopped at Stage 1 (Binary Detection) as no tumor was confirmed.")
        
    # 3. Error Case
    else:
        st.exception(result_dict)

# --------------------------
# Main Application Logic
# --------------------------

st.title("🧠 Two-Stage MRI Tumor Classification")
st.markdown("""
    Upload an MRI scan to receive an immediate diagnosis using the integrated ResNet50-based two-stage pipeline.
    The system first detects the presence of a tumor (Stage 1) and then classifies its type (Stage 2).
""")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "1. Choose an MRI Image (.jpg, .jpeg, .png)", 
    type=["jpg", "jpeg", "png"]
)

# --- Processing Logic ---
if uploaded_file is not None:
    # Save uploaded file to a temporary location (predict_mri and Grad-CAM both need a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    # Three columns: left = uploaded image, middle = Grad-CAM (placeholder), right = diagnosis / button
    col_img, col_cam, col_diag = st.columns([2, 2, 3])

    # LEFT: Uploaded image (top-aligned)
    with col_img:
        st.image(
            image_path,
            caption=f"Uploaded Image: {uploaded_file.name}",
            width=500
        )

    # MIDDLE: Grad-CAM placeholders (so layout stays aligned)
    gradcam_title_ph = col_cam.empty()
    gradcam_caption_ph = col_cam.empty()
    gradcam_image_ph = col_cam.empty()

    # RIGHT: Button + diagnosis (top-aligned: no extra vertical spacer)
    result_dict = None

    with col_diag:
        # Diagnosis button right at the top of column
        run_clicked = st.button("2. Run Diagnosis", type="primary", use_container_width=True)

        if run_clicked:
            with st.spinner('Analyzing MRI scan using two-stage pipeline...'):
                time.sleep(1)  # visual delay

                # Two-stage prediction (Stage 1 + Stage 2)
                result_dict = predict_mri(image_path)

                # If tumor is detected, generate Grad-CAM using Stage 2 model
                gradcam_image = None
                if result_dict.get("result") == "Tumor Detected":
                    # Preprocess the same uploaded image
                    original_img = image.load_img(image_path, target_size=IMG_SIZE)
                    img_array = image.img_to_array(original_img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Stage 2 predictions
                    preds = stage2_model.predict(img_array)
                    predicted_class_index = int(np.argmax(preds[0]))

                    # Grad-CAM heatmap
                    heatmap = make_gradcam_heatmap(
                        img_array,
                        stage2_model,
                        LAST_CONV_LAYER_NAME,
                        pred_index=predicted_class_index
                    )

                    # Overlay heatmap on original
                    gradcam_image = display_gradcam(image_path, heatmap)

            st.success(" Analysis Complete!")

            # Show diagnosis on the RIGHT
            if result_dict is not None:
                display_results(result_dict)

            # Fill the MIDDLE column with Grad-CAM (only for tumor cases)
            if result_dict is not None and result_dict.get("result") == "Tumor Detected" and gradcam_image is not None:
                gradcam_title_ph.subheader("Grad-CAM Visualization (Stage 2 Focus)")
                gradcam_caption_ph.caption(
                    "Regions in the MRI that were most influential for the tumor-type prediction."
                )
                gradcam_image_ph.image(gradcam_image, width=500)

    # Clean-up helper 
    @st.cache_resource
    def cleanup_temp_file():
        if os.path.exists(image_path):
            os.unlink(image_path)
