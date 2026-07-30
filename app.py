import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Bike vs Scooty Classifier",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# --- Header Section ---
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Bike & Scooty Classification</h1>
    <p class="subtitle">State-of-the-art CNN Classification</p>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "bike_scooty_cnn.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.warning("Model not found! Please run train_cnn.py first to generate bike_scooty_cnn.h5.")
        return None

model = load_model()

# --- Main App Layout ---
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown('<h3>Upload or Capture an Image for Classification</h3>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload Image", "Camera"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

with tab2:
    camera_file = st.camera_input("Take a picture")

st.markdown('</div>', unsafe_allow_html=True)

image_to_process = uploaded_file if uploaded_file is not None else camera_file

if image_to_process is not None:
    # Read the image
    image = Image.open(image_to_process).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.markdown('<h4>Original Image</h4>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<h4>Classification Result</h4>', unsafe_allow_html=True)
        
        if model is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_array)[0][0]
                
                # If the dataset labels were swapped (e.g. 0 was Scooty and 1 was Bike in YOLO),
                # we swap them here to make the predictions accurate.
                if prediction < 0.5:
                    class_name = "Scooty"
                    confidence = 1.0 - prediction
                else:
                    class_name = "Bike"
                    confidence = prediction
                
                st.markdown(
                    f"""
                    <div class="prediction-item" style="text-align: center; display: block; padding: 2rem;">
                        <h2 style="color: white; margin-bottom: 0.5rem;">{class_name}</h2>
                        <h4 style="color: white;">Confidence: {confidence:.2%}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<p style='text-align:center;'>Cannot predict. Model is missing.</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by TensorFlow and Streamlit • Built for aesthetic classification</p>
</div>
""", unsafe_allow_html=True)
