import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Blood Group Prediction",
    page_icon="🩸",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #cccccc;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_model():
    try:
        model = load_model("bloodgroup_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    try:
        # Convert PIL Image to numpy array
        img = np.array(image)
        
        # Check if image is grayscale and convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        # Resize and normalize
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_blood_group(model, image):
    try:
        # Class labels
        class_labels = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        
        # Preprocess image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None
            
        # Make prediction
        prediction = model.predict(processed_img)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100
        
        return class_labels[class_index], confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def main():
    # Load model
    model = load_prediction_model()
    if model is None:
        st.error("Could not load the model. Please ensure the model file exists.")
        return

    # Header
    st.title("Blood Group Prediction from Fingerprint 🩸")
    st.write("""
    This application predicts blood groups from fingerprint images using deep learning.
    Upload a clear fingerprint image for the best results.
    """)
    
    # File uploader
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png", "bmp"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fingerprint", use_column_width=True)
            
            # Prediction button
            if st.button("Predict Blood Group", key="predict_btn"):
                with st.spinner("Analyzing fingerprint..."):
                    result = predict_blood_group(model, image)
                    
                    if result:
                        blood_group, confidence = result
                        
                        # Display results
                        st.success(f"Predicted Blood Group: {blood_group}")
                        st.info(f"Confidence: {confidence:.2f}%")
                        
                        # Additional information
                        st.markdown("---")
                        st.markdown("""
                        **Note:**
                        - This prediction is based on machine learning analysis
                        - For medical purposes, please confirm with standard blood tests
                        - Higher confidence scores indicate more reliable predictions
                        """)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
