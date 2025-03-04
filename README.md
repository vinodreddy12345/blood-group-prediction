# Blood Group Prediction App

This Streamlit application predicts blood groups from fingerprint images using a deep learning model.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model file `bloodgroup_model.h5` is in the same directory as `app.py`

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Features
- Upload fingerprint images in various formats (JPG, PNG, BMP)
- Real-time blood group prediction
- Confidence score for predictions
- User-friendly interface
- Support for different image formats and automatic preprocessing

## Model Information
The application uses a deep learning model trained to recognize blood groups from fingerprint patterns. The model expects input images of size 128x128 pixels and supports 8 blood group classifications (A+, A-, B+, B-, AB+, AB-, O+, O-).

## Important Note
This tool is for research and demonstration purposes only. Always confirm blood group through standard medical tests.
