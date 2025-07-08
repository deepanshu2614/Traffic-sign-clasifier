import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from gtsrb_labels import gtsrb_labels


# Load trained model
model = load_model("best_model2.keras",compile=False)

# Title
st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a traffic sign to predict its class")

# Upload image
uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed)
    class_id = np.argmax(prediction)
    label = gtsrb_labels[class_id]
    
    # Display result
    st.markdown(f"### 🧠 Predicted Class: `{class_id}`")
    st.success(f"**Label:** {label}")
