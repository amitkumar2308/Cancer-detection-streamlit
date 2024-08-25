import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
def load_model_on_start():
    try:
        model = load_model('inception_chest.h5')
        st.success("Model loaded successfully!")
        return model
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make a prediction
def predict_image(model, img):
    classes = ["Adenocarcinoma Chest Lung Cancer", 
               "Large cell carcinoma Lung Cancer", 
               "No Lung Cancer / NORMAL", 
               "Squamous cell carcinoma Lung Cancer"]
    
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    prediction = model.predict(x)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence in percentage

    return predicted_class, confidence

# Main application
def main():
    st.title("🩺 Chest Cancer Detection")
    st.write("Upload a chest X-ray image to get a prediction on the type of lung cancer.")

    # Load the model
    model = load_model_on_start()

    if model:
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Convert the image to RGB to ensure it has 3 channels
            img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Predict'):
                with st.spinner('Analyzing the image...'):
                    result, confidence = predict_image(model, img)
                st.success(f"Prediction: {result}")
                st.write(f"### Confidence Level: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
