import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow_addons.metrics import F1Score 

custom_objects = {'F1Score': F1Score(num_classes=3)}
# Load the trained CNN model
model = tf.keras.models.load_model(
    r"C:\Users\srika\Dropbox\PC\Downloads\New folder (3)\val20_epochs45_testacc95.h5",
    custom_objects=custom_objects
)

# Define class labels
class_labels = ['Benign', 'Malignant', 'Normal']

# Set Streamlit page title and icon
st.set_page_config(page_title="Lung Cancer Classifier", page_icon="🫁")

# Customize Streamlit layout
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

# Add a sidebar for additional information (if needed)
st.sidebar.markdown("## Information")
st.sidebar.markdown("This is a medical app for classifying Lung Cancer.")

# Upload an image
uploaded_image = st.file_uploader("Upload a lung X-Ray image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Create a header for the uploaded image section
    st.markdown("## Uploaded Image")
    
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)

    # Convert the image to a NumPy array
    img = np.array(image)

    # Ensure the image has three channels (RGB)
    if img.ndim == 2 or img.shape[-1] != 3:
        # If the image is grayscale or has a different number of channels, convert it to RGB
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)

    # Resize the image to match the model's input size (176x208)
    img = tf.image.resize(img, (176, 208))

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # Make a prediction
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Display the prediction result
    st.markdown("## Prediction Result")
    st.markdown(f"**Predicted Class:** {class_labels[predicted_class]}")
    st.markdown(f"**Confidence:** {prediction[0][predicted_class]:.2f}")

# Add a footer and additional information (if needed)
st.markdown("---")
st.markdown("This app is for educational purposes and should not be used for medical diagnosis.")
