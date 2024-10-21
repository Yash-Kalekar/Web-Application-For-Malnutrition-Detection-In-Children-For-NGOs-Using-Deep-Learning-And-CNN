import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Title and description for the app
st.title("Image Classification with Keras Model")
st.write("Upload an image to classify it using a pre-trained Keras model.")

# Load the Keras model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# File uploader to allow image uploads
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image and convert it to RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Define a constant size for the image (224x224)
    size = (224, 224)
    
    # Resize the uploaded image to the constant size
    resized_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Display the resized image in the app
    st.image(resized_image, caption="Resized Image (80x80)", use_column_width=True)

    # Convert the resized image to an array
    image_array = np.asarray(resized_image)

    # Normalize the image array to values between -1 and 1
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create a batch of one (1, 224, 224, 3) and add the normalized image to the batch
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the prediction and confidence score
    st.write(f"**Prediction:** {class_name[2:].strip()}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
