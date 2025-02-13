import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image  # type: ignore
from layers import L1Dist

# Google Drive model download
model_url = "https://drive.google.com/uc?id=1ozMj1fsAk3wsNbJ1UvfhZS2_u9dsELcB"
model_path = "siamesemodelv2.h5"

# Download the model if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model... This may take a while ⏳"):
        gdown.download(model_url, model_path, quiet=False)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist': L1Dist})
    return model

model = load_model()

# Function to preprocess an image
def preprocess(img):
    img = img.resize((100, 100))  # Resize image to 100x100
    img = np.array(img) / 255.0   # Normalize
    return img

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

# Streamlit UI
st.title("🔍 Face Recognition using Siamese Network")

# Webcam capture button
if st.button("Capture Image"):
    captured_image = capture_image()
    if captured_image:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        os.makedirs("application_data/input_image", exist_ok=True)
        captured_image.save("application_data/input_image/input_image.jpg")
    else:
        st.warning("Failed to capture image!")

# Perform verification
if st.button("Verify Face"):
    input_image_path = "application_data/input_image/input_image.jpg"

    if not os.path.exists(input_image_path):
        st.error("No input image found. Please capture an image first.")
    else:
        input_img = preprocess(Image.open(input_image_path))

        # Compare with stored verification images
        verification_images_path = "application_data/verification_images"
        if not os.path.exists(verification_images_path) or not os.listdir(verification_images_path):
            st.error("No verification images found! Please add images to the verification folder.")
        else:
            results = []
            for img_name in os.listdir(verification_images_path):
                validation_img = preprocess(Image.open(os.path.join(verification_images_path, img_name)))

                # Make predictions
                result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
                results.append(result)

            # Calculate verification results
            detection_threshold = 0.99
            verification_threshold = 0.8
            detection = np.sum(np.array(results) > detection_threshold)
            verification = detection / len(results)
            verified = verification > verification_threshold

            # Display result
            if verified:
                st.success("✅ Face Verified!")
            else:
                st.error("❌ Face Not Recognized!")

            # Log details
            st.write(f"Detection Score: {detection}")
            st.write(f"Verification Score: {verification:.2f}")
