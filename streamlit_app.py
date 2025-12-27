import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="centered"
)

st.title("🌾 Crop Disease Detection")
st.write("Detect crop diseases using image upload or live camera")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -------------------------------
# Load class names
# -------------------------------
with open("class_names.json") as f:
    class_dict = json.load(f)

index_to_class = {v: k for k, v in class_dict.items()}

def format_name(name):
    return name.replace("___", " - ").replace("_", " ")

# -------------------------------
# Select mode
# -------------------------------
option = st.radio(
    "Choose input method:",
    ("Upload Image", "Live Camera")
)

# -------------------------------
# IMAGE UPLOAD OPTION
# -------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        img = image.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        best_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        disease_name = format_name(index_to_class[best_index])

        st.markdown("---")
        st.subheader("🧪 Prediction Result")

        if "healthy" in disease_name.lower():
            st.success(f"🌱 Leaf Status: **Healthy**")
        else:
            st.error(f"🦠 Disease Detected: **{disease_name}**")

        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

# -------------------------------
# LIVE CAMERA OPTION
# -------------------------------
if option == "Live Camera":
    camera_image = st.camera_input("📸 Capture Leaf Image")

    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Live Camera Image", use_column_width=True)

        img = image.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        best_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        disease_name = format_name(index_to_class[best_index])

        st.markdown("---")
        st.subheader("🧪 Live Prediction Result")

        if "healthy" in disease_name.lower():
            st.success(f"🌱 Leaf Status: **Healthy**")
        else:
            st.error(f"🦠 Disease Detected: **{disease_name}**")

        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))
