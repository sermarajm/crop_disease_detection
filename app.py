from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("model.h5")

# -------------------------------
# Load class names
# -------------------------------
if not os.path.exists("class_names.json"):
    raise FileNotFoundError("class_names.json not found. Run model.py first.")

with open("class_names.json") as f:
    class_dict = json.load(f)

# Reverse mapping: index -> disease name
index_to_class = {v: k for k, v in class_dict.items()}

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return "🌾 Crop Disease Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    img = preprocess_image(image)
    prediction = model.predict(img)[0]

    best_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    disease_name = index_to_class[best_index]
    disease_name = disease_name.replace("___", " - ").replace("_", " ")

    return jsonify({
        "prediction": disease_name,
        "confidence": round(confidence, 2)
    })

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
