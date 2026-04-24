from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import json

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/bird_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

# Load model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names, flush=True)


@app.route("/")
def home():
    return "Bird Identification API is Running 🚀"


def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = ImageOps.exif_transpose(img)  # fix rotated phone images
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img_array = preprocess_image(file)

        prediction = model.predict(img_array, verbose=0)[0]

        index = int(np.argmax(prediction))
        result = class_names[index]
        confidence = float(np.max(prediction)) * 100

        print("Raw prediction:", prediction, flush=True)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)