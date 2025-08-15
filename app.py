import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, base64, cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/facial_emotion_detection_model.h5")

def ensure_model():
    # If you want to download from cloud at boot, put that logic here
    # e.g., from an S3/Drive URL in env MODEL_URL
    # Example:
    # import requests
    # url = os.environ.get("MODEL_URL")
    # if url and not os.path.exists(MODEL_PATH):
    #     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    #     with requests.get(url, stream=True) as r:
    #         r.raise_for_status()
    #         with open(MODEL_PATH, "wb") as f:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 f.write(chunk)
    pass

ensure_model()
model = load_model(MODEL_PATH)
class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

def preprocess_image(img_b64):
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",", 1)[1]
    img_data = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode base64 image")
    img = cv2.resize(img, (48, 48)) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,48,48,1)
    return img

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400
        arr = preprocess_image(img_b64)
        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        return jsonify({
            "emotion": class_names[idx],
            "confidence": round(float(pred[0][idx])*100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
