import os, base64, numpy as np, cv2, threading
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- Config ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_drq.tflite")
CLASS_NAMES = ['angry','disgust','fear','happy','neutral','sad','surprise']

# ---------- App ----------
app = Flask(__name__)
CORS(app)

# ---------- TFLite load ----------
# Prefer tflite-runtime; fall back to TF if needed.
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback

_interpreter = None
_input_details = None
_output_details = None
_infer_lock = threading.Lock()

def load_interpreter():
    global _interpreter, _input_details, _output_details
    _interpreter = Interpreter(model_path=MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()

def warmup():
    # Warm the model once to avoid first-request lag
    dummy = np.zeros((1,48,48,1), dtype=np.float32)
    with _infer_lock:
        _interpreter.set_tensor(_input_details[0]['index'], dummy)
        _interpreter.invoke()
        _interpreter.get_tensor(_output_details[0]['index'])

load_interpreter()
warmup()

# ---------- Utils ----------
def preprocess_image(img_b64: str) -> np.ndarray:
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",", 1)[1]
    img_data = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode base64 image.")
    # Resize & normalize
    img = cv2.resize(img, (48, 48)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,48,48,1)
    return img

# ---------- Routes ----------
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

        with _infer_lock:
            _interpreter.set_tensor(_input_details[0]['index'], arr)
            _interpreter.invoke()
            probs = _interpreter.get_tensor(_output_details[0]['index'])

        idx = int(np.argmax(probs[0]))
        conf = float(probs[0][idx]) * 100.0
        return jsonify({"emotion": CLASS_NAMES[idx], "confidence": round(conf, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
