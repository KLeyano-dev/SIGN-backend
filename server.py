from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2, base64, numpy as np, tensorflow as tf
from collections import deque, Counter
import json
import tempfile, os

app = Flask(__name__)

# Allow HTTPS requests from frontend
CORS(app, supports_credentials=True)

SEQ_LEN = 30
sequence = deque(maxlen=SEQ_LEN)

print("\n🔵 Server starting...")

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="sign_lstm_full_float32_select.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("label_map.json", "r") as f:
    index_to_label = {int(k): v for k, v in json.load(f).items()}

def predict_tflite(seq):
    interpreter.set_tensor(input_details[0]["index"], np.expand_dims(seq, 0).astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.json
    landmarks = data.get("landmarks")
    if not landmarks:
        return jsonify({"label": "No landmarks", "confidence": 0})

    try:
        # Convert landmarks to numpy array
        landmarks = np.array(landmarks, dtype=np.float32)
    except:
        return jsonify({"label": "Invalid landmarks", "confidence": 0})

    sequence.append(landmarks)

    if len(sequence) == SEQ_LEN:
        preds = predict_tflite(np.array(sequence))
        idx = int(np.argmax(preds))
        return jsonify({"label": index_to_label[idx], "confidence": float(preds[idx])})

    return jsonify({"label": "Collecting frames...", "confidence": 0})

@app.route("/test")
def test():
    return {"status": "backend reachable"}

if __name__ == "__main__":
    print("🔵 HTTPS enabled on port 5005")
    app.run(
        host="0.0.0.0",
        port=5005,
        ssl_context=("certs/cert.pem", "certs/key.pem")  # MUST BE REAL FILES
    )
