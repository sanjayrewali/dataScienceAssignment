"""
app.py
------
Flask REST API for the Telco Customer Churn model.

Endpoint
--------
POST /predict
    Accepts a single customer's attributes as JSON, applies the exact same
    preprocessing/feature-engineering used in training (via the persisted
    pipeline), and returns the churn prediction and probability.

Example
-------
    curl -X POST http://127.0.0.1:5000/predict \
         -H "Content-Type: application/json" \
         -d @sample_request.json

Example response
----------------
    { "prediction": "Yes", "churn_probability": 0.82 }
"""

import os

import joblib
from flask import Flask, jsonify, request

import churn_utils as cu

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")

app = Flask(__name__)

# Load the artifact once at startup.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Run notebook/churn_analysis.ipynb to train and save the model first."
    )

_artifact = joblib.load(MODEL_PATH)
_pipeline = _artifact["pipeline"]
_threshold = _artifact.get("threshold", 0.5)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Telco Customer Churn Prediction API",
            "endpoints": {"POST /predict": "Predict churn for a single customer"},
            "model_metrics": _artifact.get("metrics", {}),
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    # 1. Parse JSON safely.
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # 2. Validate input and build a single-row DataFrame.
    try:
        frame = cu.validate_and_frame(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # 3. Predict (the pipeline handles feature engineering + preprocessing).
    try:
        proba = float(_pipeline.predict_proba(frame)[0, 1])
    except Exception as exc:  # noqa: BLE001 - surface unexpected scoring errors
        return jsonify({"error": f"Failed to score input: {exc}"}), 400

    prediction = "Yes" if proba >= _threshold else "No"

    # 4. Return prediction + probability.
    return jsonify(
        {
            "prediction": prediction,
            "churn_probability": round(proba, 4),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
