
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "pcos_model.joblib")

app = Flask(__name__)

# Load the trained ML model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run pcos_ml.py first."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
model_name = saved["model_name"]


def predict_patient(data):
    age = float(data["age"])
    height_cm = float(data["height_cm"])
    weight_kg = float(data["weight_kg"])
    cycle_length = float(data["cycle_length"])

    if not 10 <= age <= 60:
        raise ValueError("Age must be between 10 and 60.")
    if not 100 <= height_cm <= 230:
        raise ValueError("Height must be between 100 and 230 cm.")
    if not 20 <= weight_kg <= 300:
        raise ValueError("Weight must be between 20 and 300 kg.")
    if not 1 <= cycle_length <= 90:
        raise ValueError("Cycle length must be between 1 and 90 days.")

    bmi = weight_kg / ((height_cm / 100) ** 2)

    cycle_type = int(data["cycle_type"])
    weight_gain = int(data["weight_gain"])
    hair_growth = int(data["hair_growth"])
    skin_darkening = int(data["skin_darkening"])

    if cycle_type not in [2, 4]:
        raise ValueError("Select a valid menstrual cycle type.")

    for value in [weight_gain, hair_growth, skin_darkening]:
        if value not in [0, 1]:
            raise ValueError("Symptom values must be Yes or No.")

    patient = pd.DataFrame([{
        "Age (yrs)": age,
        "BMI": bmi,
        "Cycle(R/I)": cycle_type,
        "Cycle length(days)": cycle_length,
        "Weight gain(Y/N)": weight_gain,
        "hair growth(Y/N)": hair_growth,
        "Skin darkening (Y/N)": skin_darkening
    }])[features]

    prediction = int(model.predict(patient)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "prediction": prediction,
        "result": (
            "PCOS-positive classification"
            if prediction == 1
            else "PCOS-negative classification"
        ),
        "bmi": round(bmi, 2),
        "model": model_name,
        "model_confidence_percent": confidence,
        "notice": (
            "Educational ML output only; not a medical diagnosis."
        )
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_name=model_name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({
            "error": "Send the request as JSON."
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON object is required."
        }), 400

    required = [
        "age", "height_cm", "weight_kg", "cycle_length",
        "cycle_type", "weight_gain", "hair_growth",
        "skin_darkening"
    ]

    missing = [key for key in required if key not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing
        }), 400

    try:
        result = predict_patient(data)
        return jsonify(result), 200
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    print("PCOS interface starting...")
    print("Model:", model_name)
    app.run(host="127.0.0.1", port=5000, debug=False)
