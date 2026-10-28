
import os
import threading
import webbrowser

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, render_template_string, jsonify
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "PCOS_data_without_infertility.xlsx")
GRAPH_FILE = os.path.join(BASE_DIR, "model_accuracy.png")

FEATURES = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)"
]

TARGET = "PCOS (Y/N)"

models = {}
accuracies = {}
training_error = None
model_ready = False


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_value(value):
    """Convert common yes/no and cycle labels to numbers."""

    if pd.isna(value):
        return float("nan")

    if isinstance(value, str):
        value = value.strip().lower()

        if value in ("yes", "y", "true"):
            return 1
        if value in ("no", "n", "false"):
            return 0
        if value in ("regular", "r"):
            return 2
        if value in ("irregular", "i"):
            return 4

        try:
            return float(value)
        except ValueError:
            return float("nan")

    return value


def clean_target(value):
    if pd.isna(value):
        return float("nan")

    if isinstance(value, str):
        value = value.strip().lower()

        if value in ("yes", "y", "true", "positive"):
            return 1
        if value in ("no", "n", "false", "negative"):
            return 0

        try:
            return float(value)
        except ValueError:
            return float("nan")

    return value


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODELS
# --------------------------------------------------

def train_models():
    global models, accuracies, training_error, model_ready

    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                "Excel dataset not found. Put "
                "PCOS_data_without_infertility.xlsx "
                "in the same folder as app.py."
            )

        df = pd.read_excel(DATA_FILE, sheet_name="Full_new")
        df.columns = df.columns.astype(str).str.strip()

        # Check required columns
        missing = [
            col for col in FEATURES + [TARGET]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing Excel columns: " + ", ".join(missing)
            )

        df = df[FEATURES + [TARGET]].copy()

        for col in FEATURES:
            df[col] = df[col].apply(clean_value)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[TARGET] = df[TARGET].apply(clean_target)
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

        df = df.dropna(subset=[TARGET])

        if df.empty:
            raise ValueError("No valid target values were found.")

        # Remove rows without any usable feature values
        df = df.dropna(subset=FEATURES, how="all")

        X = df[FEATURES]
        y = df[TARGET]

        if y.nunique() != 2:
            raise ValueError(
                "The target column must contain two classes, "
                "such as 0 and 1 or No and Yes."
            )

        # Stratify only when both classes have enough rows
        stratify = y if y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify
        )

        model_types = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }

        models = {}
        accuracies = {}

        for name, estimator in model_types.items():
            pipeline = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                estimator
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            models[name] = pipeline
            accuracies[name] = round(
                accuracy_score(y_test, predictions) * 100, 2
            )

        # Save accuracy comparison graph
        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            list(accuracies.keys()),
            list(accuracies.values())
        )
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Machine Learning Model")
        plt.title("PCOS Detection - Model Accuracy Comparison")
        plt.ylim(0, 100)
        plt.xticks(rotation=15)

        for bar, value in zip(bars, accuracies.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(value) + "%",
                ha="center"
            )

        plt.tight_layout()
        plt.savefig(GRAPH_FILE, dpi=150)
        plt.close()

        model_ready = True
        training_error = None

    except Exception as error:
        models = {}
        accuracies = {}
        model_ready = False
        training_error = str(error)


# --------------------------------------------------
# WEB PAGE
# --------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>PCOS Detection Using Machine Learning</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f0f4f8;
            color: #243247;
        }
        header {
            background: #173b67;
            color: white;
            padding: 24px 15px;
            text-align: center;
        }
        main {
            max-width: 900px;
            margin: 24px auto;
            padding: 0 14px;
        }
        .card {
            background: white;
            padding: 22px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px #00000010;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        label {
            display: block;
            font-weight: bold;
            margin-bottom: 7px;
        }
        input, select {
            box-sizing: border-box;
            width: 100%;
            padding: 11px;
            border: 1px solid #ccd4df;
            border-radius: 6px;
            font-size: 15px;
        }
        button {
            background: #173b67;
            color: white;
            border: 0;
            border-radius: 7px;
            padding: 13px 20px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 18px;
        }
        button:hover {
            background: #24558d;
        }
        .result {
            background: #eaf3ff;
            padding: 16px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .error {
            background: #fff0f0;
            padding: 14px;
            border-radius: 8px;
            overflow-wrap: anywhere;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        .muted {
            color: #5e6b7b;
            font-size: 14px;
        }
        footer {
            text-align: center;
            padding: 18px;
            color: #5e6b7b;
            font-size: 13px;
        }
    </style>
</head>

<body>
<header>
    <h1>PCOS Detection Using Machine Learning</h1>
    <p>Machine Learning Prediction and Model Comparison</p>
</header>

<main>
    {% if error %}
    <section class="card error">
        <h2>Training Error</h2>
        <p>{{ error }}</p>
        <p>Check your dataset filename, sheet name and column names.</p>
    </section>
    {% endif %}

    <section class="card">
        <h2>PCOS Risk Prediction</h2>
        <p class="muted">
            Enter the information below. BMI is calculated from
            height and weight. This tool is for educational purposes only.
        </p>

        <form method="POST" action="/">
            <div class="form-grid">
                <div>
                    <label for="age">Age (years)</label>
                    <input id="age" name="age" type="number"
                           min="1" max="100" step="1" required
                           value="{{ values.get('age', '') }}">
                </div>

                <div>
                    <label for="height">Height (cm)</label>
                    <input id="height" name="height" type="number"
                           min="50" max="250" step="0.1" required
                           value="{{ values.get('height', '') }}">
                </div>

                <div>
                    <label for="weight">Weight (kg)</label>
                    <input id="weight" name="weight" type="number"
                           min="10" max="400" step="0.1" required
                           value="{{ values.get('weight', '') }}">
                </div>

                <div>
                    <label for="cycle">Menstrual cycle</label>
                    <select id="cycle" name="cycle">
                        <option value="2"
                            {% if values.get('cycle', '2') == '2' %}selected{% endif %}>
                            Regular
                        </option>
                        <option value="4"
                            {% if values.get('cycle') == '4' %}selected{% endif %}>
                            Irregular
                        </option>
                    </select>
                </div>

                <div>
                    <label for="cycle_length">Cycle length (days)</label>
                    <input id="cycle_length" name="cycle_length"
                           type="number" min="1" max="120"
                           step="1" required
                           value="{{ values.get('cycle_length', '') }}">
                </div>

                <div>
                    <label for="weight_gain">Weight gain</label>
                    <select id="weight_gain" name="weight_gain">
                        <option value="1"
                            {% if values.get('weight_gain', '0') == '1' %}selected{% endif %}>
                            Yes
                        </option>
                        <option value="0"
                            {% if values.get('weight_gain', '0') == '0' %}selected{% endif %}>
                            No
                        </option>
                    </select>
                </div>

                <div>
                    <label for="hair_growth">Excess hair growth</label>
                    <select id="hair_growth" name="hair_growth">
                        <option value="1"
                            {% if values.get('hair_growth', '0') == '1' %}selected{% endif %}>
                            Yes
                        </option>
                        <option value="0"
                            {% if values.get('hair_growth', '0') == '0' %}selected{% endif %}>
                            No
                        </option>
                    </select>
                </div>

                <div>
                    <label for="skin_darkening">Skin darkening</label>
                    <select id="skin_darkening" name="skin_darkening">
                        <option value="1"
                            {% if values.get('skin_darkening', '0') == '1' %}selected{% endif %}>
                            Yes
                        </option>
                        <option value="0"
                            {% if values.get('skin_darkening', '0') == '0' %}selected{% endif %}>
                            No
                        </option>
                    </select>
                </div>
            </div>

            <button type="submit">Calculate BMI and Predict</button>
        </form>

        {% if bmi is not none %}
        <div class="result">
            <h3>Calculated BMI: {{ bmi }}</h3>
            {% if prediction is not none %}
                <h3>Model result: {{ prediction }}</h3>
                <p>Selected model: Random Forest</p>
                <p class="muted">
                    This result is a machine-learning estimate, not a
                    medical diagnosis. Consult a qualified healthcare
                    professional for evaluation.
                </p>
            {% endif %}
        </div>
        {% endif %}
    </section>

    {% if accuracies %}
    <section class="card">
        <h2>Model Accuracy Comparison</h2>
        {% for name, accuracy in accuracies.items() %}
            <p><strong>{{ name }}:</strong> {{ accuracy }}%</p>
        {% endfor %}
        <img src="/dashboard.png" alt="Model accuracy bar graph">
        <p class="muted">
            Accuracy is calculated on a held-out test set.
            Results may change with different data or splits.
        </p>
    </section>
    {% endif %}

    <section class="card">
        <h2>API Testing</h2>
        <p>POST endpoint: <code>/api/predict</code></p>
        <p>Send a JSON body containing age, BMI, cycle, cycle_length,
        weight_gain, hair_growth and skin_darkening.</p>
        <p>Cycle: 2 = regular, 4 = irregular. Symptoms: 1 = yes, 0 = no.</p>
        <p>Use Postman with the URL
        <code>http://127.0.0.1:5001/api/predict</code>
        and select Body → raw → JSON.</p>
    </section>
</main>

<footer>
    Educational project only — not a substitute for medical advice.
</footer>
</body>
</html>
"""


# --------------------------------------------------
# PREDICTION HELPER
# --------------------------------------------------

def make_prediction(data):
    if not model_ready:
        raise RuntimeError(training_error or "Models are not ready.")

    row = pd.DataFrame([{
        "Age (yrs)": float(data["age"]),
        "BMI": float(data["bmi"]),
        "Cycle(R/I)": float(data.get("cycle", 2)),
        "Cycle length(days)": float(data["cycle_length"]),
        "Weight gain(Y/N)": float(data.get("weight_gain", 0)),
        "hair growth(Y/N)": float(data.get("hair_growth", 0)),
        "Skin darkening (Y/N)": float(data.get("skin_darkening", 0))
    }], columns=FEATURES)

    model = models["Random Forest"]
    result = model.predict(row)[0]

    # A predicted class of 1 means the positive class in the dataset.
    prediction = "PCOS-positive pattern predicted" if result == 1 else \
                 "PCOS-negative pattern predicted"

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(
            float(max(model.predict_proba(row)[0])) * 100, 2
        )

    return prediction, probability


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    values = {}
    bmi = None
    prediction = None

    if request.method == "POST":
        values = request.form.to_dict()

        try:
            height = float(request.form["height"])
            weight = float(request.form["weight"])

            if height <= 0 or weight <= 0:
                raise ValueError("Height and weight must be positive.")

            bmi = round(weight / ((height / 100) ** 2), 2)

            data = {
                "age": request.form["age"],
                "bmi": bmi,
                "cycle": request.form["cycle"],
                "cycle_length": request.form["cycle_length"],
                "weight_gain": request.form["weight_gain"],
                "hair_growth": request.form["hair_growth"],
                "skin_darkening": request.form["skin_darkening"]
            }

            prediction, probability = make_prediction(data)
            prediction += (
                " (model confidence for the selected class: "
                + str(probability) + "%)"
                if probability is not None else ""
            )

        except Exception as error:
            prediction = "Unable to predict: " + str(error)

    return render_template_string(
        HTML_PAGE,
        values=values,
        bmi=bmi,
        prediction=prediction,
        accuracies=accuracies,
        error=training_error
    )


# --------------------------------------------------
# GRAPH ROUTE
# --------------------------------------------------

@app.route("/dashboard.png")
def dashboard():
    if not os.path.exists(GRAPH_FILE):
        return "Graph is not available. Check the training error.", 404

    from flask import send_file
    return send_file(GRAPH_FILE, mimetype="image/png")


# --------------------------------------------------
# JSON API FOR POSTMAN
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a valid JSON object."
            }), 400

        required = [
            "age", "cycle_length"
        ]

        missing = [key for key in required if key not in data]

        if "bmi" not in data and not (
            "height" in data and "weight" in data
        ):
            missing.append("bmi or height and weight")

        if missing:
            return jsonify({
                "error": "Missing required fields: " + ", ".join(missing)
            }), 400

        if "bmi" not in data:
            height = float(data["height"])
            weight = float(data["weight"])

            if height <= 0 or weight <= 0:
                raise ValueError("Height and weight must be positive.")

            data["bmi"] = weight / ((height / 100) ** 2)

        prediction, probability = make_prediction(data)

        return jsonify({
            "prediction": prediction,
            "selected_class_probability_percent": probability,
            "bmi": round(float(data["bmi"]), 2),
            "model": "Random Forest",
            "note": "Educational estimate only; not a medical diagnosis."
        })

    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": "Prediction failed: " + str(error)}), 500


# --------------------------------------------------
# SERVER START
# --------------------------------------------------

if __name__ == "__main__":
    print("Training PCOS machine learning models...")
    train_models()

    if model_ready:
        print("Training completed successfully.")
        print("Model accuracies:", accuracies)
        print("Graph saved to:", GRAPH_FILE)
    else:
        print("TRAINING ERROR:", training_error)

    print("Open http://127.0.0.1:5001/ in Chrome")

    # Open browser after Flask starts
    threading.Timer(
        1.5,
        lambda: webbrowser.open("http://127.0.0.1:5001/")
    ).start()

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False,
        threaded=True
    )
