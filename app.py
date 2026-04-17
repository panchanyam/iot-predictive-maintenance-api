# =====================================
# IOT PREDICTIVE MAINTENANCE APP
# Beginner-Friendly + Better Postman Errors
# =====================================

from flask import Flask, request, render_template_string, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
#model_data = joblib.load("models/iot_predictive_maintenance_model.pkl")
model_data = joblib.load("models/retrained_iot_model.pkl")
model = model_data["model"]
features = model_data["selected_features"]

# =========================
# HTML TEMPLATE
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IoT Predictive Maintenance</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f6fc;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 550px;
            margin: 40px auto;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 0px 12px #cfcfcf;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        p {
            text-align: center;
            color: #555;
        }
        label {
            font-weight: bold;
            display: block;
            margin-top: 10px;
        }
        input {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 6px;
            border: 1px solid #ccc;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 12px;
            background: #ecf0f1;
            border-radius: 6px;
            text-align: center;
            font-size: 18px;
            color: #2c3e50;
        }
        .note {
            margin-top: 20px;
            font-size: 14px;
            color: #666;
            background: #f9f9f9;
            padding: 10px;
            border-radius: 6px;
        }
        code {
            background: #eee;
            padding: 2px 4px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>IoT Predictive Maintenance</h1>
    <p>Enter sensor values to predict machine condition</p>

    <form method="POST" action="/predict">
        {% for feature in features %}
            <label>{{ feature }}</label>
            <input type="text" name="{{ feature }}" placeholder="Enter {{ feature }}" required>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
        <div class="result">
            <b>Prediction:</b> {{ prediction }}
        </div>
    {% endif %}

    <div class="note">
        <b>Postman:</b> Use <code>POST /predict</code> with raw JSON.<br>
        <b>Check required inputs:</b> <code>/features</code>
    </div>
</div>
</body>
</html>
"""

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return render_template_string(HTML, features=features, prediction=None)

# =========================
# FEATURES
# =========================
@app.route("/features", methods=["GET"])
def show_features():
    return jsonify({
        "required_features": features,
        "sample_json_format": {feature: 0 for feature in features}
    })

# =========================
# API INFO
# =========================
@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "status": "success",
        "message": "API is running",
        "predict_url": "/predict",
        "required_features_url": "/features"
    })

# =========================
# HELPER
# =========================
def prepare_input_dataframe(data_dict):
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a JSON object like {\"feature1\": 10, \"feature2\": 20}")

    missing_features = [feature for feature in features if feature not in data_dict]
    extra_features = [key for key in data_dict if key not in features]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    if extra_features:
        raise ValueError(f"Extra features found: {extra_features}. Only use: {features}")

    input_df = pd.DataFrame([data_dict])

    # Reorder columns exactly as training time
    input_df = input_df[features]

    try:
        input_df = input_df.astype(float)
    except Exception:
        raise ValueError("All input values must be numeric")

    return input_df

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON request from Postman
        if request.is_json:
            data = request.get_json()

            if data is None:
                return jsonify({
                    "status": "error",
                    "message": "No JSON received. In Postman choose Body -> raw -> JSON"
                }), 400

            input_df = prepare_input_dataframe(data)
            prediction = model.predict(input_df)[0]

            return jsonify({
                "status": "success",
                "prediction": float(prediction) if isinstance(prediction, float) else int(prediction)
            })

        # Form request from browser
        else:
            data = request.form.to_dict()
            input_df = prepare_input_dataframe(data)
            prediction = model.predict(input_df)[0]

            return render_template_string(
                HTML,
                features=features,
                prediction=prediction
            )

    except Exception as e:
        if request.is_json:
            return jsonify({
                "status": "error",
                "message": str(e),
                "required_features": features
            }), 400

        return render_template_string(
            HTML,
            features=features,
            prediction=f"Error: {str(e)}"
        )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Starting Flask app...")
    print("Browser: http://127.0.0.1:5001")
    print("Features: http://127.0.0.1:5001/features")
    app.run(debug=True, host="0.0.0.0", port=5001)