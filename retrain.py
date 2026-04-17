# ==========================================
# RETRAIN MODEL WITH NEW DATASET
# IoT Predictive Maintenance
# ==========================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("models", exist_ok=True)

# =========================
# LOAD OLD SAVED MODEL
# =========================
print("\nLoading old saved model...")

old_model_data = joblib.load("models/iot_predictive_maintenance_model.pkl")

old_model = old_model_data["model"]
old_features = old_model_data["selected_features"]

print("Old selected features:")
print(old_features)

# =========================
# LOAD NEW DATASET
# =========================
print("\nLoading new dataset...")

df = pd.read_csv("new_dataset.csv")

print("\nNew dataset loaded successfully.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# =========================
# DEFINE FEATURES AND TARGET
# =========================
# Old features + 2 new features
new_features = old_features + ["pressure", "humidity"]
target_column = "fault"

print("\nNew features used for retraining:")
print(new_features)

# =========================
# PREPARE DATA
# =========================
X = df[new_features]
y = df[target_column]

# =========================
# TRAIN TEST SPLIT
# =========================
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# =========================
# CREATE NEW MODEL
# =========================
# Random Forest is good for sensor data and non-linear patterns
print("\nTraining new model...")

new_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

new_model.fit(X_train, y_train)

# =========================
# EVALUATE MODEL
# =========================
print("\nEvaluating retrained model...")

y_pred = new_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE RETRAINED MODEL
# =========================
retrained_model_path = "models/retrained_iot_model.pkl"

joblib.dump(
    {
        "model": new_model,
        "selected_features": new_features,
        "target_column": target_column
    },
    retrained_model_path
)

print(f"\nRetrained model saved at: {retrained_model_path}")

# =========================
# LOAD RETRAINED MODEL AGAIN
# =========================
print("\nReloading retrained model...")

saved_retrained_model = joblib.load(retrained_model_path)
loaded_model = saved_retrained_model["model"]
loaded_features = saved_retrained_model["selected_features"]

print("Reload successful.")
print("Loaded features:", loaded_features)

# =========================
# PREDICTION WITH NEW INPUT
# =========================
print("\nTesting prediction with new sample input...")

sample_data = pd.DataFrame([
    {
        "vibration": 5.5,
        "temperature": 72,
        "current": 16,
        "acoustic": 78,
        "IMF_1": 1.05,
        "IMF_3": 1.18,
        "timestamp": 92,
        "IMF_2": 1.10,
        "pressure": 114,
        "humidity": 63
    }
])

prediction = loaded_model.predict(sample_data)[0]

print("\nPrediction result:")
if prediction == 1:
    print("Fault Detected")
else:
    print("No Fault Detected")