
# 📦 End-to-End Data Science Project
# ✅ Data collection → preprocessing → model training → deployment with Flask

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from flask import Flask, request, jsonify


# 🔷 Step 1: Data collection & model training


def train_and_save_model():
    """
    Load Boston housing data, train Linear Regression model, and save it.
    """
    print("📊 Training model...")
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "house_price_model.pkl")
    print("✅ Model trained & saved as house_price_model.pkl")


# 🔷 Step 2: Load trained model

def load_model():
    """
    Load the saved model from file.
    """
    if not os.path.exists("house_price_model.pkl"):
        train_and_save_model()
    model = joblib.load("house_price_model.pkl")
    print("🚀 Model loaded successfully!")
    return model



# 🔷 Step 3: Flask API

app = Flask(__name__)
model = load_model()


@app.route("/")
def home():
    """
    Home route to test if API is running.
    """
    return "🏠 House Price Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict route: accepts JSON input with 'features' key,
    and returns predicted house price.
    """
    data = request.get_json(force=True)

    # Check input
    if "features" not in data:
        return jsonify({"error": "Please provide 'features' in request body."})

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({"predicted_price": round(float(prediction), 2)})



# 🔷 Step 4: Run Flask app

if __name__ == "__main__":
    print("🌐 Starting Flask server...")
    app.run(debug=True)