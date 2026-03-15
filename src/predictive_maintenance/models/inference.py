import joblib
import pandas as pd
from pathlib import Path

from predictive_maintenance.config import MODEL_DIR


MODEL_PATH = MODEL_DIR / "model.joblib"


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train the model first.")

    model = joblib.load(MODEL_PATH)

    return model


def predict(model, input_data: dict):

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability