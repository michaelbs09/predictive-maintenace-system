from fastapi import FastAPI

from predictive_maintenance.api.schemas import SensorInput, PredictionResponse
from predictive_maintenance.models.inference import load_model, predict
from predictive_maintenance.explainability.shap_explainer import explain_prediction


app = FastAPI(title="Predictive Maintenance API")

model = load_model()


@app.post("/predict", response_model=PredictionResponse)
def predict_failure(data: SensorInput):

    input_dict = data.dict()

    prediction, probability = predict(model, input_dict)

    explanation = explain_prediction(model, input_dict)

    return PredictionResponse(
        failure_probability=probability,
        failure_prediction=bool(prediction),
        explanation=explanation,
    )