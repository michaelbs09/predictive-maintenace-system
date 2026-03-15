from pydantic import BaseModel


class SensorInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


class PredictionResponse(BaseModel):
    failure_probability: float
    failure_prediction: bool
    explanation: dict