import pandas as pd


REQUIRED_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "Machine failure",
]

FEATURE_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

TARGET_COLUMN = "Machine failure"


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate that the dataset has required columns.
    """

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def extract_features(df: pd.DataFrame):
    """
    Extract feature matrix and target vector.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y