import pandas as pd

from pathlib import Path
from predictive_maintenance.config import DATA_DIR


def load_dataset() -> pd.DataFrame:
    """
    Load the predictive maintenance dataset.

    Returns
    pd.DataFrame
        Raw dataset.
    """
    feature_mapping = {
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear"
    }

    dataset_path: Path = DATA_DIR / "ai4i2020.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please download the dataset."
        )

    df = pd.read_csv(dataset_path)
    df = df.rename(columns=feature_mapping)

    return df