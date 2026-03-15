from predictive_maintenance.data.loader import load_dataset
from predictive_maintenance.data.split import split_dataset
from predictive_maintenance.data.preprocessing import (
    validate_dataset,
    extract_features,
)


def prepare_dataset():

    df = load_dataset()

    validate_dataset(df)

    X, y = extract_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    return X_train, X_test, y_train, y_test