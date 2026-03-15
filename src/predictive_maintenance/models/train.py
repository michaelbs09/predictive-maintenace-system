import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from predictive_maintenance.config import MODEL_DIR
from predictive_maintenance.data.prepare_dataset import prepare_dataset


def train_model() -> None:
    """
    Train predictive maintenance model and log experiment.
    """

    X_train, X_test, y_train, y_test = prepare_dataset()

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, "model")

        MODEL_DIR.mkdir(exist_ok=True)

        model_path = MODEL_DIR / "model.joblib"

        joblib.dump(model, model_path)

        print(f"Model saved to {model_path}")