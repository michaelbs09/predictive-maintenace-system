import shap
import pandas as pd


def explain_prediction(model, input_data: dict):

    df = pd.DataFrame([input_data])

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(df)

    if len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[1][0]

    explanation = dict(zip(df.columns, values.tolist()))
    return explanation