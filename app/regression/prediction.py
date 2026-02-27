from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from ..models import RegressionPredictRequest, RegressionPredictResponse
from .storage import get_model, set_model_prediction
from .types import StoredPrediction


def predict_with_stored_model(
    model_id: str, payload: RegressionPredictRequest
) -> RegressionPredictResponse:
    try:
        model = get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    if not payload.feature_values:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one feature value for prediction.",
        )

    unknown = [name for name in payload.feature_values.keys() if name not in model.feature_cols]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature(s): {', '.join(unknown)}",
        )

    ordered_cols = model.feature_cols
    train_matrix = model.x_for_plot[ordered_cols].to_numpy(dtype=float)
    defaults = np.nanmedian(train_matrix, axis=0)
    x_row = defaults.copy()

    parsed_inputs: dict[str, float] = {}
    for idx, col in enumerate(ordered_cols):
        if col in payload.feature_values:
            value = float(payload.feature_values[col])
            x_row[idx] = value
            parsed_inputs[col] = value

    if model.model_type == "spline_smoother":
        feature_name = ordered_cols[0]
        x_val = parsed_inputs.get(feature_name, float(defaults[0]))
        pred_value = float(np.asarray(model.estimator(x_val)).reshape(-1)[0])
        predicted_class = None
        positive_prob = None
    elif model.model_type == "logistic_regression":
        probs = model.estimator.predict_proba(x_row.reshape(1, -1))[0]
        positive_prob = float(probs[1])
        class_index = int(model.estimator.predict(x_row.reshape(1, -1))[0])
        if model.class_labels and 0 <= class_index < len(model.class_labels):
            predicted_class = model.class_labels[class_index]
        else:
            predicted_class = str(class_index)
        pred_value = positive_prob
    else:
        pred_value = float(model.estimator.predict(x_row.reshape(1, -1))[0])
        predicted_class = None
        positive_prob = None

    stored_prediction = StoredPrediction(
        feature_values=parsed_inputs,
        prediction_value=pred_value,
        predicted_class=predicted_class,
        positive_class_probability=positive_prob,
    )
    try:
        set_model_prediction(model_id, stored_prediction)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=exc.args[0]) from exc

    plot_x_in_inputs = model.plot_x in parsed_inputs
    if plot_x_in_inputs:
        message = "Prediction saved. Curve overlay is available."
    else:
        message = "Prediction saved. Curve overlay skipped because plot_x was not provided."

    return RegressionPredictResponse(
        model_id=model_id,
        model_type=model.model_type,
        y=model.y_col,
        prediction=pred_value,
        predicted_class=predicted_class,
        positive_class_probability=positive_prob,
        input_features=parsed_inputs,
        plot_x=model.plot_x,
        plot_x_in_inputs=plot_x_in_inputs,
        message=message,
    )
