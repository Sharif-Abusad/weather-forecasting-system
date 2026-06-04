"""
ML Service Layer
────────────────────────────────────────────────────────────
Handles:
1. Rain prediction using XGBoost
2. Temperature forecasting using LSTM
4. Weather feature engineering utilities

Project: Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────
import json

# ──────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────
# Internal Imports
# ──────────────────────────────────────────────────────────
from app.services.model_loader import (
    loaded_lstm_model,
    loaded_lstm_scaler,
    loaded_lstm_meta
)
from app.config import MODEL_XGBOOST, XGBOOST_META
from app.utils.weather_utils import calculate_humidity


with open(XGBOOST_META, "r") as file:
    meta = json.load(file)
    

BEST_THRESHOLD = meta["best_threshold"]
FEATURES = meta["features"]

# ──────────────────────────────────────────────────────────
# Load Trained XGBoost Model
# ──────────────────────────────────────────────────────────
rain_model = XGBClassifier()
rain_model.load_model(MODEL_XGBOOST)

# ──────────────────────────────────────────────────────────
# Rain Prediction Service
# ──────────────────────────────────────────────────────────
def predict_rain(
    input_dict: dict,
    threshold: float = BEST_THRESHOLD
) -> dict:
    """
    Predict whether it will rain tomorrow.

    Args:
        input_dict : Dictionary containing weather features
        threshold  : Probability threshold for classification

    Returns:
        Dictionary containing:
            - rain_tomorrow
            - probability
            - confidence
            - threshold_used
    """

    # Create dataframe in exact feature order
    row = pd.DataFrame([
        {
            feature: input_dict[feature]
            for feature in FEATURES
        }
    ])

    # Rain probability
    probability = rain_model.predict_proba(row)[0][1]

    # Binary prediction
    will_rain = probability >= threshold

    # Confidence level
    if probability > 0.75:
        confidence = "high (rain likely)"

    elif probability > 0.55:
        confidence = "medium"

    elif probability < 0.25:
        confidence = "high (no rain likely)"

    else:
        confidence = "low (uncertain)"

    return {
        "rain_tomorrow": bool(will_rain),
        "probability": round(float(probability), 4),
        "confidence": confidence,
        "threshold_used": threshold,
    }

# ──────────────────────────────────────────────────────────
# Temperature Forecasting Service
# ──────────────────────────────────────────────────────────
def predict_temperature(
    recent_df: pd.DataFrame,
    n_hours: int = 5
) -> dict:
    """
    Predict future temperature using trained LSTM model.

    Args:
        recent_df : DataFrame containing recent weather data
        n_hours   : Number of future hours to predict

    Returns:
        Dictionary containing:
            - forecast
            - model metadata
            - MAE score
    """

    # ─────────────────────────────────────────────
    # Load metadata
    # ─────────────────────────────────────────────
    meta = loaded_lstm_meta

    model = loaded_lstm_model
    scaler = loaded_lstm_scaler

    lookback = meta["lookback"]
    forecast_n = meta["forecast_n"]

    feature_columns = meta["features"]

    target_idx = meta["target_idx"]
    n_features = meta["n_features"]

    # ─────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────
    n = n_hours if n_hours else forecast_n

    assert n <= forecast_n, (
        f"n_hours={n} exceeds trained "
        f"forecast_n={forecast_n}"
    )

    assert len(recent_df) >= lookback, (
        f"Need at least {lookback} rows, "
        f"got {len(recent_df)}"
    )

    # ─────────────────────────────────────────────
    # Prepare model input
    # ─────────────────────────────────────────────
    window = (
        recent_df[feature_columns]
        .iloc[-lookback:]
        .values
    )

    window_scaled = scaler.transform(window)

    window_scaled = window_scaled.reshape(
        1,
        lookback,
        n_features
    )

    # ─────────────────────────────────────────────
    # Model prediction
    # ─────────────────────────────────────────────
    prediction_scaled = (
        model.predict(
            window_scaled,
            verbose=0
        )[0][:n]
    )

    # ─────────────────────────────────────────────
    # Inverse scaling
    # ─────────────────────────────────────────────
    dummy = np.zeros((n, n_features))

    dummy[:, target_idx] = prediction_scaled

    prediction_temp = (
        scaler.inverse_transform(dummy)
        [:, target_idx]
    )

    # ─────────────────────────────────────────────
    # Build forecast response
    # ─────────────────────────────────────────────
    forecast = [
        {
            "hour": f"+{i + 1}h",
            "temperature_C": round(float(temp), 2)
        }
        for i, temp in enumerate(prediction_temp)
    ]

    return {
        "forecast": forecast,
    }
