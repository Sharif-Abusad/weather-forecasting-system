"""
Model Loader Service
────────────────────────────────────────────────────────────
Loads:
1. Trained LSTM temperature forecasting model
2. Feature scaler
3. Model metadata

This module centralizes all model-loading logic so
other services can directly import preloaded objects.

Project: Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model
from app.config import LSTM_META, MODEL_LSTM, SCALER_PATH
# ──────────────────────────────────────────────────────────
# Project Base Directory
# ──────────────────────────────────────────────────────────
# BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────
# Model Paths
# ──────────────────────────────────────────────────────────
# MODEL_PATH = (
#     BASE_DIR /
#     "models" /
#     "lstm" /
#     "lstm_temp_model.keras"
# )

# SCALER_PATH = (
#     BASE_DIR /
#     "models" /
#     "scalers" /
#     "lstm_temp_scaler.pkl"
# )

# META_PATH = (
#     BASE_DIR /
#     "models" /
#     "lstm" /
#     "model_meta.json"
# )

# ──────────────────────────────────────────────────────────
# Custom TensorFlow Functions
# Required for loading Lambda layers safely
# ──────────────────────────────────────────────────────────
# def squeeze_last_dim(x):
#     """
#     Remove the final dimension from tensor.

#     Used inside custom Lambda layer during
#     model loading.
#     """

#     return tf.squeeze(x, axis=-1)


# def squeeze_output_shape(input_shape):
#     """
#     Define output shape after squeeze operation.

#     Args:
#         input_shape : Tensor input shape

#     Returns:
#         Tuple representing squeezed shape
#     """

#     return (
#         input_shape[0],
#         input_shape[1]
#     )

# ──────────────────────────────────────────────────────────
# # Load Trained LSTM Model
# # ──────────────────────────────────────────────────────────
# loaded_lstm_model = load_model(
#     # MODEL_PATH,
#     MODEL_LSTM,
#     custom_objects={
#         "squeeze_last_dim": squeeze_last_dim,
#         "squeeze_output_shape": squeeze_output_shape,
#     },

#     safe_mode=False,
# )
loaded_lstm_model = load_model(
    MODEL_LSTM,
    compile=False
)
# ──────────────────────────────────────────────────────────
# Load Feature Scaler
# ──────────────────────────────────────────────────────────
loaded_lstm_scaler = joblib.load(
    SCALER_PATH
)

# ──────────────────────────────────────────────────────────
# Load Model Metadata
# ──────────────────────────────────────────────────────────
with open(LSTM_META, "r") as file:
    loaded_lstm_meta = json.load(file)

# ──────────────────────────────────────────────────────────
# Model Startup Logs
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("🚀 Weather Forecasting Models Initialized")
print("=" * 55)

print(f"✅ LSTM Model Loaded")
print(f"📦 Forecast Horizon : {loaded_lstm_meta['forecast_n']} hours")
print(f"📊 Feature Count    : {loaded_lstm_meta['n_features']}")
print(f"📉 Validation MAE   : {loaded_lstm_meta['metrics']['mae_celsius']} °C")

print("=" * 55 + "\n")
