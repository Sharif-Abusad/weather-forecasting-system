"""
pipeline/train_lstm.py
━━━━━━━━━━━━━━━━━━━━━━
Trains a Bidirectional LSTM encoder-decoder model for
multi-step hourly temperature forecasting.

Input  : data/processed/azamgarh_weather_final.csv
Output : models/lstm/lstm_temp_model.keras
         models/lstm/model_meta.json
         models/scalers/lstm_temp_scaler.pkl

Usage:
    python pipeline/train_lstm.py              # use config defaults
    python pipeline/train_lstm.py --n 10       # forecast 10 hours ahead
    python pipeline/train_lstm.py --lookback 48  # 48h input window
"""

# ──────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────
import os
import sys
import json
import random
import argparse
import warnings

# ──────────────────────────────────────────────────────────
# Reproducibility Configuration
# Ensures deterministic results across runs.
# ──────────────────────────────────────────────────────────
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)

# ──────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────
import numpy as np

np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)

import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    RepeatVector,
    TimeDistributed,
    Reshape,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import (
    GlorotUniform,
    Orthogonal,
)

# ──────────────────────────────────────────────────────────
# Project Configuration
# ──────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.config import (
    DATA_PROCESSED_DIR,
    MODEL_LSTM_DIR,
    SCALER_DIR,
    LSTM_MODEL_FILE,
    LSTM_META_FILE,
    SCALER_FILE,
    FORECAST_N,
    LOOKBACK,
)

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "temperature",        # index 0 — TARGET (always first)
    "temp_lag_1",         # 1h ago  — autocorr ~0.99
    "temp_lag_24",        # same hour yesterday — autocorr ~0.85
    "temp_rolling_6",     # 6h smooth trend
    "humidity",           # strong inverse corr with temp
    "surface_pressure",   # pressure systems drive temp change
    "total_cloud_cover",  # clouds moderate temperature swings
    "wind_speed",         # advection effect
    "hour_sin",           # sin(2π × hour / 24)
    "hour_cos",           # cos(2π × hour / 24)
    "month_sin",          # sin(2π × month / 12)
    "month_cos",          # cos(2π × month / 12)
]
TARGET_IDX = 0            # temperature is at column index 0

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS     = 150
LSTM_UNITS = 64
LR_INITIAL = 5e-4
LR_MIN     = 1e-6

# ──────────────────────────────────────────────────────────
# Dataset Preparation
# Converts continuous weather observations into supervised
# learning sequences using a sliding-window approach.
#
# Example:
# Input  : previous 24 hours
# Output : next 5 hours temperatures
# ──────────────────────────────────────────────────────────
def create_sequences(
    data: np.ndarray,
    lookback: int,
    forecast_n: int,
    target_idx: int = TARGET_IDX,
):
    """
    Sliding window: past lookback hours → next forecast_n temperatures.

    Returns:
        X : (samples, lookback, n_features)
        y : (samples, forecast_n)  — temperature column only
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_n + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + forecast_n, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────────────────────────
# BiLSTM Encoder-Decoder Architecture
# ──────────────────────────────────────────────────────────
def build_model(
    lookback: int,
    forecast_n: int,
    n_features: int,
) -> Model:
    """
    Build a Bidirectional LSTM encoder-decoder model for
    multi-step temperature forecasting.
    """

    # ── Weight Initializers ────────────────────────────────
    kernel_init = GlorotUniform(seed=SEED)
    recurrent_init = Orthogonal(
        gain=1.0,
        seed=SEED
    )

    # ── Input Layer ────────────────────────────────────────
    inputs = Input(
        shape=(lookback, n_features),
        name="encoder_input"
    )

    # ======================================================
    # Encoder
    # Past weather sequence → latent representation
    # ======================================================
    x = Bidirectional(
        LSTM(
            LSTM_UNITS,
            return_sequences=True,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
            dropout=0.10,
            recurrent_dropout=0.0,
        ),
        name="bilstm_1",
    )(inputs)

    x = Dropout(
        0.15,
        seed=SEED,
        name="dropout_enc"
    )(x)

    x = LSTM(
        LSTM_UNITS // 2,
        return_sequences=False,
        kernel_initializer=kernel_init,
        recurrent_initializer=recurrent_init,
        dropout=0.10,
        name="lstm_encoder",
    )(x)

    # ======================================================
    # Bridge
    # Repeat latent vector for decoder input
    # ======================================================
    x = RepeatVector(
        forecast_n,
        name="repeat_vector"
    )(x)

    # ======================================================
    # Decoder
    # Generate future temperature sequence
    # ======================================================
    x = LSTM(
        LSTM_UNITS // 2,
        return_sequences=True,
        kernel_initializer=kernel_init,
        recurrent_initializer=recurrent_init,
        dropout=0.10,
        name="lstm_decoder",
    )(x)

    x = Dropout(
        0.15,
        seed=SEED,
        name="dropout_dec"
    )(x)

    # ======================================================
    # Output Layer
    # ======================================================
    x = TimeDistributed(
        Dense(
            1,
            kernel_initializer=kernel_init
        ),
        name="td_dense",
    )(x)

    outputs = Reshape(
        (forecast_n,),
        name="output"
    )(x)

    # ======================================================
    # Model Compilation
    # ======================================================
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="BiLSTM_TempForecaster",
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LR_INITIAL,
            clipnorm=1.0,
        ),
        loss="huber",
        metrics=["mae"],
    )

    return model

# ──────────────────────────────────────────────────────────
# Temperature Inverse Scaling Utility
#
# Converts normalized model outputs back to real-world
# temperature values in degrees Celsius.
# ──────────────────────────────────────────────────────────
def inverse_temp(
    scaled_vals: np.ndarray,
    scaler: MinMaxScaler,
    n_features: int,
) -> np.ndarray:
    """
    Reconstruct real °C values from scaled temperature predictions.

    We fill a zero array of full feature width, put scaled temps at
    TARGET_IDX, then inverse_transform and extract that column.
    The scaler must be the same object used during training.
    """
    dummy = np.zeros((len(scaled_vals), n_features), dtype=np.float32)
    dummy[:, TARGET_IDX] = scaled_vals
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


# ──────────────────────────────────────────────────────────
# Inference Helper
#
# Used by the FastAPI application to generate future
# temperature forecasts from the latest weather observations.
# ──────────────────────────────────────────────────────────
def predict_temperature(
    model: Model,
    scaler: MinMaxScaler,
    recent_df: pd.DataFrame,
    n_hours: int = None,
    lookback: int = LOOKBACK,
    forecast_n: int = FORECAST_N,
) -> dict:
    """
    Forecast temperature for the next n_hours.

    Args:
        model     : loaded Keras model
        scaler    : the SAME MinMaxScaler saved during training
        recent_df : DataFrame with FEATURES columns, at least lookback rows
        n_hours   : override horizon (must be <= trained forecast_n)
    """
    n = n_hours or forecast_n
    assert n <= forecast_n, \
        f"n_hours={n} > trained forecast_n={forecast_n}. Retrain with larger FORECAST_N."
    assert len(recent_df) >= lookback, \
        f"Need >= {lookback} rows, got {len(recent_df)}"

    window      = recent_df[FEATURES].iloc[-lookback:].values.astype(np.float32)
    scaled      = scaler.transform(window).reshape(1, lookback, len(FEATURES))
    pred_scaled = model.predict(scaled, verbose=0)[0][:n]

    dummy = np.zeros((n, len(FEATURES)), dtype=np.float32)
    dummy[:, TARGET_IDX] = pred_scaled
    temps = scaler.inverse_transform(dummy)[:, TARGET_IDX]

    return {
        "forecast": [
            {"hour": f"+{i+1}h", "temperature_C": round(float(t), 2)}
            for i, t in enumerate(temps)
        ],
        "n_hours": n,
    }


# ──────────────────────────────────────────────────────────
# Training Pipeline
#
# Workflow:
#   1. Load dataset
#   2. Scale features
#   3. Generate sequences
#   4. Train/Validation/Test split
#   5. Build model
#   6. Train model
#   7. Evaluate performance
#   8. Save artifacts
# ──────────────────────────────────────────────────────────
def main(lookback: int = LOOKBACK, forecast_n: int = FORECAST_N):
    print("=" * 60)
    print("  TRAIN_LSTM.PY — BiLSTM Temperature Forecasting")
    print("=" * 60)
    print(f"  Lookback  : {lookback}h  |  Forecast N : {forecast_n}h")
    print(f"  Features  : {len(FEATURES)}  →  {FEATURES}")
    print(f"  TF version: {tf.__version__}")

    os.makedirs(MODEL_LSTM_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR,     exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    data_path = os.path.join(DATA_PROCESSED_DIR, "azamgarh_weather_final.csv")
    print(f"\n[1/7] Loading → {data_path}")
    df = pd.read_csv(data_path, parse_dates=["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n  ❌ Missing columns: {missing}"
            f"\n  Run feature_engineer.py first and ensure it adds:"
            f"\n  hour_sin, hour_cos, month_sin, month_cos"
        )

    df_model = df[["valid_time"] + FEATURES].dropna().reset_index(drop=True)
    print(f"  Rows: {len(df_model):,}  |  Columns: {df_model.shape[1]}")
    print(f"  Temp range in dataset: "
          f"{df_model['temperature'].min():.1f}°C → "
          f"{df_model['temperature'].max():.1f}°C")

    # ── 2. Scale ───────────────────────────────────────────────────────────────
    # FIX: fit scaler on ENTIRE dataset so temperature range is fully captured.
    # Fitting on 70% only causes inverse_transform to produce wrong °C values
    # when test set contains temperatures outside the 70% training range.
    print("\n[2/7] Fitting MinMaxScaler on full dataset...")
    n_features = len(FEATURES)
    scaler     = MinMaxScaler(feature_range=(0, 1))
    scaled     = scaler.fit_transform(df_model[FEATURES].values).astype(np.float32)

    # Quick sanity check — scaled temperature should be in [0, 1]
    temp_scaled_min = scaled[:, TARGET_IDX].min()
    temp_scaled_max = scaled[:, TARGET_IDX].max()
    print(f"  Scaled temp range : [{temp_scaled_min:.4f}, {temp_scaled_max:.4f}]  (expect [0.0, 1.0])")

    scaler_path = os.path.join(SCALER_DIR, SCALER_FILE)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    # ── 3. Sequences ───────────────────────────────────────────────────────────
    print("\n[3/7] Creating sliding window sequences...")
    X, y = create_sequences(scaled, lookback, forecast_n, TARGET_IDX)
    print(f"  X : {X.shape}  |  y : {y.shape}")
    print(f"  y scaled range : [{y.min():.4f}, {y.max():.4f}]  (expect [0–1])")

    # Verify inverse_transform works correctly before training
    test_inv = inverse_temp(y[0], scaler, n_features)
    print(f"  Inverse transform check:")
    print(f"    Scaled y[0]   : {y[0]}")
    print(f"    Celsius y[0]  : {test_inv.round(2)}  ← must look like real temps")

    # ── 4. Split ───────────────────────────────────────────────────────────────
    print("\n[4/7] Time-based split: 70 / 15 / 15 ...")
    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end],  y[train_end:val_end]
    X_test,  y_test  = X[val_end:],           y[val_end:]

    print(f"  Train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")

    # ── 5. Build ───────────────────────────────────────────────────────────────
    print("\n[5/7] Building BiLSTM encoder-decoder...")
    model = build_model(lookback, forecast_n, n_features)
    model.summary()
    print(f"\n  Trainable params: {model.count_params():,}")

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print(f"\n[6/7] Training (max {EPOCHS} epochs)...")
    ckpt_path = os.path.join(MODEL_LSTM_DIR, "best_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1, min_delta=1e-4,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=LR_MIN, verbose=1, cooldown=3,
        ),
        ModelCheckpoint(
            filepath=ckpt_path, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,      # NEVER shuffle time series
        verbose=1,
    )

    best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    print(f"\n  Best epoch : {best_epoch}  |  Best val_loss : {best_val_loss:.6f}")

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    print("\n[7/7] Evaluating on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Convert ALL predictions and targets back to °C
    y_true_C = np.array([
        inverse_temp(y_test[i],        scaler, n_features)
        for i in range(len(y_test))
    ])
    y_pred_C = np.array([
        inverse_temp(y_pred_scaled[i], scaler, n_features)
        for i in range(len(y_pred_scaled))
    ])

    # Sanity check — values must be in realistic temp range
    print(f"\n  Actual temp range (test) : "
          f"[{y_true_C.min():.1f}°C, {y_true_C.max():.1f}°C]")
    print(f"  Predicted temp range     : "
          f"[{y_pred_C.min():.1f}°C, {y_pred_C.max():.1f}°C]")

    mae  = mean_absolute_error(y_true_C.flatten(), y_pred_C.flatten())
    rmse = np.sqrt(mean_squared_error(y_true_C.flatten(), y_pred_C.flatten()))
    r2   = r2_score(y_true_C.flatten(), y_pred_C.flatten())

    print(f"\n  {'─'*45}")
    print(f"  OVERALL  MAE={mae:.4f}°C  RMSE={rmse:.4f}°C  R²={r2:.4f}")
    print(f"  {'─'*45}")
    print(f"\n  {'Step':>6} {'MAE (°C)':>10} {'RMSE (°C)':>11} {'R²':>8}")
    print(f"  {'─'*40}")

    per_step = {}
    for step in range(forecast_n):
        s_mae  = mean_absolute_error(y_true_C[:, step], y_pred_C[:, step])
        s_rmse = np.sqrt(mean_squared_error(y_true_C[:, step], y_pred_C[:, step]))
        s_r2   = r2_score(y_true_C[:, step], y_pred_C[:, step])
        per_step[f"+{step+1}h"] = {
            "mae" : round(s_mae,  4),
            "rmse": round(s_rmse, 4),
            "r2"  : round(s_r2,   4),
        }
        print(f"  {f'+{step+1}h':>6} {s_mae:>10.4f} {s_rmse:>11.4f} {s_r2:>8.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_LSTM_DIR, LSTM_MODEL_FILE)
    model.save(model_path)
    print(f"\n  Model saved    → {model_path}")

    meta = {
        "model"           : "BiLSTM Encoder-Decoder",
        "target"          : "temperature",
        "features"        : FEATURES,
        "target_idx"      : TARGET_IDX,
        "lookback"        : lookback,
        "forecast_n"      : forecast_n,
        "n_features"      : n_features,
        "lstm_units"      : LSTM_UNITS,
        "batch_size"      : BATCH_SIZE,
        "learning_rate"   : LR_INITIAL,
        "seed"            : SEED,
        "best_epoch"      : best_epoch,
        "best_val_loss"   : round(best_val_loss, 6),
        "train_samples"   : int(len(X_train)),
        "val_samples"     : int(len(X_val)),
        "test_samples"    : int(len(X_test)),
        "metrics": {
            "mae_celsius"  : round(float(mae),  4),
            "rmse_celsius" : round(float(rmse), 4),
            "r2_score"     : round(float(r2),   4),
        },
        "per_step_metrics" : per_step,
        "scaler_path"      : scaler_path,
    }
    meta_path = os.path.join(MODEL_LSTM_DIR, LSTM_META_FILE)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ LSTM training complete")
    print(f"     MAE  : {mae:.4f} °C")
    print(f"     RMSE : {rmse:.4f} °C")
    print(f"     R²   : {r2:.4f}")
    print(f"{'=' * 60}")

    return model, scaler, meta


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BiLSTM temperature forecasting model"
    )
    parser.add_argument("--n",        type=int, default=FORECAST_N,
                        help=f"Forecast horizon in hours (default: {FORECAST_N})")
    parser.add_argument("--lookback", type=int, default=LOOKBACK,
                        help=f"Lookback window in hours (default: {LOOKBACK})")
    args = parser.parse_args()
    main(lookback=args.lookback, forecast_n=args.n)