"""
pipeline/feature_engineer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Builds all ML features from the cleaned ERA5 CSV.

Input  : data/processed/azamgarh_weather_clean.csv
Output : data/processed/azamgarh_weather_final.csv

Features engineered:
    Temporal  : hour, day, month, day_of_week
    Lag       : temp_lag_1, temp_lag_24
    Rolling   : temp_rolling_6
    Target    : rain_tomorrow (binary: next 24h precip > 0.1 mm)
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATA_PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join(DATA_PROCESSED_DIR, "azamgarh_weather_clean.csv")
OUTPUT_PATH = os.path.join(DATA_PROCESSED_DIR, "azamgarh_weather_final.csv")

# ── Config ────────────────────────────────────────────────────────────────────
LAG_HOURS         = [1, 24]          # hours to shift for lag features
ROLLING_WINDOW    = 6                # hours for rolling mean
RAIN_THRESHOLD_MM = 1             # mm — standard meteorological measurable rain
RAIN_LOOKAHEAD_H  = 24               # hours ahead to check for rain


# ── Feature builders ──────────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from valid_time."""
    df["hour"]        = df["valid_time"].dt.hour
    df["day"]         = df["valid_time"].dt.day
    df["month"]       = df["valid_time"].dt.month
    df["day_of_week"] = df["valid_time"].dt.dayofweek   # 0=Monday
    print(f"  Temporal features added: hour, day, month, day_of_week")
    return df


def add_lag_features(df: pd.DataFrame, lags: list = LAG_HOURS) -> pd.DataFrame:
    """
    Shift temperature column by N hours.

    temp_lag_1  = temperature 1 hour ago  (autocorr ~0.99)
    temp_lag_24 = temperature 24 hours ago (same hour yesterday, autocorr ~0.85)
    """
    for lag in lags:
        col = f"temp_lag_{lag}"
        df[col] = df["temperature"].shift(lag)
        print(f"  {col} created (shift={lag})")
    return df


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Rolling mean of temperature over the last N hours.

    Smooths short-term noise and provides the model with recent trend context.
    First (window-1) rows will have NaN — dropped in cleanup step.
    """
    col = f"temp_rolling_{window}"
    df[col] = df["temperature"].rolling(window=window, min_periods=window).mean().round(4)
    print(f"  temp_rolling_{window} created (window={window}h)")
    return df


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic (sin/cos) encodings for temporal features.

    Converts hour and month values into sine and cosine
    representations to preserve their cyclical nature
    (e.g., 23:00 is close to 00:00, December is close to January).

    Features created:
        - hour_sin
        - hour_cos
        - month_sin
        - month_cos

    Args:
        df (pd.DataFrame): Input dataframe containing
            'hour' and 'month' columns.

    Returns:
        pd.DataFrame: Dataframe with added cyclic features.
    """   
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24).round(6)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24).round(6)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(6)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(6)
    return df


def add_rain_tomorrow(
    df: pd.DataFrame,
    threshold: float = RAIN_THRESHOLD_MM,
    lookahead: int   = RAIN_LOOKAHEAD_H,
) -> pd.DataFrame:
    """
    Binary target: will it rain in the next `lookahead` hours?

    Method:
        1. Shift precipitation column backwards by 1 (so row t looks at t+1 onwards)
        2. Sum over rolling `lookahead` window going forward
        3. Label 1 if sum > threshold, else 0

    Equivalent to:
        next_24h_precip = sum(precip[t+1 : t+lookahead+1])
        rain_tomorrow   = 1 if next_24h_precip > threshold else 0

    The last `lookahead` rows will have NaN (dropped in cleanup).
    """
    # shift(-1) moves future values to current row, then rolling sums forward
    next_precip = sum(
        df["precipitation"].shift(-i)
        for i in range(1, 25)
    )
    df["rain_tomorrow"] = (next_precip > threshold).astype("Int64")  # nullable int

    rain_count   = df["rain_tomorrow"].sum()
    total_valid  = df["rain_tomorrow"].notna().sum()
    rain_pct     = 100 * rain_count / total_valid if total_valid else 0

    print(f"  rain_tomorrow created (threshold={threshold}mm, lookahead={lookahead}h)")
    print(f"    Rain=1  : {rain_count:,}  ({rain_pct:.1f}%)")
    print(f"    Rain=0  : {total_valid - rain_count:,}  ({100 - rain_pct:.1f}%)")
    print(f"    Imbalance ratio: {(total_valid - rain_count) / max(rain_count, 1):.1f}:1")
    return df


def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN in any feature or target column.

    NaNs arise from:
        - Lag features: first N rows (e.g. temp_lag_24 → first 24 rows)
        - Rolling mean: first (window-1) rows
        - rain_tomorrow: last 24 rows
    """
    feature_cols = [
        "temperature", "surface_pressure", "total_cloud_cover",
        "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
        "precipitation", "humidity", "wind_speed",
        "temp_rolling_6", "temp_lag_1", "temp_lag_24", "rain_tomorrow",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    before   = len(df)
    df       = df.dropna(subset=existing).reset_index(drop=True)
    dropped  = before - len(df)
    print(f"  Dropped {dropped} NaN rows → {len(df):,} rows remain")
    return df


def validate_features(df: pd.DataFrame) -> None:
    """Quick sanity checks on engineered features."""
    errors = []

    # Check lag-1 matches shift(1)
    expected_lag1 = df["temperature"].shift(1)
    max_diff_lag1 = (expected_lag1 - df["temp_lag_1"]).abs().max()
    if max_diff_lag1 > 0.01:
        errors.append(f"temp_lag_1 mismatch (max diff={max_diff_lag1:.4f})")

    # Check rain_tomorrow is binary
    unique_vals = df["rain_tomorrow"].dropna().unique()
    if not all(v in [0, 1] for v in unique_vals):
        errors.append(f"rain_tomorrow has non-binary values: {unique_vals}")

    # Check no nulls remain
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    if len(remaining_nulls) > 0:
        errors.append(f"Null values remain:\n{remaining_nulls}")

    if errors:
        print("  ⚠️  Validation warnings:")
        for e in errors:
            print(f"     - {e}")
    else:
        print("  Validation: ✅ all checks passed")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'─' * 55}")
    print("  FINAL FEATURE SUMMARY")
    print(f"{'─' * 55}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  {'Column':<22} {'dtype':<10} {'nulls':>6}  {'min':>8}  {'max':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*6}  {'─'*8}  {'─'*8}")
    for col in df.columns:
        if col == "valid_time":
            print(f"  {col:<22} {'datetime':<10} {'0':>6}  {'–':>8}  {'–':>8}")
        else:
            nulls  = df[col].isnull().sum()
            mn     = f"{df[col].min():.2f}" if pd.api.types.is_numeric_dtype(df[col]) else "–"
            mx     = f"{df[col].max():.2f}" if pd.api.types.is_numeric_dtype(df[col]) else "–"
            print(f"  {col:<22} {str(df[col].dtype):<10} {nulls:>6}  {mn:>8}  {mx:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  FEATURE_ENGINEER.PY — Feature Engineering")
    print("=" * 55)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    print(f"\n[1/8] Loading clean CSV from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)
    print(f"  Loaded: {len(df):,} rows · {df.shape[1]} columns")

    print("\n[2/8] Adding temporal features...")
    df = add_temporal_features(df)

    print("\n[3/8] Adding lag features...")
    df = add_lag_features(df)

    print("\n[4/8] Adding rolling features...")
    df = add_rolling_features(df)

    print("\n[5/8] Adding cyclic time features...")
    df = add_cyclic_time_features(df)

    print("\n[6/8] Engineering rain_tomorrow target...")
    df = add_rain_tomorrow(df)

    print("\n[7/8] Dropping NaN rows from window edges...")
    df = drop_nan_rows(df)

    print("\n[8/8] Validating engineered features...")
    validate_features(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print_summary(df)

    print(f"\n{'=' * 55}")
    print(f"  ✅ Saved → {OUTPUT_PATH}")
    print(f"{'=' * 55}")

    return df


if __name__ == "__main__":
    main()
