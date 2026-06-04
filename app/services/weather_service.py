"""
Current Weather Data Service
────────────────────────────────────────────────────────────
Handles:

1. Live weather retrieval from Open-Meteo API
2. Feature engineering for XGBoost rain prediction
3. Feature engineering for LSTM temperature forecasting
4. Lag feature generation (1h, 24h)
5. Rolling temperature statistics
6. Humidity calculation from temperature and dewpoint
7. Cyclical time feature encoding (hour/month)
8. Historical weather sequence preparation for LSTM input
9. Current weather response formatting

Returns:
- Current weather conditions
- LSTM model features
- XGBoost model features
- Hourly weather dataframe
- Metadata and fetch timestamp

Data Source : Open-Meteo API
Project     : Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Library Imports
# ──────────────────────────────────────────────────────────
import math
import requests
from datetime import datetime

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.config import LAT, LON, TIMEZONE, LOOKBACK, OPEN_METEO_URL
from app.utils.weather_utils import calculate_humidity

# ── Open-Meteo request parameters ────────────────────────────────────────────
def _build_params(latitude, longitude) -> dict:
    return {
        "latitude"        : latitude,
        "longitude"       : longitude,
        "timezone"        : TIMEZONE,
        "past_hours"      : LOOKBACK*2,      # 24h history for lag features
        "forecast_hours"  : 1,
        "current"         : ",".join([
            "temperature_2m",
            "apparent_temperature",
            "surface_pressure",
            "precipitation",
            "dewpoint_2m",
            "wind_speed_10m",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
        ]),

        "hourly": ",".join([
            "temperature_2m",              # needed for lag_1, lag_24, rolling_6
            "dewpoint_2m",                 # needed for hourly humidity history
        ]),

        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
        ])
    }

# ── Cyclical encoding helpers ─────────────────────────────────────────────────
def _hour_sin(hour: int) -> float:
    return round(math.sin(2 * math.pi * hour / 24), 6)

def _hour_cos(hour: int) -> float:
    return round(math.cos(2 * math.pi * hour / 24), 6)

def _month_sin(month: int) -> float:
    return round(math.sin(2 * math.pi * month / 12), 6)

def _month_cos(month: int) -> float:
    return round(math.cos(2 * math.pi * month / 12), 6)


# ── Main fetch function ───────────────────────────────────────────────────────
def fetch_current_weather(latitude, longitude) -> dict:
    """
    Fetches live weather from Open-Meteo and returns TWO feature dicts:
      - lstm_features   : for predict_temperature()
      - xgboost_features: for predict_rain()

    Also returns the full hourly DataFrame so predictor.py can
    build LSTM sequences (24h × n_features).

    Returns:
        {
            "lstm_features"    : dict,
            "xgboost_features" : dict,
            "hourly_df"        : pd.DataFrame,   # 24 rows × LSTM FEATURES
            "fetched_at"       : str,             # ISO timestamp
        }
    """
    # ── API call ──────────────────────────────────────────────────────────────
    response = requests.get(OPEN_METEO_URL, params=_build_params(latitude, longitude), timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    hourly  = data["hourly"]
    dewpoint = current["dewpoint_2m"]

    # ── Parse hourly history into DataFrame ───────────────────────────────────
    hourly_df = pd.DataFrame({
        "time"        : pd.to_datetime(hourly["time"]),
        "temperature" : hourly["temperature_2m"],
        "humidity"    : hourly["dewpoint_2m"]  # calculate_humidity(hourly["temperature_2m"], hourly["dewpoint_2m"]),
    }).dropna().sort_values("time").reset_index(drop=True)

    # ── Current timestamp ─────────────────────────────────────────────────────
    now   = datetime.now()
    hour  = now.hour
    month = now.month

    # Daily min/max temperature
    temp_min = data["daily"]["temperature_2m_min"][0]

    temp_max = data["daily"]["temperature_2m_max"][0]

    # ── Current values (direct from API) ─────────────────────────────────────
    temp_now      = current["temperature_2m"]           # °C ✅
    pressure      = current["surface_pressure"]          # hPa ✅
    precipitation = current.get("precipitation", 0.0)    # mm ✅
    wind_speed    = current["wind_speed_10m"]             # m/s ✅

    # Cloud cover: Open-Meteo gives % → divide by 100 for ERA5 0-1 fraction
    total_cloud   = round(current["cloud_cover"]     / 100, 4)
    low_cloud     = round(current["cloud_cover_low"] / 100, 4)
    mid_cloud     = round(current["cloud_cover_mid"] / 100, 4)
    high_cloud    = round(current["cloud_cover_high"]/ 100, 4)
    humidity = calculate_humidity(
        temp=temp_now,
        dewpoint=dewpoint
    )
    # ── Lag features (from hourly history) ───────────────────────────────────
    temps = hourly_df["temperature"].values

    # temp_lag_1  = temperature 1 hour ago = second-to-last hourly value
    temp_lag_1  = float(temps[-2]) if len(temps) >= 2  else temp_now

    # temp_lag_24 = temperature 24 hours ago = value at index -25
    # (past_hours=24 gives 25 values including current hour)
    temp_lag_24 = float(temps[-25]) if len(temps) >= 25 else temp_now

    # temp_rolling_6 = mean of 6 values before current hour (indices -7 to -2)
    if len(temps) >= 7:
        temp_rolling_6 = round(float(temps[-7:-1].mean()), 4)
    else:
        temp_rolling_6 = round(float(temps[:-1].mean()), 4) if len(temps) > 1 else temp_now

    current_weather = {
        # ── Temperature ─────────────────────────
        "temperature": round(temp_now, 2),
        "feels_like": round(
            current["apparent_temperature"],
            2
        ),
        
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),

        "temp_lag_1"       : round(temp_lag_1,      2),
        "temp_lag_24"      : round(temp_lag_24,     2),
        "temp_rolling_6"   : round(temp_rolling_6,  4),

        "humidity"         : round(humidity,         2),
        "surface_pressure" : round(pressure,         2),
        "total_cloud_cover": total_cloud,
        "wind_speed"       : round(wind_speed,       4),

        # ── Cloud Features ──────────────────────

        "low_cloud_cover": low_cloud,
        "medium_cloud_cover": mid_cloud,
        "high_cloud_cover": high_cloud,
        "precipitation": precipitation,
    }
    # ── Cyclical time features ────────────────────────────────────────────────
    # These replace raw month/hour — encode periodic nature correctly
    h_sin  = _hour_sin(hour)
    h_cos  = _hour_cos(hour)
    m_sin  = _month_sin(month)
    m_cos  = _month_cos(month)

    # ── Build LSTM feature dict ───────────────────────────────────────────────
    # Must match FEATURES list in train_lstm.py EXACTLY (same order)
    lstm_features = {
        "temperature"      : round(temp_now,       2),
        "temp_lag_1"       : round(temp_lag_1,      2),
        "temp_lag_24"      : round(temp_lag_24,     2),
        "temp_rolling_6"   : round(temp_rolling_6,  4),
        "humidity"         : round(humidity,         2),
        "surface_pressure" : round(pressure,         2),
        "total_cloud_cover": total_cloud,
        "wind_speed"       : round(wind_speed,       4),
        "hour_sin"         : h_sin,
        "hour_cos"         : h_cos,
        "month_sin"        : m_sin,
        "month_cos"        : m_cos,
    }

    # ── Build XGBoost feature dict ────────────────────────────────────────────
    xgboost_features = {
        "temperature"        : round(temp_now,       2),
        "surface_pressure"   : round(pressure,        2),
        "total_cloud_cover"  : total_cloud,
        "low_cloud_cover"    : low_cloud,
        "medium_cloud_cover" : mid_cloud,
        "high_cloud_cover"   : high_cloud,
        "precipitation"      : round(precipitation,   4),
        "humidity"           : round(humidity,         2),
        "wind_speed"         : round(wind_speed,       4),
        "temp_rolling_6"     : round(temp_rolling_6,   4),
        "temp_lag_1"         : round(temp_lag_1,       2),
        "temp_lag_24"        : round(temp_lag_24,      2),
        "month"              : month,
    }

    # ── Build hourly DataFrame for LSTM sequence input ────────────────────────

    hourly_df["temp_lag_1"]       = hourly_df["temperature"].shift(1)
    hourly_df["temp_lag_24"]      = hourly_df["temperature"].shift(24)
    hourly_df["temp_rolling_6"]   = hourly_df["temperature"].rolling(6).mean()
    hourly_df["total_cloud_cover"]= total_cloud   # only current is available hourly
    hourly_df["surface_pressure"] = pressure      # only current is available hourly
    hourly_df["wind_speed"]       = wind_speed    # only current is available hourly
    hourly_df["hour"]             = hourly_df["time"].dt.hour
    hourly_df["month"]            = hourly_df["time"].dt.month
    hourly_df["hour_sin"]         = hourly_df["hour"].apply(_hour_sin)
    hourly_df["hour_cos"]         = hourly_df["hour"].apply(_hour_cos)
    hourly_df["month_sin"]        = hourly_df["month"].apply(_month_sin)
    hourly_df["month_cos"]        = hourly_df["month"].apply(_month_cos)

    # Drop rows with NaN from lag computation
    hourly_df = hourly_df.dropna().reset_index(drop=True)

    return {
        "current_weather"  : current_weather,
        "lstm_features"    : lstm_features,
        "xgboost_features" : xgboost_features,
        "hourly_df"        : hourly_df.tail(24),
        "fetched_at"       : now.isoformat(),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching weather for Azamgarh...")
    result = fetch_current_weather(latitude=LAT, longitude=LON)

    print("\n── LSTM Features ──")
    for k, v in result["lstm_features"].items():
        print(f"  {k:<22} : {v}")

    print("\n── XGBoost Features ──")
    for k, v in result["xgboost_features"].items():
        print(f"  {k:<22} : {v}")

    print(f"\n── Hourly DataFrame ──")
    print(f"  Shape : {result['hourly_df'].shape}")
    print(result['hourly_df'])
    print(result['hourly_df'].columns)

    print(f"\nFetched at: {result['fetched_at']}")
