import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

# ── Location ──────────────────────────────────────────────────────
LAT       = float(os.getenv("LATITUDE",  26.00))
LON       = float(os.getenv("LONGITUDE", 83.00))
TIMEZONE  = os.getenv("TIMEZONE", "Asia/Kolkata")

# ── Model settings ────────────────────────────────────────────────
FORECAST_N = int(os.getenv("FORECAST_N", 5))          # how many hours ahead to predict
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", 5))    # how many days ahead to forecast
LOOKBACK   = int(os.getenv("LOOKBACK",  24))          # how many past hours LSTM reads

# ── Paths ─────────────────────────────────────────────────────────
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")


MODEL_LSTM_DIR    = os.getenv("MODEL_LSTM_DIR",    "models/lstm")

LSTM_MODEL_FILE   =  os.getenv("LSTM_MODEL_FILE",    "lstm_temp_model.keras")

MODEL_LSTM    = os.path.join(os.getenv("MODEL_LSTM_DIR",    "models/lstm"),
                              os.getenv("LSTM_MODEL_FILE",    "lstm_temp_model.keras"))

SCALER_DIR   = os.getenv("SCALER_DIR",    "models/scalers")

SCALER_FILE   = os.getenv("SCALER_FILE",  "lstm_temp_scaler.pkl")

SCALER_PATH   = os.path.join(os.getenv("SCALER_DIR",        "models/scalers"),
                              os.getenv("SCALER_FILE",        "lstm_temp_scaler.pkl"))

MODEL_XGBOOST_DIR = os.getenv("MODEL_XGBOOST_DIR", "models/xgboost")

XGBOOST_MODEL_FILE = os.getenv("XGBOOST_MODEL_FILE", "rain_model.json")

MODEL_XGBOOST = os.path.join(os.getenv("MODEL_XGBOOST_DIR", "models/xgboost"),
                              os.getenv("XGBOOST_MODEL_FILE", "rain_model.json"))

XGBOOST_META  = os.path.join(os.getenv("MODEL_XGBOOST_DIR", "models/xgboost"),
                              os.getenv("XGBOOST_META_FILE",  "model_meta.json"))

XGBOOST_META_FILE  = os.getenv("XGBOOST_META_FILE",  "model_meta.json")

LSTM_META     = os.path.join(os.getenv("MODEL_LSTM_DIR",    "models/lstm"),
                              os.getenv("LSTM_META_FILE",     "model_meta.json"))

LSTM_META_FILE     =   os.getenv("LSTM_META_FILE",     "model_meta.json")

# ── Open-Meteo ────────────────────────────────────────────────────
OPEN_METEO_URL = os.getenv("OPEN_METEO_BASE_URL",
                            "https://api.open-meteo.com/v1/forecast")

# ── Geocoding (city → latitude/longitude) ────────────────────────────────────────────────────
GEOCODE_URL = os.getenv("GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")

# ── FastAPI ─────────────────────────────────────────────────────────
PORT       = int(os.getenv("APP_PORT", 5000))
DEBUG      = os.getenv("DEBUG", "1") == "1"

