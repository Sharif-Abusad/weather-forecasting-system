"""
Weather API Routes
────────────────────────────────────────────────────────────
Handles:
1. Weather API endpoints
2. Current weather retrieval
3. Rain prediction requests
4. Temperature forecasting responses
5. Weather dashboard data aggregation

Project   : Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Library Imports
# ──────────────────────────────────────────────────────────
from datetime import datetime, timedelta

import pytz
from fastapi import APIRouter, HTTPException

from app.services.ml_services import (
    predict_rain,
    predict_temperature
)
from app.services.weather_service import fetch_current_weather
from app.services.forecast_service import get_forecast
from app.utils.weather_utils import validate_and_geocode

router = APIRouter()


# ──────────────────────────────────────────────────────────
# Weather Endpoint
# ──────────────────────────────────────────────────────────
@router.get("/weather/{city}")
def weather(city: str):
    """
    Complete Weather Forecast Endpoint

    Features:
    - Validates city using Open-Meteo Geocoding API
    - Fetches current weather conditions
    - Generates next-hour temperature forecast using LSTM
    - Predicts tomorrow rain probability using XGBoost
    - Returns 5-day weather forecast
    """

    try:
        # ──────────────────────────────────────────────────────
        # 1. Validate city and get coordinates
        # ──────────────────────────────────────────────────────
        latitude, longitude, display_name = validate_and_geocode(city)

        # ──────────────────────────────────────────────────────
        # 2. Fetch current weather
        # ──────────────────────────────────────────────────────
        weather_data = fetch_current_weather(latitude, longitude)

        current_weather = weather_data["current_weather"]
        current_weather["city"] = display_name

        # ──────────────────────────────────────────────────────
        # 3. Predict next temperatures using LSTM
        # ──────────────────────────────────────────────────────
        lstm_forecast = predict_temperature(weather_data["hourly_df"])

        # ──────────────────────────────────────────────────────
        # 4. Fetch 5-day forecast
        # ──────────────────────────────────────────────────────
        daily_forecast = get_forecast(latitude, longitude)

        # ──────────────────────────────────────────────────────
        # 5. Predict rain probability
        # ──────────────────────────────────────────────────────
        rain_prediction = predict_rain(weather_data["xgboost_features"])


        # ──────────────────────────────────────────────────────
        # 6. Build hourly forecast strip for frontend
        # ──────────────────────────────────────────────────────
        timezone = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(timezone)

        hourly_forecast = [
            {
                "time": (current_time + timedelta(hours=i + 1)).strftime("%H:00"),
                "temp": round(item["temperature_C"]),
                "humidity": None,
            }
            for i, item in enumerate(lstm_forecast["forecast"])
        ]

        # ──────────────────────────────────────────────────────
        # 7. Final API Response
        # ──────────────────────────────────────────────────────
        return {
            "current": current_weather,
            "hourly": hourly_forecast,
            "forecast": daily_forecast,
            "rain_prediction": rain_prediction,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )