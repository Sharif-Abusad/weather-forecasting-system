"""
Forecast Service Layer
────────────────────────────────────────────────────────────
Handles:
1. 5-day weather forecast retrieval
2. Open-Meteo API integration
3. Weather code to condition mapping
4. Forecast response formatting
5. Daily weather data transformation

Data Source : Open-Meteo API
Project     : Weather Forecasting System
"""


# ──────────────────────────────────────────────────────────
# Library Imports
# ──────────────────────────────────────────────────────────
import requests
from datetime import datetime
from fastapi import HTTPException
from app.config import OPEN_METEO_URL, TIMEZONE, FORECAST_DAYS
from app.utils.weather_utils import WEATHER_MAP

def get_forecast(latitude: float, longitude: float) -> list[dict]:
    """
    Fetches 5-day weather forecast data from Open-Meteo API.

    Args:
        latitude  (float): Geographic latitude of the city
        longitude (float): Geographic longitude of the city

    Returns:
        list[dict]:
            List containing daily forecast information such as:
            - date
            - min/max temperature
            - weather description
            - icon code
            - wind speed
    """

    # ──────────────────────────────────────────────────────────
    # Build API request parameters
    # ──────────────────────────────────────────────────────────
    params = {
        "latitude": latitude,
        "longitude": longitude,

        # Daily weather variables required from Open-Meteo
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "weathercode",
            "windspeed_10m_max",
        ]),

        # Local timezone for accurate forecast timing
        "timezone": TIMEZONE,

        # Number of forecast days to fetch
        "forecast_days": FORECAST_DAYS,
    }

    # ──────────────────────────────────────────────────────────
    # Make API request
    # ──────────────────────────────────────────────────────────
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=10
        )

        # Raise exception for HTTP errors
        response.raise_for_status()

    except requests.RequestException as e:

        # Convert request errors into FastAPI HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Forecast service unavailable: {str(e)}"
        )

    # ──────────────────────────────────────────────────────────
    # Parse JSON response
    # ──────────────────────────────────────────────────────────
    data = response.json()

    # Extract daily forecast section safely
    daily_data = data.get("daily", {})

    # ──────────────────────────────────────────────────────────
    # Extract forecast arrays
    # Each index corresponds to one forecast day
    # ──────────────────────────────────────────────────────────
    dates = daily_data.get("time", [])
    temp_min = daily_data.get("temperature_2m_min", [])
    temp_max = daily_data.get("temperature_2m_max", [])
    weather_codes = daily_data.get("weathercode", [])
    wind_speeds = daily_data.get("windspeed_10m_max", [])

    # Final forecast list
    forecasts = []

    # ──────────────────────────────────────────────────────────
    # Build structured forecast objects
    # ──────────────────────────────────────────────────────────
    for i in range(len(dates)):

        # Get weather description + icon from weather code
        weather_code = weather_codes[i]

        description, icon = WEATHER_MAP.get(
            weather_code,
            ("Unknown", "01d")
        )

        # Format date into readable format
        formatted_date = datetime.strptime(
            dates[i],
            "%Y-%m-%d"
        ).strftime("%a, %b %d")

        # Append forecast entry
        forecasts.append({

            # Example: Mon, May 26
            "date": formatted_date,

            # Static midday time for UI display
            "time": "12:00",

            # Daily minimum temperature
            "temp_min": round(temp_min[i]),

            # Daily maximum temperature
            "temp_max": round(temp_max[i]),

            # Weather condition text
            "description": description,

            # OpenWeather-style icon code used in frontend
            "icon": icon,

            # Placeholder (can be added later from API)
            "humidity": None,

            # Maximum wind speed for the day
            "wind_speed": round(wind_speeds[i]),
        })

    # ──────────────────────────────────────────────────────────
    # Return final formatted forecast list
    # ──────────────────────────────────────────────────────────
    return forecasts
