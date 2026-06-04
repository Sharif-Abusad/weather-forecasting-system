"""
Utility Functions
────────────────────────────────────────────────────────────
Provides:

1. Open-Meteo weather code mappings
2. City validation and geocoding
3. Relative humidity calculations

Used by:
- Forecast service
- Weather data service
- API routes

Project: Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Library Imports
# ──────────────────────────────────────────────────────────
import requests
from fastapi import HTTPException
from typing import Tuple
import numpy as np
from app.config import GEOCODE_URL

# ──────────────────────────────────────────────────────────
# Open-Meteo Weather Code Mapping
# Maps weather codes to:
#   • Human-readable descriptions
#   • Frontend weather icon identifiers
# Reference:
# https://open-meteo.com/en/docs
# ──────────────────────────────────────────────────────────
WEATHER_MAP = {
    0: ("Clear sky", "01d"),
    1: ("Mainly clear", "02d"),
    2: ("Partly cloudy", "03d"),
    3: ("Overcast", "04d"),

    45: ("Fog", "50d"),
    48: ("Depositing rime fog", "50d"),

    51: ("Light drizzle", "09d"),
    53: ("Moderate drizzle", "09d"),
    55: ("Dense drizzle", "09d"),

    61: ("Slight rain", "10d"),
    63: ("Moderate rain", "10d"),
    65: ("Heavy rain", "10d"),

    71: ("Slight snow", "13d"),
    73: ("Moderate snow", "13d"),
    75: ("Heavy snow", "13d"),

    80: ("Rain showers", "09d"),

    95: ("Thunderstorm", "11d"),
}

# ──────────────────────────────────────────────────────────
# City Validation & Geocoding
# Uses Open-Meteo Geocoding API to convert
# city names into latitude and longitude.
# ──────────────────────────────────────────────────────────
def validate_and_geocode(city: str) -> Tuple[float, float, str]:
    """
    Validate a city name using the Open-Meteo Geocoding API.

    Args:
        city (str): Name of the city entered by the user.

    Returns:
        Tuple[float, float, str]:
            - latitude
            - longitude
            - formatted display name

    Raises:
        HTTPException:
            400 -> Empty city name
            404 -> City not found
            503 -> Geocoding service unavailable
    """

    # ──────────────────────────────────────────────────────────
    # Validate input
    # ──────────────────────────────────────────────────────────
    city = city.strip()

    if not city:
        raise HTTPException(
            status_code=400,
            detail="City name cannot be empty."
        )

    # ──────────────────────────────────────────────────────────
    # API Request
    # ──────────────────────────────────────────────────────────
    try:
        response = requests.get(
            GEOCODE_URL,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=5,
        )

        response.raise_for_status()
        data = response.json()

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Geocoding service timed out. Please try again."
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Geocoding service unavailable. Please try again later."
        )

    # ──────────────────────────────────────────────────────────
    # Validate response
    # ──────────────────────────────────────────────────────────
    results = data.get("results")

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found. Please check the spelling and try again."
        )

    # ──────────────────────────────────────────────────────────
    # Extract top match
    # ──────────────────────────────────────────────────────────
    top_result = results[0]

    latitude = top_result["latitude"]
    longitude = top_result["longitude"]

    city_name = top_result.get("name", city)
    state     = top_result.get("admin1")
    country   = top_result.get("country")

    # ──────────────────────────────────────────────────────────
    # Build display name
    # ──────────────────────────────────────────────────────────
    location_parts = [city_name]

    if state:
        location_parts.append(state)

    if country:
        location_parts.append(country)

    display_name = ", ".join(location_parts)

    return latitude, longitude, display_name

# ──────────────────────────────────────────────────────────
# Humidity Calculation
# Computes relative humidity (%) from
# temperature and dewpoint temperature.
# Supports scalars, NumPy arrays,
# and pandas Series.
# ──────────────────────────────────────────────────────────
def calculate_humidity(temp, dewpoint):
    """
    Calculate relative humidity using temperature and dewpoint.

    Works with:
    - single float values
    - NumPy arrays
    - pandas Series
    """

    humidity = 100 * (
        np.exp((17.625 * dewpoint) / (243.04 + dewpoint))
        /
        np.exp((17.625 * temp) / (243.04 + temp))
    )

    # If scalar → return float
    if np.isscalar(humidity):
        return round(float(humidity), 2)

    # If array/series → return rounded array
    return np.round(humidity, 2)