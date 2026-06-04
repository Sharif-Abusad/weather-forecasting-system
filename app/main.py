"""
Application Entry Point
────────────────────────────────────────────────────────────
Handles:
1. FastAPI application initialization
2. API route registration
3. Static file serving
4. Frontend (index.html) delivery
5. Model metadata endpoint

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
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────────────────────
# Internal Imports
# ──────────────────────────────────────────────────────────
from app.routes.weather_routes import router


# ──────────────────────────────────────────────────────────
# FastAPI Application Initialization
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Weather Forecast API"
)

# Register application routes
app.include_router(router)

# Serve CSS, JavaScript, images, etc.
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ──────────────────────────────────────────────────────────
# Frontend Route
# ──────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    """
    Serve the frontend dashboard.

    Returns:
        index.html file
    """
    return FileResponse("templates/index.html")


# ──────────────────────────────────────────────────────────
# Model Metadata Endpoint
# ──────────────────────────────────────────────────────────
@app.get("/model-meta")
def get_model_meta():
    """
    Return metadata for both trained models.

    Returns:
        {
            "lstm": {...},
            "xgboost": {...}
        }

    Used by the frontend to display:
    - LSTM architecture
    - Forecast metrics
    - Per-step MAE
    - XGBoost accuracy
    - ROC-AUC
    - Feature importance
    """

    # Metadata file locations
    lstm_path = Path("models/lstm/model_meta.json")
    xgb_path = Path("models/xgboost/model_meta.json")

    # Load LSTM metadata
    with open(lstm_path, "r", encoding="utf-8") as f:
        lstm_meta = json.load(f)

    # Load XGBoost metadata
    with open(xgb_path, "r", encoding="utf-8") as f:
        xgb_meta = json.load(f)

    # Return combined metadata response
    return {
        "lstm": lstm_meta,
        "xgboost": xgb_meta
    }