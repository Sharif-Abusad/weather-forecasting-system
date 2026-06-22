# """
# Application Entry Point
# ────────────────────────────────────────────────────────────
# Handles:
# 1. FastAPI application initialization
# 2. API route registration
# 3. Static file serving
# 4. Frontend (index.html) delivery
# 5. Model metadata endpoint

# Project: Weather Forecasting System
# """

# # ──────────────────────────────────────────────────────────
# # Standard Library Imports
# # ──────────────────────────────────────────────────────────
# import json
# from pathlib import Path

# # ──────────────────────────────────────────────────────────
# # Third-Party Imports
# # ──────────────────────────────────────────────────────────
# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles

# # ──────────────────────────────────────────────────────────
# # Internal Imports
# # ──────────────────────────────────────────────────────────
# from app.routes.weather_routes import router


# # ──────────────────────────────────────────────────────────
# # FastAPI Application Initialization
# # ──────────────────────────────────────────────────────────
# app = FastAPI(
#     title="Weather Forecast API"
# )

# # Register application routes
# app.include_router(router)

# # Serve CSS, JavaScript, images, etc.
# app.mount(
#     "/static",
#     StaticFiles(directory="static"),
#     name="static"
# )


# # ──────────────────────────────────────────────────────────
# # Frontend Route
# # ──────────────────────────────────────────────────────────
# @app.get("/")
# def serve_ui():
#     """
#     Serve the frontend dashboard.

#     Returns:
#         index.html file
#     """
#     return FileResponse("templates/index.html")


# # ──────────────────────────────────────────────────────────
# # Model Metadata Endpoint
# # ──────────────────────────────────────────────────────────
# @app.get("/model-meta")
# def get_model_meta():
#     """
#     Return metadata for both trained models.

#     Returns:
#         {
#             "lstm": {...},
#             "xgboost": {...}
#         }

#     Used by the frontend to display:
#     - LSTM architecture
#     - Forecast metrics
#     - Per-step MAE
#     - XGBoost accuracy
#     - ROC-AUC
#     - Feature importance
#     """

#     # Metadata file locations
#     lstm_path = Path("models/lstm/model_meta.json")
#     xgb_path = Path("models/xgboost/model_meta.json")

#     # Load LSTM metadata
#     with open(lstm_path, "r", encoding="utf-8") as f:
#         lstm_meta = json.load(f)

#     # Load XGBoost metadata
#     with open(xgb_path, "r", encoding="utf-8") as f:
#         xgb_meta = json.load(f)

#     # Return combined metadata response
#     return {
#         "lstm": lstm_meta,
#         "xgboost": xgb_meta
#     }
"""
Application Entry Point
────────────────────────────────────────────────────────────
Handles:
1. FastAPI application initialization with lifespan
2. CORS middleware (required for Vercel frontend → Render backend)
3. API route registration
4. Static file serving
5. Frontend (index.html) delivery
6. Model metadata endpoint
7. Health check endpoint

Project: Weather Forecasting System
"""

# ──────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────
import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

# ──────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────
# Internal Imports
# ──────────────────────────────────────────────────────────
from app.routes.weather_routes import router
from app.services.predictor import load_models


# ──────────────────────────────────────────────────────────
# Lifespan — runs on startup and shutdown
# ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load XGBoost and LSTM models into memory once.
    Shutdown: cleanup (if needed).

    Using lifespan instead of @app.on_event("startup") which
    is deprecated in newer FastAPI versions.
    """
    print("🚀 Starting Weather Forecasting API...")
    load_models()
    print("✅ Models loaded and ready")
    yield
    print("🛑 Shutting down...")


# ──────────────────────────────────────────────────────────
# FastAPI Application Initialization
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Weather Forecasting API",
    description = "XGBoost rain prediction + BiLSTM temperature forecasting for Azamgarh, UP",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",      # Swagger UI at /docs
    redoc_url   = "/redoc",     # ReDoc at /redoc
)


# ──────────────────────────────────────────────────────────
# CORS Middleware
# Required so Vercel frontend can call this Render API.
# Without this the browser blocks every request.
# ──────────────────────────────────────────────────────────
# Read allowed origins from .env so you don't hardcode URLs
# Example .env: ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5500
# _raw_origins = os.getenv(
#     "ALLOWED_ORIGINS",
#     "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000"
# )
# ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins     = ALLOWED_ORIGINS,
#     allow_credentials = True,
#     allow_methods     = ["*"],
#     allow_headers     = ["*"],
# )


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────
# Register prediction routes under /api prefix
app.include_router(router)

# Serve CSS, JavaScript, images etc.
# Only mount if static folder exists (avoids error on first run)
if Path("static").exists():
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static",
    )


# ──────────────────────────────────────────────────────────
# Health Check Endpoint
# ──────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """
    Health check for Render and UptimeRobot monitoring.

    Returns 200 OK when the API is running.
    UptimeRobot pings this every 14 min to prevent Render sleep.
    """
    return {
        "status"       : "ok",
        "models_loaded": True,
        "version"      : "1.0.0",
    }


# ──────────────────────────────────────────────────────────
# Frontend Route
# ──────────────────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
def serve_ui():
    """
    Serve the frontend dashboard.

    Note: When deploying frontend separately on Vercel,
    this endpoint is not used. It is only active when
    running the full app locally.
    """
    index_path = Path("templates/index.html")
    if not index_path.exists():
        return JSONResponse(
            status_code = 404,
            content     = {"detail": "index.html not found"}
        )
    return FileResponse(str(index_path))


# ──────────────────────────────────────────────────────────
# Model Metadata Endpoint
# ──────────────────────────────────────────────────────────
@app.get("/model-meta", tags=["Models"])
def get_model_meta():
    """
    Return metadata for both trained models.

    Used by frontend to dynamically display:
    - LSTM: architecture, forecast metrics, per-step MAE
    - XGBoost: accuracy, ROC-AUC, feature importance

    Returns:
        {
            "lstm"    : { ...lstm model_meta.json ... },
            "xgboost" : { ...xgboost model_meta.json ... }
        }
    """
    lstm_path = Path("models/lstm/model_meta.json")
    xgb_path  = Path("models/xgboost/model_meta.json")

    # Check files exist before opening
    if not lstm_path.exists():
        return JSONResponse(
            status_code = 503,
            content     = {"detail": "LSTM model_meta.json not found. Run train_lstm.py first."}
        )
    if not xgb_path.exists():
        return JSONResponse(
            status_code = 503,
            content     = {"detail": "XGBoost model_meta.json not found. Run train_xgboost.py first."}
        )

    with open(lstm_path,  "r", encoding="utf-8") as f:
        lstm_meta = json.load(f)

    with open(xgb_path,   "r", encoding="utf-8") as f:
        xgb_meta  = json.load(f)

    return {
        "lstm"    : lstm_meta,
        "xgboost" : xgb_meta,
    }


# ──────────────────────────────────────────────────────────
# Entry Point — local development only
# Production uses: uvicorn app.main:app --host 0.0.0.0 --port 8000
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))   # Render injects PORT automatically

    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = port,
        reload  = os.getenv("FLASK_DEBUG", "0") == "1",  # reload only in dev
    )