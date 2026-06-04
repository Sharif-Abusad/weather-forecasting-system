<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0%2B-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/ERA5-ECMWF-0078D4?style=for-the-badge&logo=data&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

# 🌦️ Weather Forecasting System
### Machine Learning & Deep Learning · Azamgarh, Uttar Pradesh, India

*Predict tomorrow's rain and forecast hourly temperature using ERA5 reanalysis data,*
*XGBoost classification, and Bidirectional LSTM neural networks — served via a Flask REST API.*

<br/>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-classifier-orange)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)](https://flask.palletsprojects.com)
[![Open-Meteo](https://img.shields.io/badge/Open--Meteo-live%20data-0ea5e9)](https://open-meteo.com)
[![ERA5](https://img.shields.io/badge/ERA5-reanalysis-0078D4)](https://cds.climate.copernicus.eu)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?logo=black)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Data Source](#-data-source)
- [Models](#-models)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Pipeline Overview](#-pipeline-overview)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)

---

## 🌐 Overview

This project is an **end-to-end weather forecasting system** built specifically for **Azamgarh, Uttar Pradesh, India (26.04°N, 83.11°E)**. It combines ERA5 reanalysis data from ECMWF with state-of-the-art machine learning models to provide two types of forecasts:

| Forecast | Model | Task | Output |
|----------|-------|------|--------|
| 🌧️ **Rain Tomorrow** | XGBoost Classifier | Binary Classification | Yes / No + Probability |
| 🌡️ **Temperature** | Bidirectional LSTM | Multi-Step Regression | Next N hours in °C |

Live predictions are powered by the **Open-Meteo API** — an ERA5-compatible free weather API — which provides real-time weather data including the past 24 hours required for lag feature computation.

> **Why ERA5?** ERA5 is the gold standard reanalysis dataset from ECMWF. Unlike raw weather station data, it is spatially consistent, gap-free, and provides cloud cover at multiple atmospheric layers — critical for accurate rain prediction.

---

## 🎬 Demo

```
$ python app/main.py

 * Running on http://localhost:5000

🌍  Location   : Azamgarh, UP (26.04°N, 83.11°E)
🌧️  Rain tomorrow   : Yes (probability: 0.73, confidence: high)
🌡️  Temperature forecast:
       +1h  →  28.4°C
       +2h  →  29.1°C
       +3h  →  29.8°C
       +4h  →  29.3°C
       +5h  →  28.7°C
```

---

## ✨ Features

- 📥 **Automated ERA5 data pipeline** — monthly downloads via Copernicus CDS API
- 🔧 **Feature engineering** — lag features, rolling averages, cyclical encodings
- 📊 **Comprehensive EDA** — 17 publication-quality figures across 11 analysis sections
- 🌲 **XGBoost classifier** — time-series CV, class imbalance handling, threshold tuning
- 🧠 **BiLSTM encoder-decoder** — configurable N-step temperature forecasting
- 🌐 **Flask REST API** — clean endpoints with JSON responses
- 📡 **Open-Meteo integration** — live inference without a database or cron job
- 🧪 **Calibrated predictions** — probability outputs with confidence levels
- 📦 **Fully reproducible** — one `.env` file controls the entire system

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA TIER     │───▶│   MODEL TIER    │───▶│  SERVING TIER   │───▶│   USER TIER     │
│                 │    │                 │    │                 │    │                 │
│ ERA5 CDS API    │    │ XGBoost         │    │ Flask REST API  │    │ Web Browser     │
│ 12 monthly ZIPs │    │ rain_model.json │    │ /predict/rain   │    │ Rain: Yes/No    │
│ data/processed/ │    │ BiLSTM keras    │    │ /predict/temp   │    │ Temp: +1h..+5h  │
│ final.csv       │    │ scalers/.pkl    │    │ Open-Meteo live │    │ index.html      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

Training pipeline (one-time):
download_weather.py → preprocess.py → feature_engineer.py → train_xgboost.py / train_lstm.py

Live inference (every request):
Open-Meteo API → fetch_weather.py → predictor.py → Flask → Browser
```

---

## 📁 Project Structure

```
weather-forecasting-system/
│
├── app/                   
│   ├── routes/
│   │   └── weather_routes.py       # API route definitions
│   ├── services/
│   │   ├── forecast_service.py     # Forecast business logic
│   │   ├── ml_services.py          # ML inference service
│   │   ├── model_loader.py         # Model loading & caching
│   │   └── weather_service.py      # Weather data service
│   └── utils/
│       └── weather_utils.py        # Utility/helper functions
│
├── config.py                       # App configuration (env, paths, settings)
├── main.py                         # Application entry point
│
├── data/
│   ├── archives/                   # Monthly zipped raw data archives
│   │   ├── azamgarh_weather_APR_2025.zip
│   │   ├── azamgarh_weather_AUG_2025.zip
│   │   ├── azamgarh_weather_DEC_2025.zip
│   │   ├── azamgarh_weather_FEB_2025.zip
│   │   ├── azamgarh_weather_JAN_2025.zip
│   │   ├── azamgarh_weather_JUL_2025.zip
│   │   ├── azamgarh_weather_JUN_2025.zip
│   │   ├── azamgarh_weather_MAR_2025.zip
│   │   ├── azamgarh_weather_MAY_2025.zip
│   │   ├── azamgarh_weather_NOV_2025.zip
│   │   ├── azamgarh_weather_OCT_2025.zip
│   │   └── azamgarh_weather_SEP_2025.zip
│   │
│   ├── processed/                  # Cleaned and feature-engineered CSVs
│   │   ├── azamgarh_weather_clean.csv
│   │   ├── azamgarh_weather_final.csv
│   │   └── azamgarh_weather_raw.csv
│   │
│   └── raw/                        # Raw ERA5 data streams (by month)
│       ├── APR/
│       │   ├── data_stream-oper_stepType-a...
│       │   └── data_stream-oper_stepType-i...
│       ├── AUG/
│       ├── DEC/
│       ├── FEB/
│       ├── JAN/
│       ├── JUL/
│       ├── JUN/
│       ├── MAR/
│       ├── MAY/
│       ├── NOV/
│       ├── OCT/
│       └── SEP/
│
├── models/
│   ├── lstm/                       # LSTM temperature model
│   │   ├── best_model.keras
│   │   ├── lstm_temp_model.keras   # Trained BiLSTM model
│   │   └── model_meta.json              
│   ├── scalers/
│   │   └── lstm_temp_scaler.pkl    # Fitted scalers for normalization
│   └── xgboost/                    # XGBoost rainfall model
│       ├── model_meta.json
│       └── rain_model.json
│
├── notebooks/                          # Jupyter notebooks for EDA & training
│   ├── eda_figures/                    # Saved EDA plots
│   ├── 01_era5_eda.ipynb               # ERA5 exploratory data analysis
│   ├── 03_xgboost_train.ipynb          # XGBoost model training
│   ├── 04_lstm_train.ipynb             # LSTM model training
│   ├── download_weather.py             # Script to download weather data
│   └── weather_dataset_pipeline.ipynb
│
├── pipeline/                             # End-to-end ML pipeline scripts
│   ├── 01_download_weather.py            # Step 1: Download ERA5 data
│   ├── 02_weather_dataset_pipeline.ipynb # Step 2: Extract Zip files and merge it
│   ├── 03_preprocess.py                  # Step 3: Clean & preprocess data
│   ├── 04_feature_engineer.py            # Step 4: Feature engineering
│   ├── 05_train_xgboost.py               # Step 5: Train XGBoost model
│   └── 06_train_lstm.py                  # Step 6: Train LSTM model
│
├── static/
│   ├── script.js                   # Frontend JavaScript
│   └── style.css                   # Frontend styles
│
├── templates/
│   └── index.html                  # Main HTML template
│
├── .env                            # Environment variables (not committed)
├── .env.example                    # Keep the example so others can set up the project
├── .gitignore     
├── requirements.txt                # Python dependencies
└── README.md
```

---
## 📊 Data Source

Weather data is sourced from **ERA5 (ECMWF Reanalysis v5)** via the **Copernicus Climate Data Store (CDS)** for Azamgarh, Uttar Pradesh, India.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | ERA5 Reanalysis (ECMWF) via Copernicus CDS |
| **Location** | Azamgarh, UP, India (26.04°N, 83.11°E) |
| **Period** | January 2025 – December 2025 |
| **Frequency** | Hourly |
| **Total rows** | ~8,700 |
| **Features** | 18 columns (12 numeric + 5 temporal + 1 target) |
| **Missing values** | None |
| **Download format** | NetCDF → CSV |

### Feature Descriptions

| Column | Unit | Description |
|--------|------|-------------|
| `valid_time` | DateTime | Hourly timestamp (IST, UTC+5:30) |
| `temperature` | °C | Air temperature at 2m above surface |
| `surface_pressure` | hPa | Atmospheric pressure at surface |
| `total_cloud_cover` | 0–1 | Total cloud fraction |
| `low_cloud_cover` | 0–1 | Cloud fraction below 2 km |
| `medium_cloud_cover` | 0–1 | Cloud fraction 2–6 km |
| `high_cloud_cover` | 0–1 | Cloud fraction above 6 km |
| `precipitation` | mm | Hourly accumulated precipitation |
| `humidity` | % | Relative humidity at 2m |
| `wind_speed` | m/s | Wind speed at 10m |
| `hour` | 0–23 | Hour of day |
| `month` | 1–12 | Month of year |
| `temp_rolling_6` | °C | 6-hour rolling mean temperature |
| `temp_lag_1` | °C | Temperature 1 hour ago |
| `temp_lag_24` | °C | Temperature 24 hours ago |
| `rain_tomorrow` | 0/1 | **Target**: rain in next 24h > 0.1mm |

---

## 🤖 Models

### XGBoost Rain Classifier

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | logloss |
| `early_stopping_rounds` | 20 |
| Class imbalance | `sample_weight='balanced'` |
| Validation | 5-fold TimeSeriesSplit |

### Bidirectional LSTM

| Parameter | Value |
|-----------|-------|
| Architecture | Encoder-Decoder |
| Encoder | BiLSTM (64) → Dropout → LSTM (32) → BatchNorm |
| Bridge | RepeatVector (N) |
| Decoder | LSTM (32) → Dropout → BatchNorm → TimeDistributed Dense |
| Loss | Huber |
| Optimiser | Adam (lr=1e-3) |
| Lookback | 24 hours (configurable via `LOOKBACK`) |
| Forecast horizon | 5 hours (configurable via `FORECAST_N`) |
| Trainable params | ~66,593 |

---

## ✅ Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | |
| pip | 23.0+ | |
| CDS Account | Free | [Register here](https://cds.climate.copernicus.eu) |
| RAM | 8 GB+ | For LSTM training |
| GPU | Optional | CUDA 11.8+ for faster LSTM training |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/weather-forecasting-system.git
cd weather-forecasting-system
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your CDS API key and other config
```

### 5. Set up CDS API credentials

Create `~/.cdsapirc` in your home directory:

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR-UID:YOUR-API-KEY
```
### 6. Run the full pipeline (or skip if models already trained)
```bash
python pipeline/01_download_weather.py
python pipeline/03_preprocess.py
python pipeline/04_feature_engineer.py
python pipeline/05_train_xgboost.py
python pipeline/06_train_lstm.py
```
### 7. Start the FastAPI app
```bash
uvicorn app.main:app.py
```

Open `http://localhost:5000` in your browser.

> Get your UID and API key from your [CDS profile page](https://cds.climate.copernicus.eu/user).

## ⚙️ Pipeline Overview

The pipeline runs in sequential steps:

```
Download ERA5 Data  →  Build Dataset  →  Preprocess  →  Feature Engineer  →  Train XGBoost  →  Train LSTM  →  Serve via FastAPI
     01                    02               03               04                    05               06
```

## 📡 API Reference

### `POST /predict/rain`

Fetches live weather from Open-Meteo and returns rain prediction.

**Request:** No body required (fetches live data automatically)

**Response:**
```json
{
  "rain_tomorrow": true,
  "probability": 0.7341,
  "confidence": "high (rain likely)",
  "threshold_used": 0.38
}
```

---

### `POST /predict/temp`

Returns temperature forecast for the next N hours.

**Request body (optional):**
```json
{ "n_hours": 5 }
```

**Response:**
```json
{
  "forecast": [
    { "hour": "+1h", "temperature_C": 28.4 },
    { "hour": "+2h", "temperature_C": 29.1 },
    { "hour": "+3h", "temperature_C": 29.8 },
    { "hour": "+4h", "temperature_C": 29.3 },
    { "hour": "+5h", "temperature_C": 28.7 }
  ],
  "n_hours": 5,
  "mae_celsius": 0.85
}
```

---

### `GET /health`

```json
{ "status": "ok", "models_loaded": true }
```

---

## 📈 Results

> Fill in your actual values after running the notebooks.

### XGBoost — Rain Prediction

| Metric | Value |
|--------|-------|
| ROC-AUC | `[fill from notebook]` |
| F1-Score | `[fill from notebook]` |
| Accuracy | `[fill from notebook]` |
| Avg Precision | `[fill from notebook]` |
| Best Threshold | `[fill from notebook]` |
| CV ROC-AUC (mean ± std) | `[fill from notebook]` |

### LSTM — Temperature Forecasting

| Horizon | MAE (°C) | RMSE (°C) | R² |
|---------|----------|-----------|-----|
| +1h | `[fill]` | `[fill]` | `[fill]` |
| +2h | `[fill]` | `[fill]` | `[fill]` |
| +3h | `[fill]` | `[fill]` | `[fill]` |
| +4h | `[fill]` | `[fill]` | `[fill]` |
| +5h | `[fill]` | `[fill]` | `[fill]` |
| **Overall** | `[fill]` | `[fill]` | `[fill]` |

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.10+ |
| ML Model | XGBoost | 2.0+ |
| DL Framework | TensorFlow / Keras | 2.13+ |
| ML Utilities | Scikit-learn | 1.3+ |
| Data | Pandas, NumPy | 2.0+, 1.24+ |
| Visualisation | Matplotlib, Seaborn | 3.7+, 0.12+ |
| Web Framework | Flask | 3.0+ |
| Data Source (train) | ERA5 via cdsapi | 0.6+ |
| Data Source (live) | Open-Meteo | Free API |
| Serialisation | Joblib | 1.3+ |
| Environment | python-dotenv | 1.0+ |

---

## 🗺️ Roadmap

- [x] ERA5 data pipeline (monthly downloads)
- [x] Feature engineering (lags, rolling, target)
- [x] EDA notebook (17 figures)
- [x] XGBoost classifier with threshold tuning
- [x] BiLSTM encoder-decoder temperature forecasting
- [x] Flask REST API
- [x] Open-Meteo live integration
- [ ] Add humidity to LSTM feature set
- [ ] Sin/cos cyclical encoding for hour and month
- [ ] Extend LSTM forecast horizon to 24 hours
- [ ] Deploy to cloud (Heroku / Railway / AWS)
- [ ] Add automated daily retraining pipeline
- [ ] Add precipitation amount regression model
- [ ] Compare with Transformer-based model (PatchTST)
- [ ] Multi-location support across eastern UP

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: your feature description'`
4. Push to branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 🙏 Acknowledgements

- **ECMWF / Copernicus** — for making ERA5 reanalysis data freely available via the [Climate Data Store](https://cds.climate.copernicus.eu)
- **Open-Meteo** — for the excellent free weather API that shares ERA5 data lineage
- **XGBoost team** — Chen & Guestrin (2016) for the XGBoost algorithm
- **TensorFlow / Keras team** — for the deep learning framework
- **Scikit-learn team** — for TimeSeriesSplit and evaluation utilities

---

<div align="center">

**Made with ❤️ for Azamgarh, UP, India**

⭐ If this project helped you, please give it a star!

</div>
