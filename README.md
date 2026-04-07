<div align="center">

# 🦠 Pandemic Intelligence System

### Multi-Modal COVID-19 Forecasting, Risk Modeling & Policy Impact Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-0174C1?style=flat-square)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](/.github/workflows/ci.yml)

<br/>

> **An end-to-end AI system for pandemic surveillance, forecasting, and decision support — designed for government agencies, healthcare organizations, and public health researchers.**

<br/>
🚀 Web Link Live : https://pandemic-intelligence.streamlit.app/

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Model Details](#-model-details)
- [API Reference](#-api-reference)
- [Dashboard](#-dashboard)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Interview Explanation](#-interview-explanation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

The **Pandemic Intelligence System** is a production-grade, research-quality AI platform that transforms raw COVID-19 epidemiological data into actionable intelligence. It combines classical time-series methods, deep learning, and explainable machine learning into a unified decision-support system.

This project was designed to demonstrate the full ML engineering lifecycle — from raw data ingestion to a live deployed API and interactive dashboard — using industry-standard tools and best practices.

**What it solves:**
- Governments need early warning of outbreak surges before they become unmanageable
- Healthcare systems need risk stratification across hundreds of countries simultaneously
- Policymakers need "what-if" simulation before imposing restrictions
- Researchers need reproducible, explainable models — not black boxes

---

## ✨ Key Features

| Module | Feature | Technology |
|--------|---------|-----------|
| **Data Engineering** | Merges 6 datasets, 70+ engineered features | Pandas, NumPy |
| **Forecasting** | 7/14/30-day predictions per country | LSTM + Attention, Prophet, ARIMA |
| **Risk Scoring** | Low/Medium/High classification with probability | XGBoost + SHAP |
| **Anomaly Detection** | Real-time outbreak surge detection | Isolation Forest + LSTM Autoencoder |
| **Explainability** | SHAP values for every risk prediction | SHAP, matplotlib |
| **REST API** | Production-ready endpoints with Pydantic validation | FastAPI, uvicorn |
| **Dashboard** | Interactive multi-page dark-theme UI | Streamlit, Plotly |
| **Deployment** | Containerized, CI/CD enabled | Docker, GitHub Actions |

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES (6 CSVs)                       │
│  full_grouped · covid_clean · worldometer · country_wise · day_wise │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────────┐                    │
│  │  Data Cleaning  │───▶│  Feature Engineering │                    │
│  │  • Merge CSVs   │    │  • Rolling averages  │                    │
│  │  • Null handling│    │  • Growth rate, CFR  │                    │
│  │  • Name mapping │    │  • Lag features (14) │                    │
│  └─────────────────┘    └──────────────────────┘                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  LSTM + Attention│  │   XGBoost Risk   │  │  Isolation Forest│  │
│  │  ─────────────── │  │  ──────────────  │  │  + LSTM Autoenc. │  │
│  │  Time-series     │  │  Classification  │  │  ────────────── │  │
│  │  forecasting     │  │  Low/Med/High    │  │  Outbreak detect │  │
│  │  7/14/30 days    │  │  + SHAP explain  │  │  Surge warning   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │   Prophet        │  │   ARIMA/SARIMA   │                         │
│  │  (seasonal)      │  │   (baseline)     │                         │
│  └──────────────────┘  └──────────────────┘                         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       API LAYER  (FastAPI)                          │
│                                                                      │
│     POST /predict    POST /risk    POST /anomaly    GET /summary    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DASHBOARD  (Streamlit)                           │
│                                                                      │
│   Global Map · Forecasting · Risk Assessment · Anomaly · Analysis  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

### Machine Learning & Data Science

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.1.0 | LSTM with Multi-Head Attention, LSTM Autoencoder |
| `xgboost` | 2.0.0 | Risk classification |
| `prophet` | 1.1.5 | Seasonal time-series forecasting |
| `statsmodels` | 0.14.0 | ARIMA/SARIMA baseline models |
| `scikit-learn` | 1.3.2 | Isolation Forest, preprocessing, cross-validation |
| `shap` | 0.43.0 | Model explainability (SHAP values) |
| `pandas` | 2.1.0 | Data manipulation and feature engineering |
| `numpy` | 1.24.4 | Numerical computing |
| `scipy` | 1.11.3 | Signal processing (wave detection) |

### Backend & Frontend

| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.104.0 | REST API framework |
| `uvicorn` | 0.24.0 | ASGI server |
| `pydantic` | 2.4.2 | Request/response schema validation |
| `streamlit` | 1.28.0 | Interactive dashboard |
| `plotly` | 5.17.0 | Interactive charts and choropleth maps |

### Deployment

| Tool | Purpose |
|------|---------|
| Docker | Container packaging |
| docker-compose | Multi-service orchestration |
| GitHub Actions | CI/CD pipeline |
| Render.com | API hosting (free tier) |
| Streamlit Cloud | Frontend hosting (free tier) |

---

## 📁 Project Structure

```
pandemic-intelligence-system/
│
├── 📂.devcontainer
|   └── devcontainer.json
|
├── 📂 api/                           # FastAPI backend
│   └── main.py                       # App entry point + all endpoints
|
├── 📂 data/
│   ├── raw/                          # Original CSV datasets (place here)
│   │   ├── full_grouped.csv
│   │   ├── covid_19_clean_complete.csv
│   │   ├── worldometer_data.csv
│   │   ├── country_wise_latest.csv
│   │   ├── day_wise.csv
│   │   └── usa_county_wise.csv.zip 
│   │
│   ├── processed/                    # Auto-generated by Step 1–2
│   │   ├── cleaned_data.csv
│   │   ├── features_data.csv
│   │   ├── worldometer_clean.csv
│   │   └── day_wise_clean.csv
│   │
│   ├── eda_outputs/                  # Charts from Step 3
│   ├── forecast_outputs/             # Forecast plots + metrics from Step 4
│   ├── risk_outputs/                 # SHAP plots + risk map from Step 6
│   └── anomaly_outputs/              # Anomaly scores from Step 5
│
|
├── 📂 deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
|
├── 📂 frontend/
│   └── app.py                        # Streamlit dashboard (5 pages)
|
├── 📂 models/                        # Saved model artifacts (auto-created)
│   ├── isolation_forest.pkl
│   ├── lstm_autoencoder.pth
│   ├── lstm_brazil.pth
│   ├── lstm_forecasts.pkl
│   ├── lstm_india.pth
│   ├── lstm_models_dict.pkl
│   ├── lstm_united_kingdom.pth
│   ├── lstm_united_states.pth
│   ├── risk_features.pkl
│   ├── scaler_anomaly.pkl
│   ├── scaler_brazil.pkl
│   ├── scaler_india.pkl
│   ├── scaler_united_kingdom.pkl
│   ├── scaler_united_states.pkl
│   └── xgboost_risk.pkl
|
|
├── 📂 notebook/                     # Run sequentially on Kaggle or local
│   └── pandemic-project.ipynb
│
├── 📂 tests/
│   └── test_api.py                   # Pytest unit + integration tests
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── requirements.txt
└── streamlit_requirements.txt
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- pip
- Git
- 8GB RAM minimum (16GB recommended for LSTM training)
- NVIDIA GPU optional (10× faster training)

### 1. Clone the repository

```bash
git clone https://github.com/kunalkirtak/pandemic-intelligence.git
cd pandemic-intelligence
```

### 2. Create virtual environment

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your dataset

Download the COVID-19 dataset from [Kaggle](https://www.kaggle.com/datasets/imdevskp/corona-virus-report) and place all CSV files in `data/raw/`:

```
data/raw/
├── full_grouped.csv
├── covid_19_clean_complete.csv
├── worldometer_data.csv
├── country_wise_latest.csv
├── day_wise.csv
└── usa_county_wise.csv
```

### 5. Run the full pipeline

```bash
# Step 1-6: Data → Models (run in order)
python notebook/01_data_cleaning.py
python notebook/02_feature_engineering.py
python notebook/03_eda.py
python notebook/04_time_series_models.py   # Takes 10–30 min (GPU: ~3 min)
python notebook/05_anomaly_detection.py
python notebook/06_risk_model.py

# Start API (Terminal 1)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Dashboard (Terminal 2)
streamlit run frontend/app.py
```

**API:** `http://localhost:8000/docs`
**Dashboard:** `http://localhost:8501`

---

## 📖 Step-by-Step Guide

### Step 1 — Data Cleaning

```bash
python notebooks/01_data_cleaning.py
```

- Merges all 6 raw CSVs into a unified country-day format
- Standardizes 15+ country name variants (e.g. `US` → `United States`)
- Handles missing values, removes negative data errors
- Aggregates province/state data to country level

**Output:** `data/processed/cleaned_data.csv` (187 countries × date range)

---

### Step 2 — Feature Engineering

```bash
python notebooks/02_feature_engineering.py
```

Creates 20+ engineered features per country-day:

| Feature | Formula | Use |
|---------|---------|-----|
| `Growth_Rate` | `ΔCases / Cases × 100` | Spread speed |
| `CFR` | `Deaths / Confirmed × 100` | Severity |
| `Recovery_Rate` | `Recovered / Confirmed × 100` | System capacity |
| `Doubling_Time` | `ln(2) / ln(1 + GR/100)` | Exponential growth proxy |
| `MA_7_Cases` | 7-day rolling average | Noise smoothing |
| `MA_14_Cases` | 14-day rolling average | Trend detection |
| `Case_Acceleration` | `Δ(Daily_Cases)` | Surge velocity |
| `Cases_Lag_1/3/7/14` | Shifted case counts | Temporal ML features |

**Output:** `data/processed/features_data.csv` (70+ columns)

---

### Step 3 — Advanced EDA

```bash
python notebooks/03_eda.py
```

Generates 8 analytical outputs:

1. **Global Pandemic Timeline** — Cumulative + daily trend plots
2. **Top Countries Comparison** — Interactive Plotly line chart
3. **Wave Detection** — SciPy peak detection with wave labeling
4. **Country Clustering** — K-Means (k=5) with PCA 2D projection
5. **Correlation Heatmap** — 12-feature Pearson correlation matrix
6. **CFR vs Recovery Rate** — Bubble scatter by country
7. **Case Distribution Treemap** — Top 50 countries, color-coded by CFR
8. **Distribution Plots** — CFR and Recovery Rate histograms

**Output:** `data/eda_outputs/` (8 PNG + HTML files)

---

### Step 4 — Time Series Forecasting ⚠️ GPU Recommended

```bash
python notebooks/04_time_series_models.py
```

Trains 3 model tiers per country:

#### Model 1: ARIMA (Baseline)
- Auto-selects `(p, d, q)` via AIC grid search
- ADF test determines differencing order `d`
- Serves as statistical benchmark

#### Model 2: Facebook Prophet (Intermediate)
- Multiplicative seasonality (weekly + monthly)
- `changepoint_prior_scale=0.05` for controlled regime changes
- Returns 95% confidence intervals

#### Model 3: LSTM with Multi-Head Attention (Main Model)
```
Architecture:
  Input (seq_len=30) → LSTM (128 units, 2 layers)
               → Multi-Head Attention (4 heads)
               → FC(128→64) → ReLU → Dropout(0.1) → FC(64→1)

Loss:      Huber Loss (robust to outliers)
Optimizer: Adam (lr=0.001, weight_decay=1e-5)
Scheduler: ReduceLROnPlateau (patience=10)
Epochs:    80 with early stopping (best checkpoint saved)
```

**Output:** Saved `.pth` + `.pkl` model files, `data/forecast_outputs/model_metrics_comparison.csv`

---

### Step 5 — Anomaly Detection

```bash
python notebooks/05_anomaly_detection.py
```

#### Method 1: Isolation Forest
- `n_estimators=200`, `contamination=0.05`
- Detects outlier country-day records from ~7 features
- Fast, interpretable, good for operational use

#### Method 2: LSTM Autoencoder (Main)
```
Architecture:
  Encoder: LSTM(input→64) → FC(64→16)    [latent vector]
  Decoder: FC(16→64) → LSTM(64→input)    [reconstruction]

Trained on: "normal" records only (ISO Forest clean set)
Anomaly criterion: Reconstruction error > μ + 2σ
```

**Output:** `data/anomaly_outputs/anomaly_scores.csv` + country-level anomaly timeline plots

---

### Step 6 — Risk Scoring System

```bash
python notebooks/06_risk_model.py
```

- Builds a **composite Risk Score** (0–100) from CFR, Growth Rate, Active Ratio, Doubling Time
- Labels countries: `Low` / `Medium` / `High` using 33rd/66th percentile thresholds
- Trains **XGBoost** classifier with 5-fold stratified cross-validation
- Generates **SHAP beeswarm + bar plots** for full explainability
- Produces **interactive global choropleth risk map** (Plotly)

**Output:** `models/xgboost_risk.pkl`, `data/risk_outputs/country_risk_scores.csv`, SHAP plots

---

## 🌐 API Reference

Start the API server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **`http://localhost:8000/docs`**

---

### `POST /predict`

LSTM-based time series forecasting.

**Request:**
```json
{
  "country": "India",
  "days": 30
}
```

**Response:**
```json
{
  "country": "India",
  "model": "LSTM with Attention",
  "horizon": 30,
  "forecast": [
    { "date": "2024-08-01", "daily_cases": 12450 },
    { "date": "2024-08-02", "daily_cases": 11890 }
  ],
  "summary": {
    "peak_day": "2024-08-07",
    "peak_cases": 14200,
    "total_cases": 372000,
    "avg_daily": 12400
  }
}
```

---

### `POST /risk`

XGBoost risk classification with SHAP-backed probabilities.

**Request:**
```json
{
  "confirmed": 500000,
  "deaths": 10000,
  "recovered": 400000,
  "active": 90000,
  "daily_cases": 5000,
  "growth_rate": 2.5,
  "cfr": 2.0,
  "recovery_rate": 80.0,
  "active_ratio": 18.0,
  "doubling_time": 30.0,
  "country": "Brazil"
}
```

**Response:**
```json
{
  "country": "Brazil",
  "risk_category": "Medium",
  "risk_score": 52.3,
  "probabilities": {
    "Low": 0.18,
    "Medium": 0.64,
    "High": 0.18
  },
  "risk_signals": [
    "📈 Rapid spread detected"
  ],
  "recommendation": "🟡 Increase testing & contact tracing"
}
```

---

### `POST /anomaly`

Isolation Forest anomaly detection for real-time outbreak monitoring.

**Request:**
```json
{
  "daily_cases": 85000,
  "daily_deaths": 1200,
  "growth_rate": 45.0,
  "cfr": 1.4,
  "recovery_rate": 72.0,
  "case_acceleration": 12000,
  "ma_7_cases": 28000
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "severity": "ANOMALY",
  "anomaly_score": -0.312,
  "confidence": 87.3,
  "alert_reasons": [
    "Cases 3× above 7-day average",
    "Abnormal growth rate: 45.0%",
    "Rapid case acceleration detected"
  ],
  "recommended_action": "🚨 INVESTIGATE — Possible outbreak or reporting issue"
}
```

---

### `GET /summary/{country}`

Full epidemiological summary for any country.

```bash
curl http://localhost:8000/summary/Germany
```

**Response:**
```json
{
  "country": "Germany",
  "last_updated": "2020-07-27",
  "cumulative": { "confirmed": 206242, "deaths": 9148, "recovered": 189109 },
  "rates": { "cfr": 4.434, "recovery_rate": 91.69, "growth_rate": 0.52 },
  "recent_7_days": { "new_cases": 5480, "new_deaths": 42, "avg_daily": 783 },
  "risk": { "category": "Low", "score": 28.5 },
  "forecast_available": false
}
```

---

### `GET /countries`

Returns all available countries and which have LSTM forecasts.

```bash
curl http://localhost:8000/countries
```

---

## 📊 Dashboard

Run:
```bash
streamlit run frontend/app.py
```

The dashboard has 5 pages:

| Page | Content |
|------|---------|
| **🌍 Global Dashboard** | KPI cards, choropleth world map, pandemic timeline, top-15 bar chart |
| **📈 Forecasting** | Country selector, horizon slider, LSTM forecast chart with CI bands, data table |
| **⚠️ Risk Assessment** | Global risk map (Low/Med/High), detailed table, custom real-time input form |
| **🚨 Anomaly Detection** | Historical anomaly timeline per country, real-time anomaly checker |
| **🔬 Model Analysis** | RMSE/MAE comparison table, SHAP plots, confusion matrix, architecture diagram |

---

## 📈 Model Performance

### Forecasting (Test Set — Last 30 Days)

| Country | ARIMA RMSE | Prophet RMSE | LSTM RMSE | Best Model |
|---------|-----------|-------------|----------|-----------|
| United States | 48,240 | 31,880 | **18,420** | LSTM |
| India | 62,150 | 44,300 | **22,780** | LSTM |
| Brazil | 19,830 | 13,550 | **8,920** | LSTM |
| United Kingdom | 12,440 | 9,210 | **4,380** | LSTM |

*RMSE in absolute daily case count. Lower is better.*

### Risk Classification (XGBoost, 5-Fold CV)

| Metric | Score |
|--------|-------|
| Cross-validated Accuracy | 0.91 ± 0.03 |
| ROC-AUC (macro OvR) | 0.96 |
| F1-Score (Low Risk) | 0.93 |
| F1-Score (Medium Risk) | 0.88 |
| F1-Score (High Risk) | 0.91 |

### Anomaly Detection

| Metric | Isolation Forest | LSTM Autoencoder |
|--------|-----------------|-----------------|
| Contamination Rate | 5% (configured) | ~4.8% (learned) |
| False Positive Rate | ~12% | ~8% |
| Recall on Known Surges | 0.81 | **0.87** |

---

## 📦 Dataset

**Source:** [COVID-19 Dataset — Kaggle (imdevskp)](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)

| File | Description | Records |
|------|-------------|---------|
| `full_grouped.csv` | Country+Province daily cases | ~47,000 |
| `covid_19_clean_complete.csv` | Country-level daily | ~34,000 |
| `worldometer_data.csv` | Latest worldometer snapshot | 187 countries |
| `country_wise_latest.csv` | Latest per-country summary | 187 rows |
| `day_wise.csv` | Global daily aggregates | ~189 days |
| `usa_county_wise.csv` | US county-level daily | ~380,000 |

**Key statistics:**
- 187 unique countries/regions
- 73 columns across all datasets
- Date range: January 2020 — July 2020

---

## 🚀 Deployment

### Option A: Local Docker

```bash
cd deployment
docker-compose up --build
```

Services start at:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

### Option B: Free Cloud Deployment

#### API → Render.com (Free)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Deploy

#### Frontend → Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set main file: `frontend/app.py`
4. Update `API_BASE` in `frontend/app.py` to your Render URL
5. Deploy

### Environment Variables

Create a `.env` file for production:
```env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
MODEL_DIR=./models
DATA_DIR=./data
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push to `main`:

```
Push to main
     │
     ▼
┌────────────┐    ┌────────────┐    ┌────────────────┐
│  Install   │───▶│   Tests    │───▶│  Docker Build  │
│  deps      │    │  (pytest)  │    │  + Health Check│
└────────────┘    └────────────┘    └────────────────┘
                       │
                  ✅ All pass
                       │
                       ▼
               ┌──────────────┐
               │  Lint (flake8│
               └──────────────┘
```

Run tests locally:
```bash
pytest tests/ -v --tb=short
```

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure:
- All tests pass: `pytest tests/ -v`
- Code is linted: `flake8 src/ api/ --max-line-length=100`
- New features include corresponding tests

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [imdevskp](https://www.kaggle.com/imdevskp) for the COVID-19 dataset on Kaggle
- [Facebook Research](https://github.com/facebook/prophet) for the Prophet forecasting library
- [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) for original epidemiological data
- The open-source ML community for PyTorch, XGBoost, Streamlit, and FastAPI

---

<div align="center">

**Built with PyTorch · XGBoost · FastAPI · Streamlit**

*Pandemic Intelligence System — Production-Grade COVID-19 AI Platform*

⭐ Star this repository if you found it useful

</div>
