# ============================================================
# STEP 7: FASTAPI BACKEND
# Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import torch
import warnings
import os
warnings.filterwarnings('ignore')

# ── Import models ──────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="🦠 Pandemic Intelligence System API",
    description="""
    ## COVID-19 AI Forecasting & Risk Analysis API
    
    **Endpoints:**
    - `/predict` — LSTM 30-day case forecasting per country
    - `/risk` — XGBoost risk classification (Low/Medium/High)
    - `/anomaly` — Isolation Forest anomaly detection
    - `/countries` — Available countries list
    - `/summary/{country}` — Full country summary
    """,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load artifacts at startup ──────────────────────────────
@app.on_event("startup")
async def load_models():
    global iso_forest, scaler_anomaly, xgb_model, risk_features
    global features_df, forecast_data, risk_scores, label_map
    
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    try:
        iso_forest     = joblib.load('models/isolation_forest.pkl')
        scaler_anomaly = joblib.load('models/scaler_anomaly.pkl')
        print("✅ Anomaly model loaded")
    except Exception as e:
        print(f"⚠️  Anomaly model: {e}")
        iso_forest = scaler_anomaly = None
    
    try:
        xgb_model     = joblib.load('models/xgboost_risk.pkl')
        risk_features = joblib.load('models/risk_features.pkl')
        print("✅ Risk model loaded")
    except Exception as e:
        print(f"⚠️  Risk model: {e}")
        xgb_model = risk_features = None
    
    try:
        forecast_data = joblib.load('models/lstm_forecasts.pkl')
        print(f"✅ LSTM forecasts loaded ({len(forecast_data)} countries)")
    except Exception as e:
        print(f"⚠️  LSTM forecasts: {e}")
        forecast_data = {}
    
    try:
        features_df = pd.read_csv('data/processed/features_data.csv',
                                   parse_dates=['Date'])
        risk_scores = pd.read_csv('data/risk_outputs/country_risk_scores.csv')
        print(f"✅ Data loaded ({features_df['Country'].nunique()} countries)")
    except Exception as e:
        print(f"⚠️  Data load: {e}")
        features_df = pd.DataFrame()
        risk_scores = pd.DataFrame()

# ── Pydantic Schemas ───────────────────────────────────────
class PredictRequest(BaseModel):
    country: str = Field(..., example="United States")
    days: int    = Field(30, ge=7, le=60, description="Forecast horizon (7–60 days)")

class RiskRequest(BaseModel):
    confirmed:       float = Field(..., ge=0)
    deaths:          float = Field(..., ge=0)
    recovered:       float = Field(..., ge=0)
    active:          float = Field(..., ge=0)
    daily_cases:     float = Field(0, ge=0)
    daily_deaths:    float = Field(0, ge=0)
    growth_rate:     float = Field(0)
    cfr:             float = Field(0, ge=0, le=100)
    recovery_rate:   float = Field(0, ge=0, le=100)
    active_ratio:    float = Field(0, ge=0, le=100)
    doubling_time:   float = Field(999, ge=0)
    case_acceleration: float = Field(0)
    ma_7_cases:      float = Field(0)
    ma_14_cases:     float = Field(0)
    ma_7_deaths:     float = Field(0)
    cases_lag_1:     float = Field(0)
    cases_lag_3:     float = Field(0)
    cases_lag_7:     float = Field(0)
    deaths_lag_1:    float = Field(0)
    deaths_lag_7:    float = Field(0)
    country:         Optional[str] = Field(None, example="India")

class AnomalyRequest(BaseModel):
    daily_cases:       float = Field(..., ge=0)
    daily_deaths:      float = Field(..., ge=0)
    growth_rate:       float = Field(0)
    cfr:               float = Field(0, ge=0, le=100)
    recovery_rate:     float = Field(0, ge=0, le=100)
    case_acceleration: float = Field(0)
    ma_7_cases:        float = Field(0)

# ── Helpers ────────────────────────────────────────────────
def get_country_data(country: str):
    if features_df.empty:
        return None
    matches = features_df[features_df['Country'].str.lower() == country.lower()]
    if len(matches) == 0:
        # Fuzzy match
        matches = features_df[features_df['Country'].str.lower().str.contains(
            country.lower()[:4])]
    return matches.sort_values('Date') if len(matches) > 0 else None

# ── Endpoints ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "🟢 Running",
        "service": "Pandemic Intelligence System API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/risk", "/anomaly",
                       "/countries", "/summary/{country}", "/docs"]
    }

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "models": {
            "lstm_forecasts":   len(forecast_data) > 0,
            "xgboost_risk":     xgb_model is not None,
            "isolation_forest": iso_forest is not None,
        },
        "data_loaded": not features_df.empty,
        "countries_available": features_df['Country'].nunique() if not features_df.empty else 0
    }

@app.get("/countries", tags=["Data"])
async def list_countries():
    if features_df.empty:
        raise HTTPException(500, "Data not loaded")
    countries = sorted(features_df['Country'].unique().tolist())
    return {
        "count": len(countries),
        "countries": countries,
        "forecast_available": list(forecast_data.keys())
    }

@app.post("/predict", tags=["Forecasting"])
async def predict_cases(req: PredictRequest):
    """
    LSTM-based time series forecasting for a country.
    Returns daily case predictions for next N days.
    """
    country = req.country
    days    = req.days
    
    # Check pre-computed forecasts
    matched_key = None
    for k in forecast_data.keys():
        if k.lower() == country.lower():
            matched_key = k
            break
    
    if matched_key:
        fc = forecast_data[matched_key]
        dates    = fc['dates'][:days]
        forecast = fc['forecast'][:days]
        
        return {
            "country":  matched_key,
            "model":    "LSTM with Attention",
            "horizon":  days,
            "forecast": [
                {
                    "date":        dates[i],
                    "daily_cases": round(max(0, float(forecast[i])), 0),
                }
                for i in range(len(dates))
            ],
            "summary": {
                "peak_day":    dates[int(np.argmax(forecast))],
                "peak_cases":  round(float(np.max(forecast)), 0),
                "total_cases": round(float(np.sum(forecast)), 0),
                "avg_daily":   round(float(np.mean(forecast)), 0),
            },
            "note": "Forecast based on historical patterns. CI not included in this endpoint."
        }
    
    # Fallback: Use historical trend
    cdf = get_country_data(country)
    if cdf is None or len(cdf) == 0:
        raise HTTPException(404, f"Country '{country}' not found")
    
    # Simple exponential trend fallback
    recent = cdf['Daily_Cases'].tail(14).values.astype(float)
    recent = np.maximum(recent, 0)
    avg_growth = np.mean(np.diff(recent)) if len(recent) > 1 else 0
    last_val   = recent[-1] if len(recent) > 0 else 0
    last_date  = cdf['Date'].max()
    
    forecasts = []
    for i in range(1, days + 1):
        pred  = max(0, last_val + avg_growth * i)
        fdate = (last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        forecasts.append({"date": fdate, "daily_cases": round(pred, 0)})
    
    return {
        "country":  country,
        "model":    "Trend Extrapolation (LSTM model not available for this country)",
        "horizon":  days,
        "forecast": forecasts
    }

@app.post("/risk", tags=["Risk"])
async def get_risk_score(req: RiskRequest):
    """
    XGBoost-based risk classification.
    Returns Low / Medium / High risk with probability breakdown.
    """
    # Try lookup first
    if req.country and not risk_scores.empty:
        r = risk_scores[risk_scores['Country'].str.lower() == req.country.lower()]
        if len(r) > 0:
            row = r.iloc[0]
            return {
                "country":       req.country,
                "risk_category": row.get('XGB_Risk_Category', 'Unknown'),
                "risk_score":    round(float(row.get('Risk_Score', 0)), 2),
                "probabilities": {
                    "Low":    round(float(row.get('Risk_Prob_Low', 0)), 3),
                    "Medium": round(float(row.get('Risk_Prob_Medium', 0)), 3),
                    "High":   round(float(row.get('Risk_Prob_High', 0)), 3),
                },
                "key_indicators": {
                    "cfr":           round(float(row.get('CFR', 0)), 3),
                    "growth_rate":   round(float(row.get('Growth_Rate', 0)), 3),
                    "active_ratio":  round(float(row.get('Active_Ratio', 0)), 3),
                    "doubling_time": round(float(row.get('Doubling_Time', 999)), 1),
                },
                "source": "pre-computed"
            }
    
    # Compute on the fly
    if xgb_model is None or risk_features is None:
        raise HTTPException(503, "Risk model not loaded")
    
    input_dict = {
        'Confirmed': req.confirmed, 'Deaths': req.deaths,
        'Recovered': req.recovered, 'Active': req.active,
        'Daily_Cases': req.daily_cases, 'Daily_Deaths': req.daily_deaths,
        'Daily_Recovered': 0, 'Growth_Rate': req.growth_rate,
        'CFR': req.cfr, 'Recovery_Rate': req.recovery_rate,
        'Active_Ratio': req.active_ratio, 'Doubling_Time': req.doubling_time,
        'Case_Acceleration': req.case_acceleration,
        'MA_7_Cases': req.ma_7_cases, 'MA_14_Cases': req.ma_14_cases,
        'MA_7_Deaths': req.ma_7_deaths,
        'Cases_Lag_1': req.cases_lag_1, 'Cases_Lag_3': req.cases_lag_3,
        'Cases_Lag_7': req.cases_lag_7,
        'Deaths_Lag_1': req.deaths_lag_1, 'Deaths_Lag_7': req.deaths_lag_7
    }
    
    X = np.array([[input_dict.get(f, 0) for f in risk_features]])
    
    pred_class = int(xgb_model.predict(X)[0])
    pred_proba = xgb_model.predict_proba(X)[0]
    
    # Derived risk signals
    signals = []
    if req.cfr > 3:           signals.append("⚠️ High Case Fatality Rate")
    if req.growth_rate > 5:   signals.append("📈 Rapid spread detected")
    if req.active_ratio > 40: signals.append("🏥 High active case burden")
    if req.doubling_time < 10: signals.append("⏱️ Fast doubling time")
    
    return {
        "country":       req.country or "Custom Input",
        "risk_category": label_map[pred_class],
        "risk_score":    round(float(pred_class * 33 + pred_proba[pred_class] * 33), 2),
        "probabilities": {
            "Low":    round(float(pred_proba[0]), 3),
            "Medium": round(float(pred_proba[1]), 3),
            "High":   round(float(pred_proba[2]), 3),
        },
        "risk_signals": signals,
        "recommendation": {
            0: "✅ Monitor routine surveillance",
            1: "🟡 Increase testing & contact tracing",
            2: "🔴 Immediate intervention required"
        }[pred_class],
        "source": "real-time computation"
    }

@app.post("/anomaly", tags=["Anomaly"])
async def detect_anomaly(req: AnomalyRequest):
    """
    Isolation Forest anomaly detection.
    Detects outbreak surges and data inconsistencies.
    """
    if iso_forest is None or scaler_anomaly is None:
        raise HTTPException(503, "Anomaly model not loaded")
    
    feature_names = ['Daily_Cases', 'Daily_Deaths', 'Growth_Rate',
                      'CFR', 'Recovery_Rate', 'Case_Acceleration', 'MA_7_Cases']
    
    X = np.array([[
        req.daily_cases, req.daily_deaths, req.growth_rate,
        req.cfr, req.recovery_rate, req.case_acceleration, req.ma_7_cases
    ]])
    
    # Use only features the model was trained on
    n_expected = scaler_anomaly.n_features_in_
    X = X[:, :n_expected]
    
    X_scaled = scaler_anomaly.transform(X)
    score    = float(iso_forest.score_samples(X_scaled)[0])
    pred     = int(iso_forest.predict(X_scaled)[0])
    
    is_anomaly = pred == -1
    severity   = "ANOMALY" if is_anomaly else "NORMAL"
    
    threshold = -0.15   # Typical anomaly boundary
    confidence = min(100, max(0, (abs(score - threshold) / 0.3) * 100))
    
    reasons = []
    if req.daily_cases > req.ma_7_cases * 3:
        reasons.append("Cases 3× above 7-day average")
    if req.growth_rate > 20:
        reasons.append(f"Abnormal growth rate: {req.growth_rate:.1f}%")
    if req.cfr > 10:
        reasons.append(f"Very high CFR: {req.cfr:.1f}%")
    if req.case_acceleration > req.daily_cases * 0.5:
        reasons.append("Rapid case acceleration detected")
    
    return {
        "is_anomaly":   is_anomaly,
        "severity":     severity,
        "anomaly_score": round(score, 4),
        "confidence":   round(confidence, 1),
        "interpretation": {
            "score":        score,
            "threshold":    threshold,
            "description": ("This data point shows unusual pandemic behavior" if is_anomaly
                            else "Normal range — no outbreak signal detected")
        },
        "alert_reasons":    reasons,
        "recommended_action": ("🚨 INVESTIGATE — Possible outbreak or reporting issue"
                               if is_anomaly else "✅ Continue standard monitoring")
    }

@app.get("/summary/{country}", tags=["Data"])
async def get_country_summary(country: str):
    """Full COVID-19 summary for a country."""
    cdf = get_country_data(country)
    if cdf is None or len(cdf) == 0:
        raise HTTPException(404, f"Country '{country}' not found")
    
    latest_row = cdf.iloc[-1]
    
    # Risk info
    risk_info = {}
    if not risk_scores.empty:
        r = risk_scores[risk_scores['Country'].str.lower() == country.lower()]
        if len(r) > 0:
            risk_info = {
                "category": r.iloc[0].get('XGB_Risk_Category', 'Unknown'),
                "score":    round(float(r.iloc[0].get('Risk_Score', 0)), 2)
            }
    
    return {
        "country": latest_row['Country'],
        "last_updated": str(latest_row['Date'].date()
                            if hasattr(latest_row['Date'], 'date')
                            else latest_row['Date']),
        "cumulative": {
            "confirmed": int(latest_row.get('Confirmed', 0)),
            "deaths":    int(latest_row.get('Deaths', 0)),
            "recovered": int(latest_row.get('Recovered', 0)),
            "active":    int(latest_row.get('Active', 0)),
        },
        "rates": {
            "cfr":           round(float(latest_row.get('CFR', 0)), 3),
            "recovery_rate": round(float(latest_row.get('Recovery_Rate', 0)), 3),
            "active_ratio":  round(float(latest_row.get('Active_Ratio', 0)), 3),
            "growth_rate":   round(float(latest_row.get('Growth_Rate', 0)), 3),
            "doubling_time": round(float(latest_row.get('Doubling_Time', 999)), 1),
        },
        "recent_7_days": {
            "new_cases":  int(cdf['Daily_Cases'].tail(7).sum()),
            "new_deaths": int(cdf['Daily_Deaths'].tail(7).sum()),
            "avg_daily":  round(float(cdf['Daily_Cases'].tail(7).mean()), 0),
        },
        "risk": risk_info,
        "forecast_available": country in forecast_data or
                               any(k.lower() == country.lower() for k in forecast_data)
    }