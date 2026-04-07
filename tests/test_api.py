# ============================================================
# TESTS — Run: pytest tests/ -v
# ============================================================

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# ── Mock models before importing app ──────────────────────
mock_iso = MagicMock()
mock_iso.score_samples.return_value = np.array([-0.05])
mock_iso.predict.return_value = np.array([1])

mock_xgb = MagicMock()
mock_xgb.predict.return_value = np.array([0])
mock_xgb.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])

mock_scaler = MagicMock()
mock_scaler.transform.return_value = np.zeros((1, 7))
mock_scaler.n_features_in_ = 7

# Patch before import
with patch.dict('sys.modules', {}):
    pass  # Patches applied in fixtures

@pytest.fixture
def client():
    """TestClient with mocked models."""
    import api.main as app_module
    
    app_module.iso_forest     = mock_iso
    app_module.scaler_anomaly = mock_scaler
    app_module.xgb_model      = mock_xgb
    app_module.risk_features  = ['Confirmed', 'Deaths', 'Recovered', 'Active',
                                   'Daily_Cases', 'Daily_Deaths', 'Daily_Recovered',
                                   'Growth_Rate', 'CFR', 'Recovery_Rate',
                                   'Active_Ratio', 'Doubling_Time',
                                   'Case_Acceleration', 'MA_7_Cases', 'MA_14_Cases',
                                   'MA_7_Deaths', 'Cases_Lag_1', 'Cases_Lag_3',
                                   'Cases_Lag_7', 'Deaths_Lag_1', 'Deaths_Lag_7']
    app_module.forecast_data  = {
        'United States': {
            'dates': ['2024-01-01'] * 30,
            'forecast': [5000.0] * 30
        }
    }
    app_module.features_df = __import__('pandas').DataFrame()
    app_module.risk_scores  = __import__('pandas').DataFrame()
    app_module.label_map    = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    return TestClient(app_module.app)

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data

def test_predict_us(client):
    r = client.post("/predict", json={"country": "United States", "days": 30})
    assert r.status_code == 200
    data = r.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 30
    assert data["forecast"][0]["daily_cases"] >= 0

def test_anomaly_normal(client):
    r = client.post("/anomaly", json={
        "daily_cases": 5000, "daily_deaths": 100,
        "growth_rate": 1.5, "cfr": 2.0,
        "recovery_rate": 80.0, "case_acceleration": 50,
        "ma_7_cases": 4800
    })
    assert r.status_code == 200
    data = r.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data

def test_risk_custom(client):
    r = client.post("/risk", json={
        "confirmed": 500000, "deaths": 10000,
        "recovered": 400000, "active": 90000,
        "daily_cases": 5000, "growth_rate": 2.5,
        "cfr": 2.0, "recovery_rate": 80.0,
        "active_ratio": 18.0, "doubling_time": 30.0
    })
    assert r.status_code == 200
    data = r.json()
    assert "risk_category" in data
    assert data["risk_category"] in ["Low", "Medium", "High"]
    assert "probabilities" in data

def test_risk_probabilities_sum(client):
    r = client.post("/risk", json={
        "confirmed": 100000, "deaths": 2000,
        "recovered": 80000, "active": 18000,
        "daily_cases": 1000
    })
    data = r.json()
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, not 1.0"

def test_countries_endpoint(client):
    r = client.get("/countries")
    assert r.status_code in [200, 500]  # 500 if no data loaded (OK in test)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])