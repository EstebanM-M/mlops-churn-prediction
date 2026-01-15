"""
Integration tests for API.
"""

import pytest
from fastapi.testclient import TestClient
from serving.api import app, load_model, initialize_feature_store, initialize_monitoring

# Initialize components before creating client
load_model()
initialize_feature_store()
try:
    initialize_monitoring()
except:
    pass  # Monitoring may fail in tests

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """Test health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_endpoint(sample_customer_data):
    """Test prediction endpoint."""
    response = client.post("/predict", json=sample_customer_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "customerID" in data
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert 0 <= data["churn_probability"] <= 1


def test_predict_invalid_data():
    """Test prediction with invalid data."""
    invalid_data = {"customerID": "TEST"}  # Missing required fields
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_monitoring_health():
    """Test monitoring health endpoint."""
    response = client.get("/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_webhook_retrain():
    """Test retrain webhook."""
    response = client.post(
        "/webhook/retrain",
        json={"reason": "test_trigger", "force": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data