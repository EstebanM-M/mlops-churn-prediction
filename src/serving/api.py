"""
FastAPI application for serving churn predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0",
)

# Global variables for model and feature store
model = None
feature_store = None


class CustomerData(BaseModel):
    """Request model for customer data."""

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    customerID: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: str
    risk_level: str
    model_version: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_path: str
    feature_store_ready: bool


def load_model(model_path: str = "models/champion_catboost_v1.joblib"):
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    global model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(path)
    logger.info(f"Model loaded from {model_path}")

    return model


def initialize_feature_store():
    """Initialize Feature Store."""
    global feature_store
    feature_store = FeatureStore()
    logger.info("Feature Store initialized")


@app.on_event("startup")
def startup_event():
    """Load model and initialize components on startup."""
    logger.info("Starting up API...")

    try:
        load_model()
        initialize_feature_store()
        logger.info("✅ API startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status of the API
    """
    model_path = "models/champion_catboost_v1.joblib"

    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=model_path,
        feature_store_ready=feature_store is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerData) -> PredictionResponse:
    """
    Predict customer churn probability.

    Args:
        customer: Customer data

    Returns:
        Churn prediction with probability and risk level
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature Store not initialized")

    try:
        # Convert to DataFrame
        data_dict = customer.dict()
        df = pd.DataFrame([data_dict])

        # Create features
        features_df = feature_store.create_features(df)

        # Prepare features for prediction
        exclude_cols = [
            "customerID",
            "Churn",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols]

        # Make prediction
        prediction_proba = model.predict_proba(X)[0, 1]  # Probability of churn
        prediction_class = "Yes" if prediction_proba >= 0.5 else "No"

        # Determine risk level
        if prediction_proba >= 0.7:
            risk_level = "High"
        elif prediction_proba >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        logger.info(
            f"Prediction for {customer.customerID}: {prediction_class} ({prediction_proba:.3f})"
        )

        return PredictionResponse(
            customerID=customer.customerID,
            churn_probability=float(prediction_proba),
            churn_prediction=prediction_class,
            risk_level=risk_level,
            model_version="champion_catboost_v1",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(customers: List[CustomerData]) -> List[PredictionResponse]:
    """
    Predict churn for multiple customers.

    Args:
        customers: List of customer data

    Returns:
        List of predictions
    """
    predictions = []

    for customer in customers:
        try:
            prediction = await predict_churn(customer)
            predictions.append(prediction)
        except Exception as e:
            logger.error(f"Failed to predict for {customer.customerID}: {e}")
            # Continue with other customers

    return predictions


# Example usage for testing
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)