"""
Monitoring endpoints for drift detection.
"""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from monitoring.drift_detector import DriftDetector
from features.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Initialize components
drift_detector = None
feature_store = None


def initialize_monitoring():
    """Initialize monitoring components."""
    global drift_detector, feature_store
    
    drift_detector = DriftDetector()
    feature_store = FeatureStore()
    
    logger.info("Monitoring components initialized")


class DriftCheckResponse(BaseModel):
    """Response for drift check."""
    
    status: str
    drift_detected: bool
    drift_share: float
    number_of_drifted_columns: int
    total_columns: int
    health_status: str
    recommendation: str
    report_path: str


@router.get("/health", response_model=Dict)
async def monitoring_health() -> Dict:
    """
    Check monitoring system health.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy" if drift_detector is not None else "unhealthy",
        "drift_detector_ready": drift_detector is not None,
        "feature_store_ready": feature_store is not None,
    }


@router.post("/check-drift", response_model=DriftCheckResponse)
async def check_drift() -> DriftCheckResponse:
    """
    Check for data drift between training and validation data.
    
    This endpoint:
    1. Loads reference (training) data
    2. Loads current (validation) data
    3. Runs drift detection
    4. Returns drift metrics and recommendations
    
    Returns:
        Drift detection results
    """
    if drift_detector is None or feature_store is None:
        raise HTTPException(
            status_code=503, 
            detail="Monitoring components not initialized"
        )
    
    try:
        logger.info("Running drift check...")
        
        # Load data
        train_features = feature_store.load_features("train")
        val_features = feature_store.load_features("val")
        
        # Check drift
        should_retrain, drift_summary = drift_detector.check_drift_and_alert(
            reference_data=train_features,
            current_data=val_features,
            drift_threshold=0.3,
        )
        
        # Determine health status
        if drift_summary["dataset_drift"]:
            health_status = "WARNING"
            recommendation = "Consider retraining model"
        else:
            health_status = "HEALTHY"
            recommendation = "No action needed"
        
        return DriftCheckResponse(
            status="completed",
            drift_detected=drift_summary["dataset_drift"],
            drift_share=drift_summary["drift_share"],
            number_of_drifted_columns=drift_summary["number_of_drifted_columns"],
            total_columns=drift_summary["total_columns"],
            health_status=health_status,
            recommendation=recommendation,
            report_path=drift_summary.get("report_path", "N/A"),
        )
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


@router.get("/summary", response_model=Dict)
async def monitoring_summary() -> Dict:
    """
    Get comprehensive monitoring summary.
    
    Returns:
        Complete monitoring status
    """
    if drift_detector is None or feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Monitoring components not initialized"
        )
    
    try:
        # Load data
        train_features = feature_store.load_features("train")
        val_features = feature_store.load_features("val")
        
        # Generate summary
        summary = drift_detector.generate_monitoring_summary(
            reference_data=train_features,
            current_data=val_features,
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Monitoring summary failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring summary failed: {str(e)}"
        )