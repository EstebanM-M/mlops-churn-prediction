"""
Webhook endpoints for automated ML operations.
Handles retraining triggers, data updates, and job management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router for webhooks
router = APIRouter(prefix="/webhook", tags=["Webhooks"])

# Simple in-memory job tracker (in production, use Redis/DB)
job_tracker = {}


class WebhookResponse(BaseModel):
    """Response model for webhook calls."""

    status: str
    message: str
    job_id: Optional[str] = None
    timestamp: str


class RetrainRequest(BaseModel):
    """Request model for retraining trigger."""

    reason: str = "manual_trigger"
    force: bool = False


class NewDataRequest(BaseModel):
    """Request model for new data notification."""

    source: str
    num_samples: int
    data_path: Optional[str] = None


def generate_job_id() -> str:
    """Generate unique job ID."""
    return f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def trigger_retraining_job(reason: str) -> str:
    """
    Trigger model retraining as background job.
    
    Args:
        reason: Reason for retraining
        
    Returns:
        Job ID
    """
    job_id = generate_job_id()
    
    job_tracker[job_id] = {
        "type": "retrain",
        "status": "queued",
        "reason": reason,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
    }
    
    logger.info(f"Queued retraining job {job_id}: {reason}")
    
    return job_id


def run_retraining_task(job_id: str):
    """
    Execute retraining in background.
    
    Args:
        job_id: Job identifier
    """
    try:
        # Update job status
        job_tracker[job_id]["status"] = "running"
        job_tracker[job_id]["started_at"] = datetime.now().isoformat()
        
        logger.info(f"Starting retraining job {job_id}")
        
        # Import here to avoid circular dependencies
        from features.feature_store import FeatureStore
        from training.train import ModelTrainer
        
        # Load features
        feature_store = FeatureStore()
        train_features = feature_store.load_features("train")
        val_features = feature_store.load_features("val")
        
        # Train models
        trainer = ModelTrainer()
        results = trainer.train_all_models(train_features, val_features)
        
        # Get best model
        best_name, best_model, best_metrics = trainer.select_best_model(results)
        
        # Save model
        model_path = trainer.save_model(best_model, f"champion_{best_name}")
        
        # Update job status
        job_tracker[job_id]["status"] = "completed"
        job_tracker[job_id]["completed_at"] = datetime.now().isoformat()
        job_tracker[job_id]["result"] = {
            "best_model": best_name,
            "roc_auc": best_metrics["roc_auc"],
            "model_path": str(model_path),
        }
        
        logger.info(f"Completed retraining job {job_id}: {best_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
        
    except Exception as e:
        logger.error(f"Retraining job {job_id} failed: {e}")
        job_tracker[job_id]["status"] = "failed"
        job_tracker[job_id]["completed_at"] = datetime.now().isoformat()
        job_tracker[job_id]["error"] = str(e)


@router.post("/retrain", response_model=WebhookResponse)
async def trigger_retrain(
    request: RetrainRequest,
    background_tasks: BackgroundTasks
) -> WebhookResponse:
    """
    Trigger model retraining.
    
    This endpoint queues a retraining job that will:
    1. Load latest features
    2. Train all models
    3. Select best model
    4. Save new champion model
    
    Args:
        request: Retraining request with reason
        background_tasks: FastAPI background tasks
        
    Returns:
        Webhook response with job ID
    """
    logger.info(f"Retrain webhook triggered: {request.reason}")
    
    # Generate job ID
    job_id = trigger_retraining_job(request.reason)
    
    # Queue background task
    background_tasks.add_task(run_retraining_task, job_id)
    
    return WebhookResponse(
        status="accepted",
        message=f"Retraining job queued: {request.reason}",
        job_id=job_id,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/new-data", response_model=WebhookResponse)
async def notify_new_data(
    request: NewDataRequest,
    background_tasks: BackgroundTasks
) -> WebhookResponse:
    """
    Notify system of new data arrival.
    
    This endpoint can be called when:
    - New customer data is available
    - Batch data is uploaded
    - External system pushes data
    
    It will:
    1. Validate data availability
    2. Optionally trigger feature creation
    3. Optionally trigger retraining (if threshold met)
    
    Args:
        request: New data notification
        background_tasks: FastAPI background tasks
        
    Returns:
        Webhook response
    """
    logger.info(f"New data webhook: {request.num_samples} samples from {request.source}")
    
    # In production, you would:
    # 1. Validate data exists
    # 2. Check data quality
    # 3. Create features
    # 4. Decide if retraining needed
    
    # For now, auto-trigger retrain if > 1000 new samples
    if request.num_samples >= 1000:
        job_id = trigger_retraining_job(f"new_data_{request.source}_{request.num_samples}")
        background_tasks.add_task(run_retraining_task, job_id)
        
        return WebhookResponse(
            status="accepted",
            message=f"New data received. Retraining triggered ({request.num_samples} samples).",
            job_id=job_id,
            timestamp=datetime.now().isoformat(),
        )
    else:
        return WebhookResponse(
            status="acknowledged",
            message=f"New data received ({request.num_samples} samples). Threshold not met for auto-retrain.",
            timestamp=datetime.now().isoformat(),
        )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict:
    """
    Get status of a background job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status and details
    """
    if job_id not in job_tracker:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job_tracker[job_id]


@router.get("/jobs")
async def list_jobs() -> Dict:
    """
    List all jobs.
    
    Returns:
        Dictionary of all jobs
    """
    return {
        "total_jobs": len(job_tracker),
        "jobs": job_tracker,
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> WebhookResponse:
    """
    Cancel a queued job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Webhook response
    """
    if job_id not in job_tracker:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = job_tracker[job_id]
    
    if job["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot cancel running job")
    
    if job["status"] == "completed":
        raise HTTPException(status_code=400, detail="Job already completed")
    
    # Mark as cancelled
    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    
    logger.info(f"Cancelled job {job_id}")
    
    return WebhookResponse(
        status="success",
        message=f"Job {job_id} cancelled",
        timestamp=datetime.now().isoformat(),
    )