"""
Unit tests for Drift Detector.
"""

import pytest
import pandas as pd
import numpy as np
from monitoring.drift_detector import DriftDetector


def test_drift_detector_initialization(tmp_path):
    """Test drift detector initialization."""
    detector = DriftDetector(reports_dir=str(tmp_path))
    assert detector is not None
    assert detector.reports_dir.exists()


def test_no_drift_detected_same_data(sample_dataframe):
    """Test that same data shows no drift."""
    detector = DriftDetector()
    
    # Create features
    from features.feature_store import FeatureStore
    fs = FeatureStore()
    features = fs.create_features(sample_dataframe)
    
    # Check drift between same data
    drift_summary = detector.detect_data_drift(
        reference_data=features,
        current_data=features,
        save_report=False
    )
    
    assert drift_summary["dataset_drift"] == False
    assert drift_summary["drift_share"] == 0.0


def test_drift_detected_different_data():
    """Test that different distributions show drift."""
    detector = DriftDetector()
    
    # Create two different datasets
    ref_data = pd.DataFrame({
        "tenure": np.random.normal(30, 10, 1000),
        "MonthlyCharges": np.random.normal(50, 15, 1000),
    })
    
    curr_data = pd.DataFrame({
        "tenure": np.random.normal(50, 10, 1000),  # Different mean
        "MonthlyCharges": np.random.normal(80, 15, 1000),  # Different mean
    })
    
    drift_summary = detector.detect_data_drift(
        reference_data=ref_data,
        current_data=curr_data,
        save_report=False
    )
    
    # Should detect drift
    assert drift_summary["drift_share"] > 0