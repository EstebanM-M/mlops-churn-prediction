"""
Unit tests for Feature Store.
"""

import pytest
import pandas as pd
from features.feature_store import FeatureStore


def test_feature_store_initialization():
    """Test feature store can be initialized."""
    fs = FeatureStore()
    assert fs is not None
    assert len(fs.feature_names) > 0


def test_create_features(sample_dataframe):
    """Test feature creation."""
    fs = FeatureStore()
    features = fs.create_features(sample_dataframe)
    
    # Check new features exist
    assert "tenure_years" in features.columns
    assert "avg_monthly_spend" in features.columns
    assert "is_senior" in features.columns
    
    # Check values are correct
    assert features["tenure_years"].iloc[0] == 1.0  # 12 months / 12


def test_validate_features(sample_dataframe):
    """Test feature validation."""
    fs = FeatureStore()
    features = fs.create_features(sample_dataframe)
    
    validation = fs.validate_features(features)
    
    assert "valid" in validation
    assert validation["valid"] == True


def test_save_and_load_features(sample_dataframe, tmp_path):
    """Test saving and loading features."""
    fs = FeatureStore(cache_dir=str(tmp_path))
    features = fs.create_features(sample_dataframe)
    
    # Save
    path = fs.save_features(features, "test")
    assert path.exists()
    
    # Load
    loaded = fs.load_features("test")
    assert len(loaded) == len(features)
    assert list(loaded.columns) == list(features.columns)