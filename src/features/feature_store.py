"""
Feature Store for Churn Prediction.
Centralized feature engineering for training and serving.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature Store for creating, storing, and loading features.
    Ensures consistency between training and serving.
    """

    def __init__(self, cache_dir: str = "data/features"):
        """
        Initialize Feature Store.

        Args:
            cache_dir: Directory to cache computed features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define feature names for reference
        self.feature_names = [
            # Original features
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            # Engineered features
            "tenure_years",
            "avg_monthly_spend",
            "total_spend_per_tenure",
            "is_senior",
            "has_partner",
            "has_dependents",
            "has_phone",
            "has_internet",
            "has_online_security",
            "has_online_backup",
            "has_device_protection",
            "has_tech_support",
            "has_streaming_tv",
            "has_streaming_movies",
            "is_month_to_month",
            "is_one_year",
            "is_two_year",
            "is_paperless",
            "payment_electronic",
            "payment_mailed",
            "payment_bank_transfer",
            "payment_credit_card",
            "service_count",
            "has_premium_services",
            "age_price_ratio",
        ]

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Creating features for {len(df)} samples")

        # Make a copy to avoid modifying original
        df = df.copy()

        # ===== DATA CLEANING (IMPORTANTE) =====
        # Fix TotalCharges: convert to numeric (some values are strings with spaces)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        # Fill NaN values in TotalCharges with MonthlyCharges (for new customers)
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

        # 1. Temporal features
        df["tenure_years"] = df["tenure"] / 12

        # 2. Financial features
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 to avoid division by zero
        df["total_spend_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

        # 3. Demographic features
        df["is_senior"] = df["SeniorCitizen"].astype(int)
        df["has_partner"] = (df["Partner"] == "Yes").astype(int)
        df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)

        # 4. Service features
        df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
        df["has_online_backup"] = (df["OnlineBackup"] == "Yes").astype(int)
        df["has_device_protection"] = (df["DeviceProtection"] == "Yes").astype(int)
        df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
        df["has_streaming_tv"] = (df["StreamingTV"] == "Yes").astype(int)
        df["has_streaming_movies"] = (df["StreamingMovies"] == "Yes").astype(int)

        # 5. Contract features
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["is_one_year"] = (df["Contract"] == "One year").astype(int)
        df["is_two_year"] = (df["Contract"] == "Two year").astype(int)

        # 6. Payment features
        df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
        df["payment_electronic"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["payment_mailed"] = (df["PaymentMethod"] == "Mailed check").astype(int)
        df["payment_bank_transfer"] = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
        df["payment_credit_card"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)

        # 7. Derived features
        df["service_count"] = (
            df["has_phone"]
            + df["has_internet"]
            + df["has_online_security"]
            + df["has_online_backup"]
            + df["has_device_protection"]
            + df["has_tech_support"]
            + df["has_streaming_tv"]
            + df["has_streaming_movies"]
        )

        df["has_premium_services"] = (
            (df["has_online_security"] == 1) | (df["has_tech_support"] == 1)
        ).astype(int)

        df["age_price_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1)

        logger.info(f"Created {len(df.columns)} total features")

        return df

    def get_feature_columns(self, include_target: bool = False) -> List[str]:
        """
        Get list of feature columns for modeling.

        Args:
            include_target: Whether to include target column

        Returns:
            List of feature column names
        """
        columns = self.feature_names.copy()
        if include_target:
            columns.append("Churn")
        return columns

    def save_features(
        self, df: pd.DataFrame, split_name: str, version: str = "v1"
    ) -> Path:
        """
        Save features to parquet file.

        Args:
            df: DataFrame with features
            split_name: Name of split (train, val, test)
            version: Version identifier

        Returns:
            Path to saved file
        """
        filename = f"{split_name}_features_{version}.parquet"
        filepath = self.cache_dir / filename

        df.to_parquet(filepath, index=False)
        logger.info(f"Saved features to {filepath}")

        return filepath

    def load_features(self, split_name: str, version: str = "v1") -> pd.DataFrame:
        """
        Load features from parquet file.

        Args:
            split_name: Name of split (train, val, test)
            version: Version identifier

        Returns:
            DataFrame with features
        """
        filename = f"{split_name}_features_{version}.parquet"
        filepath = self.cache_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Features not found: {filepath}")

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded features from {filepath}: {len(df)} samples")

        return df

    def validate_features(self, df: pd.DataFrame) -> Dict:
        """
        Validate feature DataFrame for quality issues.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results dictionary
        """
        logger.info("Validating features")

        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Check for missing values
        null_counts = df[self.feature_names].isnull().sum()
        if null_counts.sum() > 0:
            results["warnings"].append(f"Found {null_counts.sum()} null values")
            results["statistics"]["null_counts"] = null_counts[null_counts > 0].to_dict()

        # Check for infinite values
        numeric_cols = df[self.feature_names].select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df[col].isin([float("inf"), float("-inf")]).any():
                results["errors"].append(f"Infinite values in {col}")
                results["valid"] = False

        # Check value ranges
        if "tenure_years" in df.columns:
            if (df["tenure_years"] < 0).any():
                results["errors"].append("Negative tenure_years detected")
                results["valid"] = False

        if "avg_monthly_spend" in df.columns:
            if (df["avg_monthly_spend"] < 0).any():
                results["errors"].append("Negative avg_monthly_spend detected")
                results["valid"] = False

        # Basic statistics
        results["statistics"]["shape"] = df.shape
        results["statistics"]["feature_count"] = len(self.feature_names)

        if results["valid"]:
            logger.info("✓ Feature validation passed")
        else:
            logger.error(f"✗ Feature validation failed: {results['errors']}")

        return results

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistical summary of features.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with statistics
        """
        stats = {}

        numeric_features = df[self.feature_names].select_dtypes(include=["float64", "int64"])

        stats["mean"] = numeric_features.mean().to_dict()
        stats["std"] = numeric_features.std().to_dict()
        stats["min"] = numeric_features.min().to_dict()
        stats["max"] = numeric_features.max().to_dict()

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.data_loader import DataLoader
    from data.data_splitter import DataSplitter

    # Load and split data
    loader = DataLoader()
    df = loader.load_data()

    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split_data(df)

    # Create Feature Store
    feature_store = FeatureStore()

    # Create features for each split
    print("\n=== Creating Features ===")
    train_features = feature_store.create_features(train_df)
    val_features = feature_store.create_features(val_df)
    test_features = feature_store.create_features(test_df)

    # Validate features
    print("\n=== Validating Features ===")
    validation = feature_store.validate_features(train_features)
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # Save features
    print("\n=== Saving Features ===")
    feature_store.save_features(train_features, "train")
    feature_store.save_features(val_features, "val")
    feature_store.save_features(test_features, "test")

    # Get statistics
    print("\n=== Feature Statistics (Train) ===")
    stats = feature_store.get_feature_statistics(train_features)
    
    # Print some example statistics (safely)
    if 'tenure_years' in stats['mean']:
        print(f"Mean tenure_years: {stats['mean']['tenure_years']:.2f}")
    
    if 'avg_monthly_spend' in stats['mean']:
        print(f"Mean avg_monthly_spend: {stats['mean']['avg_monthly_spend']:.2f}")
    
    if 'age_price_ratio' in stats['mean']:
        print(f"Mean age_price_ratio: {stats['mean']['age_price_ratio']:.2f}")

    print(f"\n✅ Feature Store Demo Complete!")
    print(f"📁 Features saved to: {feature_store.cache_dir}")
    print(f"📊 Total features created: {len(train_features.columns)}")