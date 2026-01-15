"""
Drift detection using Evidently AI v0.7+.
Monitors data drift and model performance degradation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect drift in data using statistical tests.
    """

    def __init__(self, reports_dir: str = "data/monitoring"):
        """
        Initialize drift detector.

        Args:
            reports_dir: Directory to save drift reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Define feature columns (same as in training)
        self.feature_columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
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

    def kolmogorov_smirnov_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for drift detection.
        
        Args:
            reference: Reference data
            current: Current data
            threshold: P-value threshold for drift detection
            
        Returns:
            Test results
        """
        statistic, pvalue = stats.ks_2samp(reference, current)
        
        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "drift_detected": pvalue < threshold,
        }

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save_report: bool = True,
    ) -> Dict:
        """
        Detect data drift between reference and current datasets.

        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset (e.g., production data)
            save_report: Whether to save report

        Returns:
            Drift detection results
        """
        logger.info("Detecting data drift...")

        # Select only feature columns that exist in both datasets
        available_features = [
            col for col in self.feature_columns
            if col in reference_data.columns and col in current_data.columns
        ]

        logger.info(f"Checking drift for {len(available_features)} features")

        # Test each feature for drift
        drift_results = {}
        drifted_features = []

        for feature in available_features:
            ref_data = reference_data[feature].dropna()
            curr_data = current_data[feature].dropna()
            
            if len(ref_data) > 0 and len(curr_data) > 0:
                result = self.kolmogorov_smirnov_test(
                    ref_data.values,
                    curr_data.values
                )
                drift_results[feature] = result
                
                if result["drift_detected"]:
                    drifted_features.append(feature)

        # Calculate drift metrics
        num_drifted = len(drifted_features)
        total_features = len(available_features)
        drift_share = num_drifted / total_features if total_features > 0 else 0.0

        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_columns": total_features,
            "number_of_drifted_columns": num_drifted,
            "drift_share": drift_share,
            "dataset_drift": drift_share > 0.3,  # 30% threshold
            "drifted_features": drifted_features,
            "drift_details": drift_results,
        }

        # Save simple text report
        if save_report:
            report_path = self.reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self._save_text_report(drift_summary, report_path)
            drift_summary["report_path"] = str(report_path)
            logger.info(f"Report saved to {report_path}")

        # Log results
        if drift_summary["dataset_drift"]:
            logger.warning(
                f"⚠️ Data drift detected! "
                f"{num_drifted}/{total_features} features drifted ({drift_share:.1%})"
            )
            if drifted_features:
                logger.warning(f"Drifted features: {drifted_features[:5]}...")
        else:
            logger.info(
                f"✓ No significant data drift detected ({drift_share:.1%} drift share)"
            )

        return drift_summary

    def _save_text_report(self, drift_summary: Dict, report_path: Path):
        """Save drift report as text file."""
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DRIFT DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {drift_summary['timestamp']}\n")
            f.write(f"Total Features: {drift_summary['total_columns']}\n")
            f.write(f"Drifted Features: {drift_summary['number_of_drifted_columns']}\n")
            f.write(f"Drift Share: {drift_summary['drift_share']:.2%}\n")
            f.write(f"Dataset Drift: {drift_summary['dataset_drift']}\n\n")
            
            if drift_summary['drifted_features']:
                f.write("DRIFTED FEATURES:\n")
                f.write("-" * 60 + "\n")
                for feature in drift_summary['drifted_features']:
                    details = drift_summary['drift_details'][feature]
                    f.write(f"\n{feature}:\n")
                    f.write(f"  KS Statistic: {details['statistic']:.4f}\n")
                    f.write(f"  P-value: {details['pvalue']:.4f}\n")
            else:
                f.write("No drifted features detected.\n")

    def check_drift_and_alert(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_threshold: float = 0.5,
    ) -> Tuple[bool, Dict]:
        """
        Check for drift and determine if retraining is needed.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            drift_threshold: Threshold for drift detection (0-1)

        Returns:
            Tuple of (should_retrain, drift_summary)
        """
        logger.info("Checking drift and generating alerts...")

        # Detect data drift
        drift_summary = self.detect_data_drift(
            reference_data, 
            current_data,
            save_report=True
        )

        # Determine if retraining needed
        drift_share = drift_summary.get("drift_share", 0.0)
        should_retrain = drift_share >= drift_threshold

        if should_retrain:
            logger.warning(
                f"🚨 ALERT: Drift threshold exceeded ({drift_share:.2%} >= {drift_threshold:.2%})"
            )
            logger.warning("🔄 Recommendation: Trigger model retraining")
            drift_summary["alert"] = "RETRAIN_RECOMMENDED"
            drift_summary["alert_reason"] = (
                f"Drift share {drift_share:.2%} exceeds threshold {drift_threshold:.2%}"
            )
        else:
            logger.info(
                f"✓ Drift within acceptable range ({drift_share:.2%} < {drift_threshold:.2%})"
            )
            drift_summary["alert"] = "OK"

        return should_retrain, drift_summary

    def generate_monitoring_summary(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> Dict:
        """
        Generate comprehensive monitoring summary.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset

        Returns:
            Complete monitoring summary
        """
        logger.info("Generating monitoring summary...")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_samples": len(reference_data),
            "current_samples": len(current_data),
        }

        # Data drift
        try:
            drift_results = self.detect_data_drift(reference_data, current_data)
            summary["data_drift"] = drift_results
        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            summary["data_drift"] = {"error": str(e)}

        # Overall health
        if "data_drift" in summary and not summary["data_drift"].get("error"):
            if summary["data_drift"].get("dataset_drift", False):
                summary["health_status"] = "WARNING"
                summary["recommendation"] = "Monitor closely, consider retraining"
            else:
                summary["health_status"] = "HEALTHY"
                summary["recommendation"] = "No action needed"
        else:
            summary["health_status"] = "UNKNOWN"

        return summary


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from features.feature_store import FeatureStore

    # Load features
    feature_store = FeatureStore()
    train_features = feature_store.load_features("train")
    val_features = feature_store.load_features("val")

    # Initialize drift detector
    detector = DriftDetector()

    print("\n=== Data Drift Detection ===")
    drift_summary = detector.detect_data_drift(
        reference_data=train_features,
        current_data=val_features,
        save_report=True,
    )

    print(f"\nDataset Drift: {drift_summary.get('dataset_drift', 'N/A')}")
    print(f"Drift Share: {drift_summary.get('drift_share', 0):.2%}")
    print(f"Drifted Features: {drift_summary.get('number_of_drifted_columns', 0)}/{drift_summary.get('total_columns', 0)}")

    if drift_summary.get('drifted_features'):
        print(f"Features with drift: {drift_summary['drifted_features'][:5]}")

    # Check if retraining needed
    print("\n=== Drift Alert Check ===")
    should_retrain, alert_summary = detector.check_drift_and_alert(
        reference_data=train_features,
        current_data=val_features,
        drift_threshold=0.3,
    )

    print(f"Should Retrain: {should_retrain}")
    print(f"Alert: {alert_summary.get('alert', 'N/A')}")

    # Generate full monitoring summary
    print("\n=== Monitoring Summary ===")
    summary = detector.generate_monitoring_summary(
        reference_data=train_features,
        current_data=val_features,
    )

    print(f"Health Status: {summary.get('health_status', 'N/A')}")
    print(f"Recommendation: {summary.get('recommendation', 'N/A')}")

    print(f"\n✅ Monitoring complete!")
    print(f"📊 Report saved to: {drift_summary.get('report_path', 'N/A')}")