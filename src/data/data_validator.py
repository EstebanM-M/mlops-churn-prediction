"""
Data validator for quality checks.
Simple, robust validation without complex Great Expectations setup.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality with custom checks."""

    def __init__(self):
        """Initialize data validator."""
        self.validation_results = []

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate DataFrame with comprehensive checks.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating data: {len(df)} rows, {len(df.columns)} columns")

        results = {
            "success": True,
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "warnings": [],
            "errors": [],
            "statistics": {},
        }

        # Check 1: Required columns exist
        self._check_required_columns(df, results)

        # Check 2: No null values in critical columns
        self._check_null_values(df, results)

        # Check 3: Data types are correct
        self._check_data_types(df, results)

        # Check 4: Value ranges are valid
        self._check_value_ranges(df, results)

        # Check 5: Categorical values are valid
        self._check_categorical_values(df, results)

        # Check 6: Unique constraints
        self._check_unique_constraints(df, results)

        # Calculate statistics
        results["statistics"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "churn_rate": float(df["Churn"].value_counts(normalize=True).get("Yes", 0)),
        }

        # Overall success
        results["success"] = results["failed_checks"] == 0

        logger.info(
            f"Validation complete: {results['passed_checks']}/{results['total_checks']} checks passed"
        )

        return results

    def _check_required_columns(self, df: pd.DataFrame, results: Dict) -> None:
        """Check that all required columns exist."""
        required_columns = [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
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
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

        results["total_checks"] += 1
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            results["failed_checks"] += 1
            results["errors"].append(f"Missing columns: {missing_columns}")
            logger.error(f"Missing columns: {missing_columns}")
        else:
            results["passed_checks"] += 1
            logger.info("✓ All required columns present")

    def _check_null_values(self, df: pd.DataFrame, results: Dict) -> None:
        """Check for null values in critical columns."""
        critical_columns = ["customerID", "Churn"]

        results["total_checks"] += 1
        null_counts = df[critical_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if len(columns_with_nulls) > 0:
            results["failed_checks"] += 1
            results["errors"].append(
                f"Null values in critical columns: {columns_with_nulls.to_dict()}"
            )
            logger.error(f"Null values found: {columns_with_nulls.to_dict()}")
        else:
            results["passed_checks"] += 1
            logger.info("✓ No null values in critical columns")

        # Check for excessive nulls in other columns
        null_percentage = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentage[null_percentage > 50]

        if len(high_null_cols) > 0:
            results["warnings"].append(
                f"Columns with >50% nulls: {high_null_cols.to_dict()}"
            )
            logger.warning(f"High null percentage: {high_null_cols.to_dict()}")

    def _check_data_types(self, df: pd.DataFrame, results: Dict) -> None:
        """Check that columns have expected data types."""
        results["total_checks"] += 1

        type_errors = []

        # Numeric columns
        numeric_cols = ["tenure", "MonthlyCharges"]
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                type_errors.append(f"{col} should be numeric, got {df[col].dtype}")

        # Categorical/string columns
        categorical_cols = ["gender", "Contract", "Churn"]
        for col in categorical_cols:
            if col in df.columns and not pd.api.types.is_object_dtype(df[col]):
                type_errors.append(f"{col} should be string/object, got {df[col].dtype}")

        if type_errors:
            results["failed_checks"] += 1
            results["errors"].extend(type_errors)
            logger.error(f"Data type errors: {type_errors}")
        else:
            results["passed_checks"] += 1
            logger.info("✓ Data types are correct")

    def _check_value_ranges(self, df: pd.DataFrame, results: Dict) -> None:
        """Check that numeric values are within expected ranges."""
        results["total_checks"] += 1

        range_errors = []

        # tenure: 0-100 months
        if "tenure" in df.columns:
            invalid_tenure = df[(df["tenure"] < 0) | (df["tenure"] > 100)]
            if len(invalid_tenure) > 0:
                range_errors.append(f"tenure out of range: {len(invalid_tenure)} rows")

        # MonthlyCharges: 0-200
        if "MonthlyCharges" in df.columns:
            invalid_charges = df[(df["MonthlyCharges"] < 0) | (df["MonthlyCharges"] > 200)]
            if len(invalid_charges) > 0:
                range_errors.append(
                    f"MonthlyCharges out of range: {len(invalid_charges)} rows"
                )

        if range_errors:
            results["failed_checks"] += 1
            results["errors"].extend(range_errors)
            logger.error(f"Value range errors: {range_errors}")
        else:
            results["passed_checks"] += 1
            logger.info("✓ Value ranges are valid")

    def _check_categorical_values(self, df: pd.DataFrame, results: Dict) -> None:
        """Check that categorical columns have valid values."""
        results["total_checks"] += 1

        categorical_errors = []

        # Define valid values for categorical columns
        valid_values = {
            "gender": ["Male", "Female"],
            "Churn": ["Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }

        for col, valid_vals in valid_values.items():
            if col in df.columns:
                invalid_vals = set(df[col].dropna().unique()) - set(valid_vals)
                if invalid_vals:
                    categorical_errors.append(
                        f"{col} has invalid values: {invalid_vals}"
                    )

        if categorical_errors:
            results["failed_checks"] += 1
            results["errors"].extend(categorical_errors)
            logger.error(f"Categorical value errors: {categorical_errors}")
        else:
            results["passed_checks"] += 1
            logger.info("✓ Categorical values are valid")

    def _check_unique_constraints(self, df: pd.DataFrame, results: Dict) -> None:
        """Check uniqueness constraints."""
        results["total_checks"] += 1

        # customerID should be unique
        if "customerID" in df.columns:
            duplicate_ids = df["customerID"].duplicated().sum()
            if duplicate_ids > 0:
                results["failed_checks"] += 1
                results["errors"].append(
                    f"Duplicate customerIDs found: {duplicate_ids}"
                )
                logger.error(f"Found {duplicate_ids} duplicate customerIDs")
            else:
                results["passed_checks"] += 1
                logger.info("✓ customerID is unique")
        else:
            results["failed_checks"] += 1
            results["errors"].append("customerID column not found")

    def generate_report(self, results: Dict) -> str:
        """
        Generate human-readable validation report.

        Args:
            results: Validation results dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nOverall Status: {'PASSED' if results['success'] else 'FAILED'}")
        report.append(
            f"Checks: {results['passed_checks']}/{results['total_checks']} passed"
        )

        if results["statistics"]:
            report.append("\n--- Statistics ---")
            for key, value in results["statistics"].items():
                report.append(f"  {key}: {value}")

        if results["errors"]:
            report.append("\n--- Errors ---")
            for error in results["errors"]:
                report.append(f"  {error}")

        if results["warnings"]:
            report.append("\n--- Warnings ---")
            for warning in results["warnings"]:
                report.append(f"  {warning}")

        report.append("=" * 60)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_data()

    # Validate data
    validator = DataValidator()
    results = validator.validate_data(df)

    # Generate report
    report = validator.generate_report(results)
    print(report)

    # Save report to file
    report_path = Path("data/processed/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")