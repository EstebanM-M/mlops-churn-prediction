"""
Model training with MLflow tracking.
Trains multiple models and logs results to MLflow.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and evaluate multiple models with MLflow tracking.
    """

    def __init__(
        self,
        experiment_name: str = "churn-prediction",
        model_dir: str = "models",
        mlflow_tracking_uri: str = "./mlruns",
    ):
        """
        Initialize ModelTrainer.

        Args:
            experiment_name: Name of MLflow experiment
            model_dir: Directory to save models
            mlflow_tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "Churn"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from DataFrame.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column

        Returns:
            Tuple of (X, y)
        """
        # Encode target: Yes -> 1, No -> 0
        y = (df[target_col] == "Yes").astype(int)

        # Select feature columns (exclude ID and target)
        exclude_cols = ["customerID", "Churn", "gender", "Partner", "Dependents",
                       "PhoneService", "MultipleLines", "InternetService",
                       "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                       "TechSupport", "StreamingTV", "StreamingMovies",
                       "Contract", "PaperlessBilling", "PaymentMethod"]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]

        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

        return metrics

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Training XGBoost...")

        with mlflow.start_run(run_name="XGBoost"):
            # Model parameters
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
            }

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"XGBoost metrics: {metrics}")

            return model, metrics

    def train_lightgbm(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Training LightGBM...")

        with mlflow.start_run(run_name="LightGBM"):
            # Model parameters
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            }

            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"LightGBM metrics: {metrics}")

            return model, metrics

    def train_catboost(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[cb.CatBoostClassifier, Dict[str, float]]:
        """
        Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Tuple of (model, metrics)
        """
        logger.info("Training CatBoost...")

        with mlflow.start_run(run_name="CatBoost"):
            # Model parameters
            params = {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "verbose": False,
            }

            # Train model
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )

            # Predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"CatBoost metrics: {metrics}")

            return model, metrics

    def train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, Tuple]:
        """
        Train all models and return results.

        Args:
            train_df: Training DataFrame with features and target
            val_df: Validation DataFrame with features and target

        Returns:
            Dictionary with model results
        """
        # Prepare data
        X_train, y_train = self.prepare_data(train_df)
        X_val, y_val = self.prepare_data(val_df)

        results = {}

        # Train XGBoost
        xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_val, y_val)
        results["xgboost"] = (xgb_model, xgb_metrics)

        # Train LightGBM
        lgb_model, lgb_metrics = self.train_lightgbm(X_train, y_train, X_val, y_val)
        results["lightgbm"] = (lgb_model, lgb_metrics)

        # Train CatBoost
        cb_model, cb_metrics = self.train_catboost(X_train, y_train, X_val, y_val)
        results["catboost"] = (cb_model, cb_metrics)

        return results

    def select_best_model(
        self, results: Dict[str, Tuple], metric: str = "roc_auc"
    ) -> Tuple[str, object, Dict[str, float]]:
        """
        Select best model based on metric.

        Args:
            results: Dictionary with model results
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, model, metrics)
        """
        best_score = -1
        best_name = None
        best_model = None
        best_metrics = None

        for name, (model, metrics) in results.items():
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
                best_metrics = metrics

        logger.info(f"Best model: {best_name} ({metric}={best_score:.4f})")

        return best_name, best_model, best_metrics

    def save_model(self, model: object, model_name: str, version: str = "v1") -> Path:
        """
        Save model to disk.

        Args:
            model: Model to save
            model_name: Name of model
            version: Version identifier

        Returns:
            Path to saved model
        """
        filename = f"{model_name}_{version}.joblib"
        filepath = self.model_dir / filename

        joblib.dump(model, filepath)
        logger.info(f"Saved model to {filepath}")

        return filepath

    def compare_models(self, results: Dict[str, Tuple]) -> pd.DataFrame:
        """
        Create comparison DataFrame of all models.

        Args:
            results: Dictionary with model results

        Returns:
            DataFrame with model comparison
        """
        comparison = []

        for name, (model, metrics) in results.items():
            row = {"model": name}
            row.update(metrics)
            comparison.append(row)

        df = pd.DataFrame(comparison)
        df = df.sort_values("roc_auc", ascending=False)

        return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from features.feature_store import FeatureStore

    # Load features
    feature_store = FeatureStore()
    train_features = feature_store.load_features("train")
    val_features = feature_store.load_features("val")

    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(train_features, val_features)

    # Compare models
    print("\n=== Model Comparison ===")
    comparison = trainer.compare_models(results)
    print(comparison.to_string(index=False))

    # Select and save best model
    best_name, best_model, best_metrics = trainer.select_best_model(results)
    print(f"\n🏆 Best Model: {best_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")

    # Save best model
    model_path = trainer.save_model(best_model, f"champion_{best_name}")
    print(f"\n💾 Saved to: {model_path}")

    print("\n✅ Training Complete!")
    print(f"📊 View results: mlflow ui")