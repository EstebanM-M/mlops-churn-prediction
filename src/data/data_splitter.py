"""
Data splitter for train/validation/test splits.
Handles stratified splitting and data saving.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train, validation, and test sets."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        output_dir: str = "data/processed",
    ):
        """
        Initialize data splitter.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
            output_dir: Directory to save processed data
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_data(
        self, df: pd.DataFrame, target_column: str = "Churn", stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column for stratification
            stratify: Whether to use stratified splitting

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data: {len(df)} total samples")

        # Prepare stratification column
        stratify_col = df[target_column] if stratify else None

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )

        # Second split: separate validation from training
        stratify_col_train = train_val_df[target_column] if stratify else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify_col_train,
        )

        logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Val set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        # Log class distribution
        if target_column in df.columns:
            logger.info("\nClass distribution:")
            logger.info(f"Overall: {df[target_column].value_counts(normalize=True).to_dict()}")
            logger.info(f"Train: {train_df[target_column].value_counts(normalize=True).to_dict()}")
            logger.info(f"Val: {val_df[target_column].value_counts(normalize=True).to_dict()}")
            logger.info(f"Test: {test_df[target_column].value_counts(normalize=True).to_dict()}")

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        Save train/val/test splits to CSV files.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            prefix: Optional prefix for filenames

        Returns:
            Dictionary with paths to saved files
        """
        paths = {}

        # Add prefix if provided
        prefix = f"{prefix}_" if prefix else ""

        # Save train set
        train_path = self.output_dir / f"{prefix}train.csv"
        train_df.to_csv(train_path, index=False)
        paths["train"] = train_path
        logger.info(f"Saved train set to {train_path}")

        # Save validation set
        val_path = self.output_dir / f"{prefix}val.csv"
        val_df.to_csv(val_path, index=False)
        paths["val"] = val_path
        logger.info(f"Saved validation set to {val_path}")

        # Save test set
        test_path = self.output_dir / f"{prefix}test.csv"
        test_df.to_csv(test_path, index=False)
        paths["test"] = test_path
        logger.info(f"Saved test set to {test_path}")

        return paths

    def load_splits(self, prefix: str = "") -> Dict[str, pd.DataFrame]:
        """
        Load previously saved train/val/test splits.

        Args:
            prefix: Optional prefix for filenames

        Returns:
            Dictionary with loaded DataFrames
        """
        prefix = f"{prefix}_" if prefix else ""

        splits = {}
        splits["train"] = pd.read_csv(self.output_dir / f"{prefix}train.csv")
        splits["val"] = pd.read_csv(self.output_dir / f"{prefix}val.csv")
        splits["test"] = pd.read_csv(self.output_dir / f"{prefix}test.csv")

        logger.info(f"Loaded splits from {self.output_dir}")
        logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        return splits


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_data()

    # Split data
    splitter = DataSplitter(test_size=0.2, val_size=0.2, random_state=42)
    train_df, val_df, test_df = splitter.split_data(df, target_column="Churn", stratify=True)

    # Save splits
    paths = splitter.save_splits(train_df, val_df, test_df)

    print("\n=== Data Splits ===")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"\nSaved to:")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")