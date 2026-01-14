"""
Data loader for Telco Customer Churn dataset.
"""

import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataLoader:
    """Load Telco Customer Churn dataset."""

    DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.data_dir / "telco_churn.csv"

    def download_data(self, force: bool = False) -> Path:
        """
        Download dataset from URL.

        Args:
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file
        """
        if self.data_path.exists() and not force:
            logger.info(f"Data already exists at {self.data_path}")
            return self.data_path

        logger.info(f"Downloading data from {self.DATA_URL}")

        response = requests.get(self.DATA_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(self.data_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        logger.info(f"Data downloaded successfully to {self.data_path}")
        return self.data_path

    def load_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load data into DataFrame.

        Args:
            force_download: Force re-download of data

        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists() or force_download:
            self.download_data(force=force_download)

        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Churn distribution:\n{df['Churn'].value_counts()}")

        return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = DataLoader()
    df = loader.load_data()

    print("\n=== Dataset Info ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")