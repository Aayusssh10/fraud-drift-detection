"""Data loading utilities for the fraud-drift-detection project."""
import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV = DATA_DIR / "creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def download(force: bool = False) -> Path:
    """Download the dataset from Kaggle into data/ using the Kaggle API.

    Requires a Kaggle API token at ~/.kaggle/kaggle.json and that the dataset's
    terms have been accepted on its Kaggle page at least once.
    """
    if RAW_CSV.exists() and not force:
        print(f"{RAW_CSV} already exists, skipping download (use --force to re-download).")
        return RAW_CSV

    from kaggle.api.kaggle_api_extended import KaggleApi

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Expected {RAW_CSV} after download but it's missing.")
    return RAW_CSV


def load_raw() -> pd.DataFrame:
    """Load the raw transactions CSV into a DataFrame."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run `python src/data.py --download` first."
        )
    return pd.read_csv(RAW_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/inspect the fraud dataset.")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    if args.download:
        path = download(force=args.force)
        df = pd.read_csv(path)
        print(f"Downloaded to {path}")
        print(f"Shape: {df.shape}")
        print(f"Class balance:\n{df['Class'].value_counts()}")
        print(f"Fraud rate: {df['Class'].mean():.4%}")
