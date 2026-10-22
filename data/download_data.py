"""
Download script for the Historical Flood & Meteorological Dataset.

Data Source:
  Gauhar, N., Das, S., & Moury, K. S. (2021).
  "Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm."
  2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST), IEEE.
  GitHub: https://github.com/n-gauhar/Flood-prediction

This dataset aggregates 65+ years (1949–2015) of authentic meteorological records from
the Bangladesh Meteorological Department across 35 stations, mapped against historical
flood disaster records from the Flood Forecasting and Warning Centre (BWDB) and UN OCHA / Dartmouth Flood Observatory.
"""

import os
import sys
import urllib.request
import pandas as pd

RAW_DATA_URL = "https://raw.githubusercontent.com/n-gauhar/Flood-prediction/master/FloodPrediction.csv"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
DEST_FILE = os.path.join(RAW_DIR, "FloodPrediction.csv")


def download_dataset(force: bool = False) -> str:
    """Download the historical flood dataset if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(DEST_FILE) and not force:
        file_size = os.path.getsize(DEST_FILE)
        if file_size > 100_000:
            print(f"[INFO] Dataset already exists at: {DEST_FILE} ({file_size:,} bytes). Skipping download.")
            return DEST_FILE

    print(f"[INFO] Fetching historical flood dataset from open repository...")
    print(f"       URL: {RAW_DATA_URL}")
    req = urllib.request.Request(
        RAW_DATA_URL,
        headers={"User-Agent": "FloodRiskPrediction-EducationalProject/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()

        with open(DEST_FILE, "wb") as f:
            f.write(content)

        # Quick validation
        df = pd.read_csv(DEST_FILE)
        print(f"[SUCCESS] Downloaded successfully! Shape: {df.shape} (rows, cols)")
        print(f"          Saved to: {DEST_FILE}")
        return DEST_FILE

    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    download_dataset()
