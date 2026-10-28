"""
Data Ingestion Module for UCI Diabetes 130-US Hospitals (1999-2008) Dataset.
Adheres to Course Assignment Item 1, Item 4, and Ingestion Expectations:
- Maintains raw immutable copy of extracted data.
- Computes cryptographic SHA-256 integrity hash.
- Logs extraction timestamp, source endpoint, file status, row count, and byte size.
- Graceful fallback and error-handling.
"""

import os
import sys
import json
import hashlib
import zipfile
import logging
from datetime import datetime, timezone
import urllib.request
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UCI_DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
]


def compute_sha256(filepath: str) -> str:
    """Calculates SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_extract_uci_data(raw_dir: str = "data/raw") -> dict:
    """
    Downloads and extracts the UCI Diabetes 130-US Hospitals dataset.
    Returns extraction manifest dictionary.
    """
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "dataset_diabetes.zip")
    data_csv = os.path.join(raw_dir, "diabetic_data.csv")
    mapping_csv = os.path.join(raw_dir, "IDs_mapping.csv")
    manifest_path = os.path.join(raw_dir, "extraction_manifest.json")

    download_success = False
    used_url = None
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Healthcare-ETL/1.0"}

    # Check if raw files already exist
    if os.path.exists(data_csv) and os.path.exists(mapping_csv):
        logger.info(f"Raw CSV files already exist in {raw_dir}. Verifying integrity...")
        try:
            df_check = pd.read_csv(data_csv, nrows=5)
            if "encounter_id" in df_check.columns and "patient_nbr" in df_check.columns:
                logger.info("Existing raw dataset verified.")
                download_success = True
                used_url = "LOCAL_CACHE"
        except Exception as e:
            logger.warning(f"Verification of existing data failed ({e}), re-downloading...")

    if not download_success:
        for url in UCI_DATASET_URLS:
            try:
                logger.info(f"Attempting download from: {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, "wb") as out_file:
                    total_size = resp.headers.get("Content-Length")
                    logger.info(f"Downloading {total_size or 'unknown'} bytes...")
                    while chunk := resp.read(65536):
                        out_file.write(chunk)
                download_success = True
                used_url = url
                logger.info(f"Download completed successfully from {url}")
                break
            except Exception as e:
                logger.warning(f"Download failed from {url}: {e}")

        if not download_success:
            raise RuntimeError("All download endpoints for UCI Diabetes 130-US Hospitals failed.")

        # Extract Zip Archive
        logger.info(f"Extracting archive {zip_path} to {raw_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Check nested zip structure if applicable
            namelist = zip_ref.namelist()
            logger.info(f"Files found in archive: {namelist}")
            zip_ref.extractall(raw_dir)

            # Sometimes dataset_diabetes.zip contains another dataset_diabetes.zip
            for item in namelist:
                if item.endswith(".zip") and not item.startswith("__MACOSX"):
                    nested_zip = os.path.join(raw_dir, item)
                    logger.info(f"Extracting nested zip archive: {nested_zip}")
                    with zipfile.ZipFile(nested_zip, "r") as nested_ref:
                        nested_ref.extractall(raw_dir)

    # Locate extracted diabetic_data.csv if extracted into a subdirectory
    if not os.path.exists(data_csv):
        for root, _, files in os.walk(raw_dir):
            if "diabetic_data.csv" in files:
                src_path = os.path.join(root, "diabetic_data.csv")
                import shutil
                shutil.copy2(src_path, data_csv)
                logger.info(f"Relocated diabetic_data.csv from {src_path} to {data_csv}")
            if "IDs_mapping.csv" in files and not os.path.exists(mapping_csv):
                src_map = os.path.join(root, "IDs_mapping.csv")
                import shutil
                shutil.copy2(src_map, mapping_csv)
                logger.info(f"Relocated IDs_mapping.csv from {src_map} to {mapping_csv}")

    # Compute stats & build extraction manifest
    raw_hash = compute_sha256(data_csv) if os.path.exists(data_csv) else "N/A"
    raw_size = os.path.getsize(data_csv) if os.path.exists(data_csv) else 0

    df_sample = pd.read_csv(data_csv, low_memory=False)
    row_count, col_count = df_sample.shape

    manifest = {
        "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_name": "Diabetes 130-US hospitals for years 1999-2008 Data Set",
        "dataset_repository": "UCI Machine Learning Repository",
        "source_url": used_url,
        "raw_file_name": os.path.basename(data_csv),
        "raw_file_path": os.path.abspath(data_csv),
        "raw_sha256": raw_hash,
        "file_size_bytes": raw_size,
        "total_records_extracted": row_count,
        "total_attributes_extracted": col_count,
        "column_names": list(df_sample.columns),
        "status": "SUCCESS"
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Extraction Manifest written to {manifest_path}")
    logger.info(f"Extracted {row_count:,} rows and {col_count} columns. Checksum: {raw_hash[:16]}...")
    return manifest


if __name__ == "__main__":
    download_and_extract_uci_data()
