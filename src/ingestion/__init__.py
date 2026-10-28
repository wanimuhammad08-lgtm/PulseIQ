"""Data Ingestion module for raw public healthcare datasets."""
from .ingest import download_and_extract_uci_data

__all__ = ["download_and_extract_uci_data"]
