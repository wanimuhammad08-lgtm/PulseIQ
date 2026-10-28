"""
End-to-End Pipeline Orchestrator.
Orchestrates:
1. Ingestion of raw public healthcare dataset
2. Data quality checks and rejected record logging
3. Clinical ICD-9 standardization and feature engineering
4. Dimensional modeling and relational database loading
5. Data mart verification and sample table export
6. Execution telemetry and evidence logging
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from src.ingestion.ingest import download_and_extract_uci_data
from src.pipeline.data_quality import validate_and_clean_raw_data
from src.pipeline.transform import transform_and_create_tables
from src.database.db_manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HealthcareOrchestrator")


class HealthcarePipelineOrchestrator:
    """Executes and monitors the complete Healthcare ETL workflow."""

    def __init__(
        self,
        base_dir: str = ".",
        evidence_dir: str = "reports/execution_evidence"
    ):
        self.base_dir = base_dir
        self.evidence_dir = os.path.join(base_dir, evidence_dir)
        os.makedirs(self.evidence_dir, exist_ok=True)

    def run_pipeline(self) -> Dict[str, Any]:
        start_time = time.time()
        start_utc = datetime.now(timezone.utc).isoformat()
        logger.info("=" * 70)
        logger.info("STARTING HEALTHCARE READMISSION ETL PIPELINE EXECUTION")
        logger.info("=" * 70)

        pipeline_metrics = {
            "execution_id": f"RUN_{int(start_time)}",
            "start_time_utc": start_utc,
            "steps": {},
            "status": "RUNNING"
        }

        try:
            # STEP 1: Ingestion
            logger.info("\n--- STEP 1: DATA INGESTION & HASH VERIFICATION ---")
            t0 = time.time()
            ingest_manifest = download_and_extract_uci_data(raw_dir=os.path.join(self.base_dir, "data/raw"))
            t_ingest = time.time() - t0
            pipeline_metrics["steps"]["ingestion"] = {
                "duration_seconds": round(t_ingest, 2),
                "records_extracted": ingest_manifest["total_records_extracted"],
                "sha256": ingest_manifest["raw_sha256"],
                "status": "SUCCESS"
            }

            # STEP 2: Data Quality & Validation
            logger.info("\n--- STEP 2: DATA QUALITY, MISSING VALUE CODING & REJECTION AUDIT ---")
            t0 = time.time()
            cleaned_df, quality_report = validate_and_clean_raw_data(
                raw_csv_path=os.path.join(self.base_dir, "data/raw/diabetic_data.csv"),
                mapping_csv_path=os.path.join(self.base_dir, "data/raw/IDs_mapping.csv"),
                staging_dir=os.path.join(self.base_dir, "data/staging"),
                rejected_dir=os.path.join(self.base_dir, "data/rejected")
            )
            t_quality = time.time() - t0
            pipeline_metrics["steps"]["data_quality"] = {
                "duration_seconds": round(t_quality, 2),
                "initial_records": quality_report["initial_raw_records"],
                "valid_records": quality_report["valid_curated_records"],
                "rejected_records": quality_report["total_rejected_records"],
                "rejection_rate_percent": quality_report["rejection_rate_percent"],
                "status": "SUCCESS"
            }

            # STEP 3: Feature Engineering & Dimensional Transformation
            logger.info("\n--- STEP 3: ICD-9 MAPPING, SURROGATE KEYS & DIMENSIONAL MODELING ---")
            t0 = time.time()
            curated_tables = transform_and_create_tables(
                cleaned_df=cleaned_df,
                curated_dir=os.path.join(self.base_dir, "data/curated")
            )
            t_trans = time.time() - t0
            pipeline_metrics["steps"]["transformation"] = {
                "duration_seconds": round(t_trans, 2),
                "tables_generated": list(curated_tables.keys()),
                "total_curated_encounters": len(curated_tables["admission"]),
                "unique_patients": len(curated_tables["patient_summary"]),
                "status": "SUCCESS"
            }

            # STEP 4: Database Storage & Data Mart
            logger.info("\n--- STEP 4: RELATIONAL DATABASE STORAGE & SAMPLE EXPORT ---")
            t0 = time.time()
            db = DatabaseManager(db_path=os.path.join(self.base_dir, "data/healthcare.db"))
            db.initialize_schema()
            row_counts = db.load_curated_tables(curated_tables, if_exists="replace")
            verified_counts = db.verify_database()
            db.export_sample_tables(sample_dir=os.path.join(self.base_dir, "data/sample_tables"))
            t_db = time.time() - t0
            pipeline_metrics["steps"]["database_storage"] = {
                "duration_seconds": round(t_db, 2),
                "loaded_counts": row_counts,
                "verified_counts": verified_counts,
                "status": "SUCCESS"
            }

            # Finalize Execution
            total_duration = time.time() - start_time
            end_utc = datetime.now(timezone.utc).isoformat()
            pipeline_metrics["end_time_utc"] = end_utc
            pipeline_metrics["total_duration_seconds"] = round(total_duration, 2)
            pipeline_metrics["status"] = "SUCCESS"

            # Write Pipeline Execution Evidence
            summary_path = os.path.join(self.evidence_dir, "pipeline_run_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_metrics, f, indent=2)

            logger.info("=" * 70)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info(f"Total Duration: {total_duration:.2f}s")
            logger.info(f"Verified Database Counts: {verified_counts}")
            logger.info(f"Execution evidence saved to {summary_path}")
            logger.info("=" * 70)

            return pipeline_metrics

        except Exception as e:
            logger.exception(f"Pipeline execution failed: {e}")
            pipeline_metrics["status"] = "FAILED"
            pipeline_metrics["error"] = str(e)
            fail_path = os.path.join(self.evidence_dir, "pipeline_run_failure.json")
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_metrics, f, indent=2)
            raise


if __name__ == "__main__":
    orchestrator = HealthcarePipelineOrchestrator()
    orchestrator.run_pipeline()
