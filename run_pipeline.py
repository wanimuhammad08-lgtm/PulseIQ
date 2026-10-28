"""
1-Click Reproducible Pipeline Execution Runner.
Course: Data Engineering and MLOps
Project 5: Hospital Readmission Analytics and Prediction

Executes:
1. Ingestion of raw UCI Diabetes data
2. Data quality checks & anomaly rejection log
3. ICD-9 clinical categorization & surrogate key hashing
4. SQLite/PostgreSQL schema creation and data mart population
5. Verification of relational constraints and sample table extraction
6. Pipeline telemetry logging
"""

import sys
import os
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.orchestrator import HealthcarePipelineOrchestrator

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" HEALTHCARE READMISSION DATA PIPELINE (PART 1) - ONE-CLICK RUNNER")
    print("=" * 75 + "\n")
    orchestrator = HealthcarePipelineOrchestrator()
    metrics = orchestrator.run_pipeline()
    print("\n" + "=" * 75)
    print(" PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print(f" Execution Status : {metrics['status']}")
    print(f" Total Duration   : {metrics['total_duration_seconds']} seconds")
    print(f" Evidence Log     : reports/execution_evidence/pipeline_run_summary.json")
    print("=" * 75 + "\n")
