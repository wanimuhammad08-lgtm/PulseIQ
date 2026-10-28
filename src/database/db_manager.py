"""
Database Storage and Data Mart Manager.
Handles:
- DDL schema execution (SQLite / PostgreSQL)
- Bulk loading curated dimensional tables with transaction safety
- Verification of row counts and integrity
- Exporting sample populated tables for assignment grading verification (Submission Item 3)
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages analytical database connections, schema migrations, and batch ingestion."""

    def __init__(self, db_path: str = "data/healthcare.db", db_url: Optional[str] = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # Check environment variable for PostgreSQL if specified, else SQLite
        self.db_url = db_url or os.environ.get("DATABASE_URL", f"sqlite:///{os.path.abspath(db_path)}")
        self.is_sqlite = self.db_url.startswith("sqlite")
        self.engine = create_engine(self.db_url)

    def initialize_schema(self, schema_sql_path: Optional[str] = None) -> None:
        """Executes DDL schema creation script."""
        if schema_sql_path is None:
            schema_sql_path = "sql/schema_sqlite.sql" if self.is_sqlite else "sql/schema_postgres.sql"

        if not os.path.exists(schema_sql_path):
            raise FileNotFoundError(f"Schema file not found at {schema_sql_path}")

        logger.info(f"Initializing database schema from {schema_sql_path}...")
        with open(schema_sql_path, "r", encoding="utf-8") as f:
            sql_script = f.read()

        if self.is_sqlite:
            # Execute with native sqlite3 for script execution support
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.executescript(sql_script)
            conn.commit()
            conn.close()
        else:
            with self.engine.begin() as conn:
                for statement in sql_script.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(text(statement))

        logger.info("Database schema initialized successfully.")

    def load_curated_tables(
        self,
        tables_dict: Dict[str, pd.DataFrame],
        if_exists: str = "append"
    ) -> Dict[str, int]:
        """Loads Pandas DataFrames into database tables."""
        row_counts = {}
        # Order matters for foreign key dependencies:
        load_order = ["patient_summary", "admission", "diagnosis", "readmission", "healthcare_mart"]

        logger.info(f"Loading {len(load_order)} tables into database...")
        for table_name in load_order:
            if table_name in tables_dict:
                df = tables_dict[table_name]
                logger.info(f"Loading table '{table_name}' with {len(df):,} records...")
                if self.is_sqlite:
                    df.to_sql(
                        table_name,
                        con=self.engine,
                        if_exists=if_exists,
                        index=False,
                        chunksize=5000
                    )
                else:
                    df.to_sql(
                        table_name,
                        con=self.engine,
                        if_exists=if_exists,
                        index=False,
                        chunksize=2000,
                        method="multi"
                    )
                row_counts[table_name] = len(df)

        logger.info("All tables loaded into database successfully.")
        return row_counts

    def verify_database(self) -> Dict[str, int]:
        """Verifies database table counts directly via SQL queries."""
        counts = {}
        tables = ["patient_summary", "admission", "diagnosis", "readmission", "healthcare_mart"]
        with self.engine.connect() as conn:
            for tbl in tables:
                try:
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                    counts[tbl] = int(res)
                except Exception as e:
                    counts[tbl] = -1
                    logger.warning(f"Could not verify count for {tbl}: {e}")
        return counts

    def export_sample_tables(self, sample_dir: str = "data/sample_tables", n_rows: int = 15) -> None:
        """Exports sample populated rows to CSV and JSON for submission evidence (Item 3)."""
        os.makedirs(sample_dir, exist_ok=True)
        tables = ["patient_summary", "admission", "diagnosis", "readmission", "healthcare_mart"]
        sample_manifest = {}

        with self.engine.connect() as conn:
            for tbl in tables:
                df_sample = pd.read_sql(text(f"SELECT * FROM {tbl} LIMIT {n_rows}"), con=conn)
                csv_path = os.path.join(sample_dir, f"sample_{tbl}.csv")
                df_sample.to_csv(csv_path, index=False)
                sample_manifest[tbl] = {
                    "sample_file": csv_path,
                    "sample_row_count": len(df_sample),
                    "columns": list(df_sample.columns)
                }

        manifest_path = os.path.join(sample_dir, "sample_tables_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(sample_manifest, f, indent=2)

        logger.info(f"Exported sample populated tables to {sample_dir}")


if __name__ == "__main__":
    db = DatabaseManager()
    db.initialize_schema()
    print("Verification:", db.verify_database())
