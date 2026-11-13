"""Airflow DAG: hourly incremental re-index of the knowledge base."""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="rag_incremental_ingest",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=3)},
    tags=["genai", "rag", "pgvector"],
) as dag:
    BashOperator(
        task_id="ingest_changed_documents",
        bash_command="cd /opt/airflow/project && python src/pipeline.py --source ${DOCS_PATH:-sample_docs}",
    )
