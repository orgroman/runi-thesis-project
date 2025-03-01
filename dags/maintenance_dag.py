import logging
from datetime import datetime, timedelta
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from pymongo import MongoClient

from airflow_utils.mongo_utils import get_mongo_client, get_batch_stats, reset_hanging_batches

# Configure logging
logger = logging.getLogger(__name__)

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1
}

def collect_system_stats(**kwargs):
    """Collect statistics about the batch processing system."""
    client = get_mongo_client(Variable.get("mongodb_uri", "mongodb://localhost:27017/"))
    
    # Get batch statistics
    stats = get_batch_stats(client)
    
    # Calculate success rates
    if stats["total_batches"] > 0:
        stats["batch_success_rate"] = round(stats["completed_batches"] / stats["total_batches"] * 100, 2)
    else:
        stats["batch_success_rate"] = 0
        
    if stats["total_files"] > 0:
        stats["file_success_rate"] = round(stats["completed_files"] / stats["total_files"] * 100, 2)
    else:
        stats["file_success_rate"] = 0
    
    # Store stats in MongoDB for historical tracking
    db = client.patent_negation
    stats["timestamp"] = datetime.now()
    db.system_stats.insert_one(stats)
    
    logger.info(f"System stats collected: {json.dumps(stats, default=str)}")
    return stats

def detect_and_reset_hanging_batches(**kwargs):
    """Detect and reset batches that appear to be hanging."""
    client = get_mongo_client(Variable.get("mongodb_uri", "mongodb://localhost:27017/"))
    
    # Get the threshold from variables or use default (6 hours)
    hours_threshold = int(Variable.get("hanging_batch_threshold_hours", "6"))
    
    # Reset hanging batches
    count = reset_hanging_batches(client, hours_threshold)
    
    logger.info(f"Reset {count} hanging batches (threshold: {hours_threshold} hours)")
    return count

def cleanup_old_records(**kwargs):
    """Clean up old records to prevent database bloat."""
    client = get_mongo_client(Variable.get("mongodb_uri", "mongodb://localhost:27017/"))
    db = client.patent_negation
    
    # Get retention days from variables or use defaults
    retention_days = {
        "processed_batches": int(Variable.get("retention_days_processed_batches", "30")),
        "failed_batches": int(Variable.get("retention_days_failed_batches", "7"))
    }
    
    results = {}
    
    # Clean up old processed batches
    cutoff_date = datetime.now() - timedelta(days=retention_days["processed_batches"])
    processed_result = db.batch_requests.delete_many({
        "status": "completed",
        "processed_at": {"$lt": cutoff_date}
    })
    results["processed_batches_deleted"] = processed_result.deleted_count
    
    # Clean up old failed batches
    cutoff_date = datetime.now() - timedelta(days=retention_days["failed_batches"])
    failed_result = db.batch_requests.delete_many({
        "status": {"$in": ["failed", "cancelled", "expired"]},
        "processed_at": {"$lt": cutoff_date}
    })
    results["failed_batches_deleted"] = failed_result.deleted_count
    
    logger.info(f"Cleanup complete: {results}")
    return results

# Create the DAG
with DAG(
    'system_maintenance',
    default_args=default_args,
    description='Maintenance tasks for batch processing system',
    schedule_interval=timedelta(hours=6),
    start_date=days_ago(1),
    catchup=False,
    tags=['maintenance', 'monitor'],
) as dag:
    
    # Task 1: Collect system statistics
    stats_task = PythonOperator(
        task_id='collect_system_stats',
        python_callable=collect_system_stats,
        provide_context=True,
    )
    
    # Task 2: Detect and reset hanging batches
    reset_hanging_task = PythonOperator(
        task_id='reset_hanging_batches',
        python_callable=detect_and_reset_hanging_batches,
        provide_context=True,
    )
    
    # Task 3: Clean up old records
    cleanup_task = PythonOperator(
        task_id='cleanup_old_records',
        python_callable=cleanup_old_records,
        provide_context=True,
    )
    
    # Define task dependencies - run stats first, then others in parallel
    stats_task >> [reset_hanging_task, cleanup_task]
