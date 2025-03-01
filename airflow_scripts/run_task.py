import argparse
import logging
import sys
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_task(dag_id, task_id, execution_date=None):
    """
    Run a specific Airflow task through the REST API.
    
    Args:
        dag_id: The DAG identifier
        task_id: The task identifier
        execution_date: Optional execution date for the task
    """
    # Use current datetime if execution_date not provided
    if not execution_date:
        execution_date = datetime.now().isoformat()
    
    # Airflow API endpoint for running a task
    url = f"http://localhost:8080/api/v1/dags/{dag_id}/dagRuns/manual_task_run/taskInstances/{task_id}"
    
    # Authentication - default username and password for local Airflow
    auth = ("airflow", "airflow")
    
    # Prepare request body
    payload = {
        "execution_date": execution_date
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # First, create a manual DAG run
        create_run_url = f"http://localhost:8080/api/v1/dags/{dag_id}/dagRuns"
        create_payload = {
            "dag_run_id": "manual_task_run",
            "execution_date": execution_date,
        }
        
        create_response = requests.post(
            create_run_url,
            auth=auth,
            headers=headers,
            json=create_payload
        )
        
        if create_response.status_code not in (200, 201):
            logger.error(f"Failed to create DAG run. Status code: {create_response.status_code}")
            logger.error(f"Response: {create_response.text}")
            return False
        
        # Now run the specific task
        response = requests.post(
            url,
            auth=auth,
            headers=headers,
            json=payload
        )
        
        # Check if request was successful
        if response.status_code in (200, 201):
            logger.info(f"Successfully triggered task: {task_id} in DAG: {dag_id}")
            logger.info(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Failed to run task {task_id}. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Error running task: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run a specific Airflow task")
    parser.add_argument("--dag-id", required=True, help="DAG ID")
    parser.add_argument("--task-id", required=True, help="Task ID to run")
    parser.add_argument("--execution-date", help="Execution date (ISO format)")
    
    args = parser.parse_args()
    
    try:
        # Run the task
        success = run_task(args.dag_id, args.task_id, args.execution_date)
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
