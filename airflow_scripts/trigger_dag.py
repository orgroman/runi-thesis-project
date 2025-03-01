import argparse
import json
import logging
import sys
import requests
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def trigger_dag(dag_id, conf=None, host="localhost", port=8080, username="airflow", password="airflow"):
    """
    Trigger an Airflow DAG run through the REST API.
    
    Args:
        dag_id: The DAG identifier
        conf: Optional JSON configuration for the DAG run
        host: Airflow webserver host
        port: Airflow webserver port
        username: Airflow username
        password: Airflow password
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Airflow API endpoint
    url = f"http://{host}:{port}/api/v1/dags/{dag_id}/dagRuns"
    
    logger.info(f"Attempting to trigger DAG {dag_id} at {url}")
    
    # Authentication
    auth = (username, password)
    
    # Prepare request body
    payload = {
        "dag_run_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "execution_date": datetime.now().isoformat(),
        "conf": conf or {}
    }
    
    logger.debug(f"Request payload: {json.dumps(payload)}")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # First check if the Airflow server is reachable
        health_url = f"http://{host}:{port}/health"
        try:
            health_response = requests.get(health_url)
            if health_response.status_code != 200:
                logger.error(f"Airflow server health check failed with status code {health_response.status_code}")
                logger.error(f"Response: {health_response.text}")
                logger.error(f"Please make sure Airflow is running at {host}:{port}")
                return False
        except requests.RequestException as e:
            logger.error(f"Airflow server is not reachable at {host}:{port}")
            logger.error(f"Error: {str(e)}")
            logger.error("Please make sure Airflow is running and accessible")
            return False
        
        # Now check if the DAG exists
        dag_url = f"http://{host}:{port}/api/v1/dags/{dag_id}"
        try:
            dag_response = requests.get(dag_url, auth=auth)
            if dag_response.status_code == 404:
                logger.error(f"DAG '{dag_id}' not found in Airflow")
                logger.error("Please check the DAG ID or make sure the DAG is loaded into Airflow")
                return False
            elif dag_response.status_code == 401:
                logger.error("Authentication failed. Please check your Airflow username and password.")
                return False
            elif dag_response.status_code != 200:
                logger.error(f"Error checking DAG: status code {dag_response.status_code}")
                logger.error(f"Response: {dag_response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Error checking if DAG exists: {str(e)}")
            return False
        
        # Make the request to trigger the DAG
        response = requests.post(
            url,
            auth=auth,
            headers=headers,
            json=payload
        )
        
        # Check if request was successful
        if response.status_code in (200, 201):
            logger.info(f"Successfully triggered DAG: {dag_id}")
            logger.info(f"Response: {response.json()}")
            return True
        else:
            logger.error(f"Failed to trigger DAG {dag_id}. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your Airflow username and password.")
            elif response.status_code == 404:
                logger.error(f"DAG {dag_id} not found. Please check the DAG ID.")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Error triggering DAG: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Trigger an Airflow DAG run")
    parser.add_argument("--dag-id", required=True, help="DAG ID to trigger")
    parser.add_argument("--conf", default="{}", help="JSON configuration for the DAG run")
    parser.add_argument("--host", default=os.environ.get("AIRFLOW_HOST", "localhost"), 
                        help="Airflow webserver host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIRFLOW_PORT", "8080")), 
                        help="Airflow webserver port")
    parser.add_argument("--username", default=os.environ.get("AIRFLOW_USERNAME", "airflow"), 
                        help="Airflow username")
    parser.add_argument("--password", default=os.environ.get("AIRFLOW_PASSWORD", "airflow"), 
                        help="Airflow password")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse configuration if provided
        if args.conf.startswith('{'):
            try:
                conf = json.loads(args.conf)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                fixed_conf = args.conf.replace("'", '"')
                try:
                    conf = json.loads(fixed_conf)
                    logger.warning(f"Fixed JSON format issue: {args.conf} -> {fixed_conf}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON configuration: {args.conf}")
                    logger.error(f"JSON error: {str(e)}")
                    sys.exit(1)
        else:
            # If not a JSON object, treat as a simple key
            conf = {"trigger": args.conf}
        
        logger.debug(f"Using configuration: {json.dumps(conf)}")
        
        # Trigger the DAG
        success = trigger_dag(
            dag_id=args.dag_id,
            conf=conf,
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password
        )
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
