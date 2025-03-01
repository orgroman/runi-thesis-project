import argparse
import json
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

def trigger_dag(dag_id, conf=None):
    """Trigger an Airflow DAG run through the REST API."""
    # Same implementation as trigger_dag.py
    # ...

def main():
    parser = argparse.ArgumentParser(description="Trigger an Airflow DAG run")
    parser.add_argument("--dag-id", required=True, help="DAG ID to trigger")
    parser.add_argument("--conf-file", help="JSON configuration file for the DAG run")
    
    args = parser.parse_args()
    
    try:
        # Load configuration from file if provided
        conf = {}
        if args.conf_file:
            try:
                with open(args.conf_file, 'r') as f:
                    conf = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading configuration file: {str(e)}")
                sys.exit(1)
        
        # Trigger the DAG
        success = trigger_dag(args.dag_id, conf)
        
        # Return appropriate exit code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
