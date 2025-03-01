import argparse
import json
import logging
import sys
import requests
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_airflow_setup(host="localhost", port=8080, username="airflow", password="airflow"):
    """
    Check if Airflow is properly set up and running.
    
    Args:
        host: Airflow webserver host
        port: Airflow webserver port
        username: Airflow username
        password: Airflow password
        
    Returns:
        dict: Results of the checks
    """
    results = {
        "server_accessible": False,
        "health_check": False,
        "authentication_valid": False,
        "dags_loaded": False,
        "dags_list": [],
        "errors": []
    }
    
    try:
        # Check if server is accessible
        logger.info(f"Checking if Airflow server is accessible at http://{host}:{port}")
        base_url = f"http://{host}:{port}"
        
        try:
            response = requests.get(base_url, timeout=5)
            results["server_accessible"] = True
            logger.info(f"✓ Airflow server is accessible at {base_url}")
        except requests.RequestException as e:
            logger.error(f"✗ Airflow server is not accessible: {str(e)}")
            results["errors"].append(f"Server not accessible: {str(e)}")
            return results
            
        # Check health
        try:
            health_url = f"{base_url}/health"
            health_response = requests.get(health_url, timeout=5)
            if health_response.status_code == 200:
                results["health_check"] = True
                health_data = health_response.json()
                logger.info(f"✓ Airflow health check passed: {json.dumps(health_data)}")
            else:
                logger.error(f"✗ Airflow health check failed: {health_response.status_code}")
                results["errors"].append(f"Health check failed: {health_response.text}")
        except requests.RequestException as e:
            logger.error(f"✗ Error during health check: {str(e)}")
            results["errors"].append(f"Health check error: {str(e)}")
            
        # Check authentication
        try:
            auth_url = f"{base_url}/api/v1/dags"
            auth_response = requests.get(auth_url, auth=(username, password), timeout=5)
            if auth_response.status_code == 200:
                results["authentication_valid"] = True
                logger.info(f"✓ Authentication successful")
            elif auth_response.status_code == 401:
                logger.error(f"✗ Authentication failed. Please check username and password.")
                results["errors"].append("Authentication failed")
            else:
                logger.error(f"✗ Authentication check returned unexpected status code: {auth_response.status_code}")
                results["errors"].append(f"Authentication check failed: {auth_response.text}")
        except requests.RequestException as e:
            logger.error(f"✗ Error during authentication check: {str(e)}")
            results["errors"].append(f"Authentication check error: {str(e)}")
            
        # Check if DAGs are loaded
        if results["authentication_valid"]:
            try:
                dags_response = auth_response
                dags_data = dags_response.json()
                
                if "dags" in dags_data and len(dags_data["dags"]) > 0:
                    results["dags_loaded"] = True
                    results["dags_list"] = [dag["dag_id"] for dag in dags_data["dags"]]
                    logger.info(f"✓ Found {len(results['dags_list'])} DAGs: {', '.join(results['dags_list'])}")
                else:
                    logger.warning("⚠ No DAGs found. Make sure you have DAGs in your dags folder.")
            except Exception as e:
                logger.error(f"✗ Error checking DAGs: {str(e)}")
                results["errors"].append(f"DAGs check error: {str(e)}")
                
    except Exception as e:
        logger.error(f"✗ Unexpected error during setup check: {str(e)}")
        results["errors"].append(f"Unexpected error: {str(e)}")
        
    # Print summary
    print("\n--- Airflow Setup Check Summary ---")
    print(f"Server accessible: {'✓' if results['server_accessible'] else '✗'}")
    print(f"Health check: {'✓' if results['health_check'] else '✗'}")
    print(f"Authentication: {'✓' if results['authentication_valid'] else '✗'}")
    print(f"DAGs loaded: {'✓' if results['dags_loaded'] else '✗'}")
    
    if results["dags_list"]:
        print(f"\nAvailable DAGs:")
        for dag in results["dags_list"]:
            print(f"  - {dag}")
    
    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")
            
    print("\nSuggestions:")
    if not results["server_accessible"]:
        print("  - Make sure Docker and Airflow services are running")
        print("  - Check if you need to use a different host or port")
    if not results["authentication_valid"]:
        print("  - Check your Airflow username and password")
    if not results["dags_loaded"]:
        print("  - Make sure your DAGs are in the correct folder")
        print("  - Check for syntax errors in your DAG files")
        print("  - Restart Airflow services if needed")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Check Airflow setup")
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
        results = check_airflow_setup(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password
        )
        
        # Return appropriate exit code
        if not results["server_accessible"] or len(results["errors"]) > 0:
            sys.exit(1)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
