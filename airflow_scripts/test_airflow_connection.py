import argparse
import requests
import sys
import logging

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_airflow_connection(host="localhost", port=8080, username=None, password=None):
    """
    Simple test to check if Airflow is accessible.
    """
    base_url = f"http://{host}:{port}"
    print(f"\n--- Testing Airflow Connection ---")
    print(f"Attempting to connect to Airflow at: {base_url}")
    
    # 1. First, just try to access the server
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✓ Server is reachable at {base_url}")
        print(f"  Status code: {response.status_code}")
        
        # Print HTML title to confirm it's Airflow
        if response.status_code == 200:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else "No title found"
                print(f"  Page title: {title}")
            except ImportError:
                print("  Install BeautifulSoup4 to extract page title")
            except Exception as e:
                print(f"  Could not extract page title: {str(e)}")
        
    except requests.RequestException as e:
        print(f"✗ Failed to connect to {base_url}")
        print(f"  Error: {str(e)}")
        return False
    
    # 2. Try to access the health endpoint
    try:
        health_url = f"{base_url}/health"
        print(f"\nTrying health endpoint: {health_url}")
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✓ Health check succeeded")
            try:
                print(f"  Response: {response.json()}")
            except:
                print(f"  Response: {response.text}")
        else:
            print(f"✗ Health check failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
    except requests.RequestException as e:
        print(f"✗ Failed to access health endpoint: {str(e)}")
    
    # 3. Try to access the API if credentials provided
    if username and password:
        try:
            api_url = f"{base_url}/api/v1/dags"
            print(f"\nTrying API endpoint: {api_url}")
            response = requests.get(api_url, auth=(username, password), timeout=5)
            
            if response.status_code == 200:
                print(f"✓ API authentication successful")
                try:
                    data = response.json()
                    dag_count = len(data.get("dags", []))
                    print(f"  Found {dag_count} DAGs")
                    if dag_count > 0:
                        print(f"  First few DAGs: {', '.join([dag['dag_id'] for dag in data['dags'][:5]])}")
                except:
                    print(f"  Could not parse response JSON")
            elif response.status_code == 401:
                print(f"✗ API authentication failed. Check username and password.")
            else:
                print(f"✗ API request failed with status code: {response.status_code}")
                print(f"  Response: {response.text}")
        except requests.RequestException as e:
            print(f"✗ Failed to access API endpoint: {str(e)}")
    else:
        print("\nSkipping API test - no credentials provided")
    
    print("\nTo start Airflow locally:")
    print("1. Go to your Airflow directory: cd /c:/Users/orgrd/workspace/airflow-docker/")
    print("2. Run: docker-compose up -d")
    print("3. Wait for all services to start (may take a minute)")
    print("4. Access the UI at http://localhost:8080 (username: airflow, password: airflow)")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Airflow connection")
    parser.add_argument("--host", default="localhost", help="Airflow host")
    parser.add_argument("--port", type=int, default=8080, help="Airflow port")
    parser.add_argument("--username", help="Airflow username")
    parser.add_argument("--password", help="Airflow password")
    
    args = parser.parse_args()
    
    result = test_airflow_connection(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password
    )
    
    # Success is just being able to connect, not actually triggering a DAG
    sys.exit(0 if result else 1)
