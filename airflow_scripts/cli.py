#!/usr/bin/env python
import argparse
import os
import sys
import logging
import subprocess
import requests
import json
from pathlib import Path
from datetime import datetime

# Import our utility scripts
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from deploy_dags import deploy_dags
from test_airflow_connection import test_airflow_connection
from trigger_dag import trigger_dag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AirflowCLI:
    """Command Line Interface for managing Airflow workflows."""
    
    def __init__(self):
        self.airflow_docker_dir = os.environ.get(
            "AIRFLOW_DOCKER_DIR", r"C:\Users\orgrd\workspace\airflow-docker"
        )
        self.dags_source_dir = os.environ.get(
            "AIRFLOW_DAGS_SOURCE", r"C:\Users\orgrd\workspace\repos\runi-thesis-project\dags"
        )
        self.airflow_host = os.environ.get("AIRFLOW_HOST", "localhost")
        self.airflow_port = int(os.environ.get("AIRFLOW_PORT", "8080"))
        self.airflow_username = os.environ.get("AIRFLOW_USERNAME", "airflow")
        self.airflow_password = os.environ.get("AIRFLOW_PASSWORD", "airflow")
    
    def start_services(self, args):
        """Start Airflow services using Docker Compose."""
        print(f"Starting Airflow services in {self.airflow_docker_dir}")
        
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.airflow_docker_dir,
                check=True
            )
            print("✓ Airflow services started successfully")
            print("\nNote: It may take a minute for all services to be ready")
            print("You can check status with 'python airflow_scripts/cli.py status'")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error starting Airflow services: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
        
        return True
    
    def stop_services(self, args):
        """Stop Airflow services using Docker Compose."""
        print(f"Stopping Airflow services in {self.airflow_docker_dir}")
        
        try:
            subprocess.run(
                ["docker-compose", "down"],
                cwd=self.airflow_docker_dir,
                check=True
            )
            print("✓ Airflow services stopped successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error stopping Airflow services: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
        
        return True
    
    def check_status(self, args):
        """Check status of Airflow services."""
        return test_airflow_connection(
            host=self.airflow_host,
            port=self.airflow_port,
            username=self.airflow_username,
            password=self.airflow_password
        )
    
    def deploy_dags(self, args):
        """Deploy DAG files to Airflow."""
        source_dir = args.source or self.dags_source_dir
        target_dir = os.path.join(self.airflow_docker_dir, "dags")
        
        print(f"Deploying DAGs from {source_dir} to {target_dir}")
        try:
            deploy_dags(
                source_dir=source_dir,
                target_dir=target_dir,
                backup=not args.no_backup
            )
            print("✓ DAGs deployed successfully")
            return True
        except Exception as e:
            print(f"✗ Error deploying DAGs: {e}")
            return False
    
    def trigger_dag(self, args):
        """Trigger a DAG run."""
        dag_id = args.dag_id
        
        # Parse conf if provided
        conf = {}
        if args.conf:
            try:
                if args.conf.startswith("{"):
                    # Try parsing as JSON
                    try:
                        conf = json.loads(args.conf)
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        fixed_conf = args.conf.replace("'", '"')
                        conf = json.loads(fixed_conf)
                else:
                    # Treat as simple key
                    conf = {"trigger": args.conf}
            except Exception as e:
                print(f"✗ Error parsing configuration: {e}")
                return False
        
        print(f"Triggering DAG: {dag_id}")
        try:
            success = trigger_dag(
                dag_id=dag_id,
                conf=conf,
                host=self.airflow_host,
                port=self.airflow_port,
                username=self.airflow_username,
                password=self.airflow_password
            )
            
            if success:
                print(f"✓ Successfully triggered DAG: {dag_id}")
                print(f"  Check progress at: http://{self.airflow_host}:{self.airflow_port}/dags/{dag_id}/grid")
            else:
                print(f"✗ Failed to trigger DAG: {dag_id}")
            
            return success
        except Exception as e:
            print(f"✗ Error triggering DAG: {e}")
            return False
    
    def list_dags(self, args):
        """List available DAGs from Airflow API."""
        url = f"http://{self.airflow_host}:{self.airflow_port}/api/v1/dags"
        auth = (self.airflow_username, self.airflow_password)
        
        try:
            response = requests.get(url, auth=auth)
            
            if response.status_code == 200:
                dags = response.json().get("dags", [])
                
                if not dags:
                    print("No DAGs found")
                    return True
                
                print(f"Found {len(dags)} DAGs:")
                print("-" * 50)
                print(f"{'DAG ID':<30} {'Active':<10} {'Paused':<10}")
                print("-" * 50)
                
                for dag in dags:
                    print(f"{dag['dag_id']:<30} {str(dag.get('is_active', False)):<10} {str(dag.get('is_paused', True)):<10}")
                
                return True
            else:
                print(f"✗ Error listing DAGs: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Error listing DAGs: {e}")
            return False
        
    def view_logs(self, args):
        """View logs from Airflow containers."""
        container = args.container or "airflow-webserver"
        lines = args.lines or 100
        follow = args.follow
        
        try:
            cmd = ["docker-compose", "logs"]
            if follow:
                cmd.append("-f")
            
            cmd.extend(["--tail", str(lines), container])
            
            print(f"Viewing logs for container '{container}':")
            print("-" * 50)
            
            subprocess.run(
                cmd,
                cwd=self.airflow_docker_dir,
            )
            
            return True
            
        except KeyboardInterrupt:
            print("\nStopped following logs")
            return True
        except Exception as e:
            print(f"✗ Error viewing logs: {e}")
            return False

def main():
    """Main entry point for the CLI."""
    airflow_cli = AirflowCLI()
    
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Command line interface for managing Airflow workflows"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start Airflow services")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop Airflow services")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check Airflow status")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy DAGs to Airflow")
    deploy_parser.add_argument("--source", help="Source directory containing DAG files")
    deploy_parser.add_argument("--no-backup", action="store_true", help="Skip backup of existing DAGs")
    
    # Trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger a DAG run")
    trigger_parser.add_argument("dag_id", help="ID of the DAG to trigger")
    trigger_parser.add_argument("--conf", help="Configuration for the DAG run (JSON or simple value)")
    
    # List DAGs command
    list_parser = subparsers.add_parser("list", help="List available DAGs")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View logs from Airflow containers")
    logs_parser.add_argument("--container", choices=["airflow-webserver", "airflow-scheduler", "airflow-worker"],
                           help="Container to view logs from")
    logs_parser.add_argument("--lines", type=int, help="Number of lines to show")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "start":
        return 0 if airflow_cli.start_services(args) else 1
    elif args.command == "stop":
        return 0 if airflow_cli.stop_services(args) else 1
    elif args.command == "status":
        return 0 if airflow_cli.check_status(args) else 1
    elif args.command == "deploy":
        return 0 if airflow_cli.deploy_dags(args) else 1
    elif args.command == "trigger":
        return 0 if airflow_cli.trigger_dag(args) else 1
    elif args.command == "list":
        return 0 if airflow_cli.list_dags(args) else 1
    elif args.command == "logs":
        return 0 if airflow_cli.view_logs(args) else 1
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
