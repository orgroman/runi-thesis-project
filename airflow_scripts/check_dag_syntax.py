import argparse
import os
import sys
import importlib.util
import logging
from pathlib import Path
import tempfile
import shutil
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dag_syntax(dag_file_path):
    """
    Check syntax of a DAG file using Python's built-in capabilities.
    
    Args:
        dag_file_path: Path to the DAG file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if the file exists
        if not os.path.exists(dag_file_path):
            return False, f"File does not exist: {dag_file_path}"
        
        # Compile the file to check for syntax errors
        with open(dag_file_path, 'r') as file:
            code = file.read()
        
        compile(code, dag_file_path, 'exec')
        
        # Try to import the file in a temporary directory
        # This catches import errors that might occur in a real import
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, os.path.basename(dag_file_path))
            shutil.copy(dag_file_path, temp_file)
            
            # Run in a subprocess to isolate the import
            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.append('{temp_dir}'); import {os.path.splitext(os.path.basename(dag_file_path))[0]}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, result.stderr.strip()
        
        return True, "Syntax is valid"
        
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_airflow_dag_parsing(dag_file_path, airflow_docker_dir):
    """
    Use airflow to parse the DAG file and check for Airflow-specific errors.
    
    Args:
        dag_file_path: Path to the DAG file
        airflow_docker_dir: Path to the Airflow docker directory
        
    Returns:
        tuple: (is_valid, output)
    """
    try:
        # Copy the DAG file to the Airflow dags directory
        dag_filename = os.path.basename(dag_file_path)
        target_path = os.path.join(airflow_docker_dir, "dags", f"temp_{dag_filename}")
        
        # Create a copy of the file
        shutil.copy(dag_file_path, target_path)
        
        # Run Airflow parse command in Docker
        command = [
            "docker-compose", 
            "-f", 
            os.path.join(airflow_docker_dir, "docker-compose.yaml"),
            "exec", 
            "-T", 
            "airflow-webserver",
            "airflow", 
            "dags", 
            "parse", 
            f"/opt/airflow/dags/temp_{dag_filename}",
            "-S"
        ]
        
        result = subprocess.run(
            command,
            cwd=airflow_docker_dir,
            capture_output=True,
            text=True
        )
        
        # Clean up the temporary file
        try:
            os.remove(target_path)
        except:
            pass
        
        if result.returncode == 0:
            return True, "DAG parsed successfully by Airflow"
        else:
            return False, result.stderr.strip() or result.stdout.strip()
        
    except Exception as e:
        return False, f"Error running Airflow parser: {str(e)}"

def check_all_dags_in_directory(directory, airflow_docker_dir=None):
    """
    Check syntax of all DAG files in a directory.
    
    Args:
        directory: Directory containing DAG files
        airflow_docker_dir: Path to the Airflow docker directory
    """
    directory = Path(directory)
    results = []
    
    # Find all Python files in the directory
    for file_path in directory.glob("*.py"):
        file_str = str(file_path)
        
        print(f"\nChecking {file_path.name}")
        print("-" * (10 + len(file_path.name)))
        
        # Check Python syntax
        is_valid, message = check_dag_syntax(file_str)
        
        if is_valid:
            print(f"✓ Python syntax: Valid")
        else:
            print(f"✗ Python syntax: {message}")
        
        # Check Airflow parsing if docker directory is provided
        airflow_valid = False
        airflow_message = "Skipped (no Airflow directory specified)"
        
        if airflow_docker_dir and is_valid:
            airflow_valid, airflow_message = check_airflow_dag_parsing(file_str, airflow_docker_dir)
            
            if airflow_valid:
                print(f"✓ Airflow parsing: Valid")
            else:
                print(f"✗ Airflow parsing: Error")
                print(f"  {airflow_message}")
        elif airflow_docker_dir:
            print(f"⚠ Airflow parsing: Skipped due to Python syntax errors")
        else:
            print(f"⚠ Airflow parsing: {airflow_message}")
        
        results.append({
            "file": file_path.name,
            "python_valid": is_valid,
            "python_message": message,
            "airflow_valid": airflow_valid,
            "airflow_message": airflow_message
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Check syntax of Airflow DAG files")
    parser.add_argument("--file", help="Path to a specific DAG file to check")
    parser.add_argument("--dir", default="/c:/Users/orgrd/workspace/repos/runi-thesis-project/dags",
                        help="Directory containing DAG files")
    parser.add_argument("--airflow-dir", default="/c:/Users/orgrd/workspace/airflow-docker",
                        help="Airflow docker directory for advanced checks")
    
    args = parser.parse_args()
    
    try:
        if args.file:
            # Check a specific file
            is_valid, message = check_dag_syntax(args.file)
            print(f"File: {args.file}")
            print(f"Valid: {is_valid}")
            print(f"Message: {message}")
            
            if is_valid and args.airflow_dir:
                airflow_valid, airflow_message = check_airflow_dag_parsing(args.file, args.airflow_dir)
                print(f"Airflow valid: {airflow_valid}")
                print(f"Airflow message: {airflow_message}")
                
            return 0 if is_valid else 1
        else:
            # Check all files in the directory
            results = check_all_dags_in_directory(args.dir, args.airflow_dir)
            
            # Print summary
            print("\n--- Summary ---")
            all_valid = all(result["python_valid"] for result in results)
            airflow_valid = all(result.get("airflow_valid", False) for result in results)
            
            print(f"Checked {len(results)} Python files")
            print(f"Python syntax: {sum(1 for r in results if r['python_valid'])} valid, "
                  f"{sum(1 for r in results if not r['python_valid'])} invalid")
            
            if args.airflow_dir:
                print(f"Airflow parsing: {sum(1 for r in results if r.get('airflow_valid', False))} valid, "
                      f"{sum(1 for r in results if not r.get('airflow_valid', False) and r['python_valid'])} invalid")
            
            return 0 if all_valid else 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
