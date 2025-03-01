import argparse
import logging
from pathlib import Path
import os
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_dag_imports(dag_path: Path, add_virtualenv: bool = False):
    """
    Fix imports in a DAG file to use VirtualenvOperator with required dependencies.
    
    Args:
        dag_path: Path to the DAG file
        add_virtualenv: Whether to add VirtualenvOperator to the DAG
    """
    # Read the original file
    with open(dag_path, 'r') as f:
        content = f.read()
    
    # Check for required imports
    required_packages = []
    if "from openai import OpenAI" in content:
        required_packages.append("openai==1.63.2")
    if "from pymongo import MongoClient" in content:
        required_packages.append("pymongo==4.11.1")
        required_packages.append("bson==0.5.10")
    if "import pendulum" in content:
        required_packages.append("pendulum==3.0.0")
    if "from tenacity import" in content:
        required_packages.append("tenacity==9.0.0")
    
    # Nothing to fix if no dependencies found
    if not required_packages and not add_virtualenv:
        logger.info(f"No dependencies to fix in {dag_path.name}")
        return
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        # Add the imports for VirtualenvOperator
        if add_virtualenv:
            temp.write("from airflow.providers.python.operators.python_virtualenv import PythonVirtualenvOperator\n")
        
        # Write the original content with modifications
        lines = content.split('\n')
        for line in lines:
            # Add the requirements to the PythonOperator constructor
            if "PythonOperator(" in line and add_virtualenv:
                # Replace PythonOperator with VirtualenvOperator
                line = line.replace("PythonOperator(", "PythonVirtualenvOperator(")
            
            # Add requirements to the operator if needed
            if "task_id=" in line and "requirements=[" not in line and add_virtualenv:
                line_indent = len(line) - len(line.lstrip())
                requirements_str = ", ".join([f'"{pkg}"' for pkg in required_packages])
                line = f"{line},\n{' ' * line_indent}requirements=[{requirements_str}]"
            
            temp.write(line + '\n')
    
    # Replace the original file with the modified one
    shutil.move(temp.name, dag_path)
    logger.info(f"Fixed imports in {dag_path.name}")

def fix_all_dags_in_directory(directory: Path, add_virtualenv: bool = False):
    """
    Fix imports in all DAG files in a directory.
    
    Args:
        directory: Directory containing DAG files
        add_virtualenv: Whether to add VirtualenvOperator to the DAGs
    """
    directory = Path(directory)
    
    # Find all Python files in the directory
    for file_path in directory.glob("*.py"):
        fix_dag_imports(file_path, add_virtualenv)
    
    # Create requirements.txt if it doesn't exist
    requirements_file = directory / "requirements.txt"
    if not requirements_file.exists():
        with open(requirements_file, 'w') as f:
            f.write("openai==1.63.2\n")
            f.write("pymongo==4.11.1\n")
            f.write("bson==0.5.10\n")
            f.write("pendulum==3.0.0\n")
            f.write("tenacity==9.0.0\n")
        logger.info(f"Created requirements.txt in {directory}")

def main():
    parser = argparse.ArgumentParser(description="Fix dependencies in Airflow DAG files")
    parser.add_argument("--dags-dir", default=r"C:\Users\orgrd\workspace\repos\runi-thesis-project\dags",
                      help="Directory containing DAG files")
    parser.add_argument("--add-virtualenv", action="store_true",
                      help="Add VirtualenvOperator to the DAGs")
    
    args = parser.parse_args()
    
    try:
        fix_all_dags_in_directory(args.dags_dir, args.add_virtualenv)
        return 0
    except Exception as e:
        logger.error(f"Error fixing dependencies: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
