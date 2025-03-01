import os
import subprocess
import argparse
import logging
from pathlib import Path

from create_uv_dockerfile import create_uv_dockerfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_airflow_uv(
    output_dir: str, 
    pyproject_path: str = None,
    python_version: str = "3.11",
    airflow_version: str = "2.10.5",
    dags_source: str = None,
    build: bool = False
):
    """
    Set up an Airflow environment using UV package manager.
    
    Args:
        output_dir: Directory where the Airflow setup will be created
        pyproject_path: Path to pyproject.toml file
        python_version: Python version to use
        airflow_version: Airflow version to use
        dags_source: Source directory containing DAG files (will be copied to output_dir/dags)
        build: Whether to build and start the Docker image after setup
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default paths
    if not pyproject_path:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    else:
        pyproject_path = Path(pyproject_path)
        
    # Check if pyproject.toml exists
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    # Create the Dockerfile and docker-compose files
    create_uv_dockerfile(
        output_dir=output_dir,
        pyproject_path=pyproject_path,
        python_version=python_version,
        airflow_version=airflow_version
    )
    
    # If dags source directory is provided, copy DAGs to output_dir/dags
    if dags_source:
        dags_source = Path(dags_source)
        if not dags_source.exists():
            logger.warning(f"DAGs source directory {dags_source} does not exist, skipping")
        else:
            dags_target = output_dir / "dags"
            logger.info(f"Copying DAGs from {dags_source} to {dags_target}")
            
            # Copy all Python files from source to target
            for py_file in dags_source.glob("*.py"):
                with open(py_file, 'r') as src, open(dags_target / py_file.name, 'w') as dst:
                    dst.write(src.read())
            
            # Copy requirements.txt if it exists
            req_file = dags_source / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as src, open(dags_target / "requirements.txt", 'w') as dst:
                    dst.write(src.read())
    
    # Build and start Docker container if requested
    if build:
        logger.info(f"Building and starting Airflow container in {output_dir}")
        try:
            subprocess.run(
                ["./build_and_start.sh"],
                cwd=output_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building and starting Airflow container: {e}")
            raise
    else:
        logger.info(f"Airflow UV setup created at {output_dir}")
        logger.info("To build and start the container:")
        logger.info(f"1. cd {output_dir}")
        logger.info("2. ./build_and_start.sh")

def main():
    parser = argparse.ArgumentParser(description="Set up Airflow with UV package manager")
    parser.add_argument("--output-dir", default=r"C:\Users\orgrd\workspace\airflow-uv",
                      help="Directory where the Airflow setup will be created")
    parser.add_argument("--pyproject", default=None,
                      help="Path to pyproject.toml file")
    parser.add_argument("--dags-source", default=r"C:\Users\orgrd\workspace\repos\runi-thesis-project\dags",
                      help="Source directory containing DAG files")
    parser.add_argument("--python-version", default="3.11",
                      help="Python version to use")
    parser.add_argument("--airflow-version", default="2.10.5",
                      help="Airflow version to use")
    parser.add_argument("--build", action="store_true",
                      help="Build and start the Docker container after setup")
    
    args = parser.parse_args()
    
    try:
        setup_airflow_uv(
            output_dir=args.output_dir,
            pyproject_path=args.pyproject,
            python_version=args.python_version,
            airflow_version=args.airflow_version,
            dags_source=args.dags_source,
            build=args.build
        )
        return 0
    except Exception as e:
        logger.error(f"Error setting up Airflow with UV: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
