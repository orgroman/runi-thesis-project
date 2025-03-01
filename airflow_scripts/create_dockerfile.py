import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dockerfile(airflow_dir: str, requirements_file: str = None, airflow_version: str = "2.10.5"):
    """
    Create a custom Dockerfile for Airflow with additional dependencies.
    
    Args:
        airflow_dir: Directory where the Dockerfile will be created
        requirements_file: Path to requirements.txt file (optional)
        airflow_version: Airflow version to use
    """
    airflow_dir = Path(airflow_dir)
    
    # Create the Dockerfile content
    dockerfile_content = f"""
FROM apache/airflow:{airflow_version}

USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && apt-get autoremove -yqq --purge \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages
RUN pip install --no-cache-dir --user \\
    openai==1.63.2 \\
    pymongo==4.11.1 \\
    bson==0.5.10 \\
    pendulum==3.0.0 \\
    tenacity==9.0.0
"""

    # If a requirements file is provided, use it instead of hardcoded packages
    if requirements_file and os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        # Replace the hardcoded packages with the contents of requirements.txt
        dockerfile_content = f"""
FROM apache/airflow:{airflow_version}

USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && apt-get autoremove -yqq --purge \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python packages from requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt
"""

    # Write the Dockerfile
    dockerfile_path = airflow_dir / "Dockerfile"
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    # If requirements file is provided, copy it to the airflow directory
    if requirements_file and os.path.exists(requirements_file):
        shutil.copy(requirements_file, airflow_dir / "requirements.txt")
    
    logger.info(f"Created Dockerfile at {dockerfile_path}")
    logger.info("To build and use the custom image:")
    logger.info("1. Navigate to the airflow directory: cd " + str(airflow_dir))
    logger.info("2. Build the image: docker build -t custom-airflow:latest .")
    logger.info("3. Update your docker-compose.yaml to use this image by setting:")
    logger.info("   AIRFLOW_IMAGE_NAME=custom-airflow in the environment or directly in the file")

def main():
    parser = argparse.ArgumentParser(description="Create custom Airflow Dockerfile")
    parser.add_argument("--airflow-dir", default=r"C:\Users\orgrd\workspace\airflow-docker",
                      help="Directory where the Dockerfile will be created")
    parser.add_argument("--requirements", default=None,
                      help="Path to requirements.txt file")
    parser.add_argument("--airflow-version", default="2.10.5",
                      help="Airflow version to use")
    
    args = parser.parse_args()
    
    try:
        create_dockerfile(
            airflow_dir=args.airflow_dir,
            requirements_file=args.requirements,
            airflow_version=args.airflow_version
        )
        return 0
    except Exception as e:
        logger.error(f"Error creating Dockerfile: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
