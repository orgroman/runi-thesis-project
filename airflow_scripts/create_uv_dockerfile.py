import os
import argparse
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_uv_dockerfile(
    output_dir: str, 
    pyproject_path: str = None,
    python_version: str = "3.11",
    airflow_version: str = "2.10.5"
):
    """
    Create a custom Dockerfile for Airflow using UV package manager.
    
    Args:
        output_dir: Directory where the Dockerfile will be created
        pyproject_path: Path to pyproject.toml file
        python_version: Python version to use
        airflow_version: Airflow version to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default pyproject path
    if not pyproject_path:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    else:
        pyproject_path = Path(pyproject_path)
        
    # Check if pyproject.toml exists
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    # Create the Dockerfile content
    dockerfile_content = f"""
# Custom Airflow image using uv package manager
FROM ghcr.io/astral-sh/uv:python{python_version}-bookworm-slim

# Install system dependencies
USER root
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
    build-essential \\
    default-libmysqlclient-dev \\
    libpq-dev \\
    netcat-openbsd \\
    curl \\
    gnupg \\
    && apt-get autoremove -yqq --purge \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Create airflow user
RUN useradd -ms /bin/bash -d /opt/airflow airflow

# Set working directory
WORKDIR /opt/airflow

# Copy pyproject.toml (and lockfile if exists)
COPY {pyproject_path.name} /opt/airflow/pyproject.toml

# Install Apache Airflow with UV
ENV UV_SYSTEM_PYTHON=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Airflow with UV
RUN uv pip install --system apache-airflow=={airflow_version} --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-{airflow_version}/constraints-{python_version}.txt"

# Install project dependencies from pyproject.toml
RUN uv pip install --system --no-deps -e .

# Set airflow user as the owner of the airflow directory
RUN chown -R airflow:airflow /opt/airflow

# Switch to airflow user
USER airflow

# Create airflow directories
RUN mkdir -p /opt/airflow/dags \\
    /opt/airflow/logs \\
    /opt/airflow/plugins \\
    /opt/airflow/config

# Set Airflow home
ENV AIRFLOW_HOME=/opt/airflow

# Initialize Airflow DB
RUN airflow db init

# Create admin user
RUN airflow users create \\
    --username admin \\
    --password admin \\
    --firstname Admin \\
    --lastname User \\
    --role Admin \\
    --email admin@example.com

# Expose port for Airflow webserver
EXPOSE 8080

# Start Airflow webserver and scheduler
CMD ["airflow", "webserver"]
"""

    # Write the Dockerfile
    dockerfile_path = output_dir / "Dockerfile"
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    # Copy pyproject.toml to output dir
    shutil.copy(pyproject_path, output_dir / pyproject_path.name)
    
    # Create a docker-compose.yml file
    docker_compose_content = f"""
version: '3'
services:
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7.2-bookworm
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 30s
      retries: 50

  airflow-webserver:
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    depends_on:
      - postgres
      - redis
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CORE__FERNET_KEY=''
      - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./config:/opt/airflow/config
      - ./plugins:/opt/airflow/plugins
    ports:
      - "8080:8080"
    command: airflow webserver
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  airflow-scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    depends_on:
      - airflow-webserver
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CORE__FERNET_KEY=''
      - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./config:/opt/airflow/config
      - ./plugins:/opt/airflow/plugins
    command: airflow scheduler

  airflow-worker:
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    depends_on:
      - airflow-scheduler
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CORE__FERNET_KEY=''
      - DUMB_INIT_SETSID=0
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./config:/opt/airflow/config
      - ./plugins:/opt/airflow/plugins
    command: airflow celery worker

volumes:
  postgres-db-volume:
"""

    # Write the docker-compose file
    docker_compose_path = output_dir / "docker-compose.yml"
    with open(docker_compose_path, 'w') as f:
        f.write(docker_compose_content)
    
    # Create a build script
    build_script_content = """#!/bin/bash
set -e

# Build custom Airflow image with UV
echo "Building custom Airflow image with UV package manager..."
docker-compose build

# Start the services
echo "Starting Airflow services..."
docker-compose up -d

echo "Airflow services are starting. You can access the web UI at http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
"""

    # Write the build script
    build_script_path = output_dir / "build_and_start.sh"
    with open(build_script_path, 'w') as f:
        f.write(build_script_content)
    
    # Make the build script executable
    os.chmod(build_script_path, 0o755)
    
    # Create folders for Airflow
    (output_dir / "dags").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    (output_dir / "plugins").mkdir(exist_ok=True)
    
    logger.info(f"Created Dockerfile, docker-compose.yml, and build script at {output_dir}")
    logger.info("To build and run the custom Airflow image:")
    logger.info(f"1. cd {output_dir}")
    logger.info("2. ./build_and_start.sh")
    logger.info("3. Access Airflow UI at http://localhost:8080")
    logger.info("4. Username: admin, Password: admin")

def main():
    parser = argparse.ArgumentParser(description="Create custom Airflow Dockerfile using UV")
    parser.add_argument("--output-dir", default=r"C:\Users\orgrd\workspace\airflow-uv",
                      help="Directory where the files will be created")
    parser.add_argument("--pyproject", default=None,
                      help="Path to pyproject.toml file")
    parser.add_argument("--python-version", default="3.11",
                      help="Python version to use")
    parser.add_argument("--airflow-version", default="2.10.5",
                      help="Airflow version to use")
    
    args = parser.parse_args()
    
    try:
        create_uv_dockerfile(
            output_dir=args.output_dir,
            pyproject_path=args.pyproject,
            python_version=args.python_version,
            airflow_version=args.airflow_version
        )
        return 0
    except Exception as e:
        logger.error(f"Error creating UV Dockerfile: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
