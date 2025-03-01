"""
Script to run both the Temporal worker and workflow starter in parallel.
This makes it easier to debug and run the complete workflow from a single entry point.
"""
import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("patent_workflow_complete.log"),
    ]
)
logger = logging.getLogger(__name__)

async def ensure_mongodb_running():
    """Check if MongoDB is running, and provide instructions if not."""
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        # Check if server responds
        client.admin.command('ping')
        logger.info("MongoDB is running")
        return True
    except Exception as e:
        logger.error(f"MongoDB is not running: {e}")
        logger.error("Please start MongoDB before continuing.")
        logger.error("You can start MongoDB with: 'mongod --dbpath=/path/to/data'")
        return False

async def ensure_temporal_running():
    """Check if Temporal server is running, and provide instructions if not."""
    try:
        # Try to connect to Temporal
        from temporalio.client import Client
        client = await Client.connect("localhost:7233")
        logger.info("Temporal server is running")
        return True
    except Exception as e:
        logger.error(f"Temporal server is not running: {e}")
        logger.error("Please start Temporal before continuing.")
        logger.error("You can start Temporal with: 'temporal server start-dev'")
        return False

async def run_worker():
    """Run the Temporal worker."""
    logger.info("Starting Temporal worker...")
    worker_path = Path(__file__).parent / "worker.py"
    
    # Start worker process
    worker_process = subprocess.Popen(
        [sys.executable, str(worker_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Give worker time to initialize
    time.sleep(3)
    
    # Check if worker started successfully
    if worker_process.poll() is not None:
        logger.error("Worker failed to start")
        output, _ = worker_process.communicate()
        logger.error(f"Worker output: {output}")
        return None
    
    logger.info("Worker started successfully")
    return worker_process

async def run_workflow(csv_path, batch_size):
    """Run the workflow starter."""
    logger.info(f"Starting workflow with CSV: {csv_path}, batch size: {batch_size}")
    starter_path = Path(__file__).parent / "starter.py"
    
    # Prepare starter arguments
    starter_args = [sys.executable, str(starter_path), "--batch-size", str(batch_size)]
    
    # Start starter process and wait for completion
    starter_process = subprocess.Popen(
        starter_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output from starter
    logger.info("Workflow starter running, streaming output:")
    for line in iter(starter_process.stdout.readline, ''):
        if not line:
            break
        print(f"[STARTER] {line.rstrip()}")
    
    # Wait for starter to complete
    starter_process.wait()
    logger.info(f"Workflow starter exited with code: {starter_process.returncode}")
    return starter_process.returncode

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the complete Patent Negation Analysis Temporal workflow")
    parser.add_argument("--csv", default="C:\\Users\\orgrd\\workspace\\data\\patentmatch_test\\patentmatch_test_no_claims.csv", 
                       help="Path to patent CSV file")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of records per batch")
    args = parser.parse_args()
    
    # Set environment variables (can be used for configuration)
    os.environ["CSV_PATH"] = args.csv
    
    # Check prerequisites
    mongo_running = await ensure_mongodb_running()
    temporal_running = await ensure_temporal_running()
    
    if not (mongo_running and temporal_running):
        logger.error("Prerequisites not met. Please start MongoDB and Temporal server.")
        return 1
    
    # Start worker
    worker_process = await run_worker()
    if worker_process is None:
        return 1
    
    try:
        # Run workflow
        return_code = await run_workflow(args.csv, args.batch_size)
        
        # Keep worker running for a bit longer to process any remaining activities
        logger.info("Waiting for worker to process remaining activities...")
        time.sleep(10)
        
        return return_code
    finally:
        # Ensure worker is terminated
        if worker_process and worker_process.poll() is None:
            logger.info("Terminating worker...")
            worker_process.terminate()
            worker_process.wait()
            logger.info("Worker terminated")

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
