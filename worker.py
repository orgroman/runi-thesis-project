import asyncio
import logging
import os
import sys
import inspect
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker

# Setup path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import workflows and activities
from workflow import PatentNegationAnalysisWorkflow
from workflow_optimized import PatentNegationAnalysisOptimizedWorkflow
import activities
from mongodb import setup_collections, get_mongo_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load sandbox configuration if exists
if os.path.exists(".sandbox.env"):
    with open(".sandbox.env", "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value
    logger.info("Temporal sandbox configuration loaded")

async def main():
    """Run a worker to execute workflows and activities."""
    # Set up MongoDB collections
    setup_collections()
    logger.info("MongoDB setup complete")
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info("Connected to Temporal server")
    
    # Extract activity functions from the activities module
    activity_list = []
    for name in dir(activities):
        if not name.startswith("_"):  # Skip private attributes
            attr = getattr(activities, name)
            if callable(attr) and hasattr(attr, "__module__") and attr.__module__.startswith("activities."):
                activity_list.append(attr)
    
    logger.info(f"Found {len(activity_list)} activities to register")
    
    # Create worker with registered workflows and activities
    worker = Worker(
        client,
        task_queue="patent-negation-task-queue",
        workflows=[
            PatentNegationAnalysisWorkflow,
            PatentNegationAnalysisOptimizedWorkflow
        ],
        activities=activity_list  # Pass the list of activity functions
    )
    
    # Print available workflows
    logger.info("Available workflows:")
    for workflow_name in worker._workflow_registry:
        logger.info(f" - {workflow_name}")
    
    # Print available activities
    logger.info("Available activities:")
    for activity_name in worker._activity_registry:
        logger.info(f" - {activity_name}")
    
    # Run worker until interrupted
    logger.info("Starting worker")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
