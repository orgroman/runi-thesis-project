import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

import activities
from workflow import PatentNegationAnalysisWorkflow
from workflow_concurrent import PatentNegationAnalysisConcurrentWorkflow
from workflow_fix import PatentNegationAnalysisFixedWorkflow
from workflow_hotfix import PatentNegationAnalysisHotfixWorkflow  # Add the hotfix workflow
from workflow_native_async import PatentNegationAsyncWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("worker.log"),
    ]
)

logger = logging.getLogger(__name__)

async def run_worker():
    """Run a worker with all registered activities and workflows."""
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info("Connected to Temporal server")
    
    # List all available activities for verification
    activity_list = dir(activities)
    actual_activities = [getattr(activities, name) for name in activity_list 
                        if not name.startswith("_") and callable(getattr(activities, name))]
    
    logger.info(f"Available activities: {[a.__name__ for a in actual_activities if hasattr(a, '__name__')]}")
    
    # Create and run worker with all workflows and activities
    worker = Worker(
        client,
        task_queue="patent-negation-task-queue",
        workflows=[
            PatentNegationAnalysisWorkflow, 
            PatentNegationAnalysisConcurrentWorkflow,
            PatentNegationAnalysisFixedWorkflow,
            PatentNegationAnalysisHotfixWorkflow,  # Add the hotfix workflow
            PatentNegationAsyncWorkflow
        ],
        activities=activities
    )
    
    logger.info("Starting worker with all registered activities and workflows")
    logger.info("Press Ctrl+C to exit")
    
    await worker.run()

if __name__ == "__main__":
    asyncio.run(run_worker())
