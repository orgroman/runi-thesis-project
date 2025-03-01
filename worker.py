import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

# Import sandbox_config but don't rely on it doing anything special
import sandbox_config

import activities
from workflow import PatentNegationAnalysisWorkflow
from mongodb import setup_collections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("patent_workflow.log"),
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Run the Temporal worker for the patent negation analysis workflow."""
    
    # Setup MongoDB collections and indexes
    setup_collections()
    logger.info("MongoDB setup complete")
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info("Connected to Temporal server")
    
    # Create a list of activity functions to register with the worker
    activity_list = [
        activities.authenticate_with_azure,
        activities.load_patent_data,
        activities.prepare_jsonl_files,
        activities.register_file_in_mongodb,
        activities.upload_file_to_openai,
        activities.submit_batch_request,
        activities.wait_for_batch_completion,
        activities.monitor_batch_status,
        activities.process_batch_results,
        activities.handle_batch_error,
    ]
    
    # Create worker for the task queue
    worker = Worker(
        client,
        task_queue="patent-negation-task-queue",
        workflows=[PatentNegationAnalysisWorkflow],
        activities=activity_list
    )
    
    # Start the worker
    logger.info("Starting worker")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
