import asyncio
import logging
import sys
from datetime import datetime

from temporalio.client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log"),
    ]
)

logger = logging.getLogger(__name__)

WORKFLOW_ID = "patent-negation-analysis"

async def check_workflow_status():
    """Check the status of a workflow execution and print details."""
    client = await Client.connect("localhost:7233")
    
    # Get the handle for the workflow
    handle = client.get_workflow_handle(WORKFLOW_ID)
    
    try:
        # Get workflow description which includes the status and other metadata
        desc = await handle.describe()
        
        # Log the workflow ID
        logger.info(f"Workflow ID: {handle.id}")
        
        # In the Python SDK, desc.id is the run ID directly
        logger.info(f"Run ID: {desc.id}")
        
        # Log the status
        logger.info(f"Status: {desc.status}")
        
        # Log start time
        logger.info(f"Started: {desc.start_time}")
        
        # Calculate execution time
        if desc.start_time:
            execution_time = datetime.now().astimezone() - desc.start_time
            logger.info(f"Execution time: {execution_time}")
            
        # Check if workflow has failed - access status in a safer way
        if hasattr(desc.status, 'name') and desc.status.name == "FAILED":
            try:
                # Get workflow execution history
                history = await handle.fetch_history()
                
                # Access and print history information if available
                logger.info(f"History events: {len(history.events) if history and hasattr(history, 'events') else 'No events'}")
            except Exception as hist_error:
                logger.error(f"Error fetching history: {str(hist_error)}")
    
    except Exception as e:
        logger.error(f"Error checking workflow status: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def run_worker_debug():
    """Run a simplified worker for debugging purposes."""
    from temporalio.worker import Worker
    import activities
    from workflow import PatentNegationAnalysisWorkflow
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info(f"Connected to Temporal server")
    
    # Create and run worker
    worker = Worker(
        client,
        task_queue="patent-negation-task-queue",
        workflows=[PatentNegationAnalysisWorkflow],
        activities=activities
    )
    
    logger.info("Starting worker for debugging...")
    await worker.run()

async def main():
    """Main function to process commands."""
    if len(sys.argv) < 2:
        print("Usage: python debug.py [check|terminate]")
        return
        
    command = sys.argv[1].lower()
    
    if command == "check":
        await check_workflow_status()
    elif command == "terminate":
        client = await Client.connect("localhost:7233")
        handle = client.get_workflow_handle(WORKFLOW_ID)
        await handle.terminate("Manually terminated")
        logger.info(f"Workflow {WORKFLOW_ID} terminated.")
    else:
        logger.error(f"Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())
