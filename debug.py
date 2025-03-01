import asyncio
import logging
import sys
from datetime import datetime

from temporalio.client import Client
from temporalio.common import RetryPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log"),
    ]
)

logger = logging.getLogger(__name__)

async def check_workflow_status():
    """Check the status of the running workflow."""
    try:
        # Connect to Temporal server
        client = await Client.connect("localhost:7233")
        logger.info(f"Connected to Temporal server")
        
        # Try to get the workflow handle
        handle = client.get_workflow_handle("patent-negation-analysis")
        
        # Get workflow description
        desc = await handle.describe()
        
        logger.info(f"Workflow ID: {desc.id}")
        logger.info(f"Run ID: {desc.run_id}")
        logger.info(f"Status: {desc.status}")
        logger.info(f"Started: {desc.start_time}")
        logger.info(f"Execution time: {datetime.now() - desc.start_time}")
        
        # Check for pending activities
        response = await client.workflow_service.get_workflow_execution_history(
            workflow_id=desc.id,
            run_id=desc.run_id
        )
        
        logger.info(f"History events: {len(response.history.events)}")
        
        # Analyze last few events
        last_events = response.history.events[-10:] if len(response.history.events) >= 10 else response.history.events
        logger.info("Last events:")
        for event in last_events:
            logger.info(f"  {event.event_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking workflow status: {str(e)}")
        return False

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
    if len(sys.argv) < 2:
        print("Usage: python debug.py [check|worker]")
        return
    
    command = sys.argv[1]
    
    if command == "check":
        await check_workflow_status()
    elif command == "worker":
        await run_worker_debug()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())
