import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from temporalio.client import Client
from workflow_fix import PatentNegationAnalysisFixedWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fixed_workflow.log"),
    ]
)

logger = logging.getLogger(__name__)

async def terminate_existing_workflow():
    """Terminate the existing problematic workflow."""
    try:
        client = await Client.connect("localhost:7233")
        handle = client.get_workflow_handle("patent-negation-analysis")
        await handle.terminate("Terminating due to activity registration issues")
        logger.info("Successfully terminated existing workflow")
    except Exception as e:
        logger.error(f"Could not terminate existing workflow: {str(e)}")

async def main():
    """Run the fixed patent negation analysis workflow."""
    if len(sys.argv) < 2:
        print("Usage: python run_fixed.py <csv_path> [batch_size]")
        return
    
    # Parse command line arguments
    csv_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info(f"Connected to Temporal server")
    
    # Create a workflow ID based on the CSV filename and timestamp
    csv_filename = Path(csv_path).stem
    workflow_id = f"patent-negation-fixed-{csv_filename}-{int(datetime.now().timestamp())}"
    
    # Start the workflow
    logger.info(f"Starting fixed workflow {workflow_id} for {csv_path}")
    handle = await client.start_workflow(
        PatentNegationAnalysisFixedWorkflow.run,
        args=[csv_path, batch_size],
        id=workflow_id,
        task_queue="patent-negation-task-queue"
    )
    
    logger.info(f"Workflow started with ID: {workflow_id}")
    logger.info(f"Monitor progress at http://localhost:8080/namespaces/default/workflows/{workflow_id}")
    
    # Wait for workflow to complete
    result = await handle.result()
    logger.info(f"Workflow completed with status: {result['status']}")
    
    if result['status'] == 'completed':
        logger.info(f"Processed {result.get('files_uploaded', 0)} files")
    else:
        logger.error(f"Workflow failed: {result.get('error', 'unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
