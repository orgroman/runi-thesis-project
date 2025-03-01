import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from temporalio.client import Client
from workflow_hotfix import PatentNegationAnalysisHotfixWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hotfix_workflow.log"),
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Run the hotfixed patent negation analysis workflow."""
    if len(sys.argv) < 2:
        print("Usage: python run_hotfix.py <csv_path> [batch_size]")
        sys.exit(1)
    
    # Parse command line arguments
    csv_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info(f"Connected to Temporal server")
    
    # Create a workflow ID based on the CSV filename and timestamp
    csv_filename = Path(csv_path).stem
    workflow_id = f"patent-negation-hotfix-{csv_filename}-{int(datetime.now().timestamp())}"
    
    # Start the workflow
    logger.info(f"Starting hotfix workflow {workflow_id} for {csv_path}")
    handle = await client.start_workflow(
        PatentNegationAnalysisHotfixWorkflow.run,
        args=[csv_path, batch_size],
        id=workflow_id,
        task_queue="patent-negation-task-queue"
    )
    
    logger.info(f"Workflow started with ID: {workflow_id}")
    logger.info(f"Monitor progress at http://localhost:8080/namespaces/default/workflows/{workflow_id}")
    
    # Wait for workflow to complete
    try:
        result = await handle.result()
        logger.info(f"Workflow completed with status: {result['status']}")
        
        if result['status'] == 'completed':
            logger.info(f"Processed {result.get('files_uploaded', 0)} files")
        else:
            logger.error(f"Workflow failed: {result.get('error', 'unknown error')}")
    except Exception as e:
        logger.error(f"Error waiting for workflow result: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
